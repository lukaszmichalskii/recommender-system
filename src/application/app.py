import argparse
import logging
import os.path
import pathlib
import sys
import threading
import time
import typing

from analysis.analyze import analyze
from application import common, log
from application.common import (
    STEPS,
    STEPS_CHOICES,
    STANDARD_STEPS,
    SUPPORTED_FORMAT,
)
from application.files_operations import (
    get_ratings,
    MalformedFileFormat,
    save_recommendations,
    save_model_evaluation,
    save_model_learn_history,
    get_model_evaluation,
    get_model_learn_history,
    get_recommendations,
)
from application.google_find import google_search
from collaborative_filtering.cf_recommender import CFRecommender
from collaborative_filtering.cf_utils import load_movies


class ThreadExt(threading.Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None
    ):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return


threads_pool = []


def get_help_epilog():
    return """
Exit codes:
    0 - successful execution
    1 - known error
    any other code indicated unrecoverable error - recommendations and statistics might be invalid

Environment variables:
    RECOMMENDATIONS_LIMIT: Specifies how much recommendation should be presented
                           Default: 10
    PRECISION            : Recommendations predictions metric. Enable to customize model performance.
                           High PRECISION value provide more accurate recommendations but require longer execution time.
                           Low PRECISION value provide less accurate recommendations with faster execution time.
                           Default precision was selected based on a lot of analysis and optimal value was determined  
                           Default: 200
    CPU_THREADS          : Specifies how many cores should be utilized during application execution.
                           Default: Use hyper-threading by count physical cores and threads that could be executed in 
                           parallel on single core
                 
Examples:
    Run system:
    python3 app_runner.py --ratings ratings_file.csv
    Collect statistics from run:
    python3 app_runner.py --ratings ratings_file.csv --output results_dir
    Movies information only:
    python3 app_runner.py --ratings ratings_file.csv

More info: <https://github.com/lukaszmichalskii/recommender-system>"""


def threads_join():
    if len(threading.enumerate()) > 1:
        for thread, _ in threads_pool:
            if thread in threading.enumerate():
                thread.join()


def sigint_dcrt(runner):
    def sigint_exit_proc(*args, **kwargs):
        start = time.time()
        try:
            exit_code = runner(*args, **kwargs)
            threads_join()
            end = time.time()
            logging.getLogger("CFRS").info(f"Execution time: {end-start:.2f}")
            return exit_code
        except KeyboardInterrupt:
            logging.getLogger("CFRS").error(
                f"SIGINT interruption, finishing IO non daemon threads..."
            )
            threads_join()
            end = time.time()
            logging.getLogger("CFRS").info(f"Execution time: {end - start:.2f}")
            logging.getLogger("CFRS").info(f"App finished with exit code 3")
            return 3

    return sigint_exit_proc


def run_app(
    args: argparse.Namespace,
    argv: typing.List[str],
    logger: logging.Logger,
    environment: common.Environment,
) -> int:
    def recommend_step():
        logger.info(f"CFR engine starting...")
        debug = False
        if args.verbose:
            debug = True
        start_rs = time.time()
        recommendations, predictions, history = cf_recommender.recommend(
            ratings, environment.precision, debug=debug
        )
        end_rs = time.time()
        logger.info(f"Recommendation system execution time: {end_rs - start_rs:.2f}")

        rcmd_thread = threading.Thread(
            target=save_recommendations,
            args=(
                rcmd_file,
                predictions,
                recommendations,
                cf_recommender.rated,
                movies_list,
                environment.recommendations_limit
                if environment.recommendations_limit < len(movies_list)
                else len(movies_list),
                rcmd_filelock,
            ),
        )

        mdeval_thread = threading.Thread(
            target=save_model_evaluation,
            args=(
                mdeval_file,
                predictions,
                cf_recommender.ratings,
                movies_list,
                mdeval_filelock,
            ),
        )

        mdlearn_hist_thread = threading.Thread(
            target=save_model_learn_history,
            args=(mdhist_file, history, mdhist_filelock),
        )

        threads_pool.append((rcmd_thread, "rcmd_thread"))
        threads_pool.append((mdeval_thread, "mdeval_thread"))
        threads_pool.append((mdlearn_hist_thread, "mdlearn_hist_thread"))

        rcmd_thread.start()
        mdeval_thread.start()
        mdlearn_hist_thread.start()

    def analysis_step():
        for thread, thread_id in threads_pool:
            if (
                thread_id == "mdeval_thread" or thread_id == "mdlearn_hist_thread"
            ) and thread.is_alive():
                thread.join()

        mdeval_data = get_model_evaluation(mdeval_file, mdeval_filelock)
        history = get_model_learn_history(mdhist_file, mdhist_filelock)

        analysis_thread = threading.Thread(
            target=analyze, args=(mdeval_data, history, output)
        )
        threads_pool.append((analysis_thread, "analysis_thread"))
        analysis_thread.start()

    def find_step():
        for thread, thread_id in threads_pool:
            if thread_id == "rcmd_thread" and thread.is_alive():
                thread.join()

        rcmd_data = get_recommendations(rcmd_file, rcmd_filelock)

        semaphore = threading.BoundedSemaphore(value=environment.cpu)
        info = []
        call_threads_pool = []
        for i, movie in enumerate(rcmd_data):
            call_thread = ThreadExt(
                target=google_search,
                args=(f"{movie}, Movie", semaphore, environment.api_key),
            )
            call_threads_pool.append((call_thread, movie))
            call_thread.start()

        for thread, movie in call_threads_pool:
            url = thread.join()
            info.append({"movie": movie, "url": url})

        for i in info:
            print(f"Movie: {i.get('movie')}, URL: {i.get('url')}")

    start = time.time()
    try:
        if environment.os != "linux":
            logger.warning(
                f"You are using toolkit on {environment.os}. Some functionalities may not work correctly"
            )
        logger.info(
            f"pythonApp: {sys.executable} argv: {argv} {environment.to_info_string()}"
        )
        if os.path.exists(args.output) and os.listdir(args.output):
            logger.error(f"Output directory {args.output} is not empty.")
            logger.info("App finished with exit code 1")
            return 1

        output = pathlib.Path(args.output)
        if not output.exists():
            logger.info(
                f"Path '{str(output.absolute())}' not exist, creating results storage space {output.absolute()}..."
            )
            output.mkdir()

        ratings_file = pathlib.Path(args.ratings)
        if not ratings_file.exists():
            logger.info(f"Ratings file '{str(ratings_file)}' not exist.")
            logger.info("App finished with exit code 1")
            return 1
        if ratings_file.suffix != SUPPORTED_FORMAT:
            logger.error(
                f"Unsupported ratings file format, required {SUPPORTED_FORMAT} files only."
            )
            logger.info("App finished with exit code 1")
            return 1

        try:
            ratings = get_ratings(ratings_file)
        except MalformedFileFormat as e:
            logger.error(f"Error during user ratings reading. Details: {str(e)}")
            logger.info("App finished with exit code 1")
            print(
                "Provided ratings file does not follow required format for recommendation engine calculations."
            )
            return 1

        logger.info("Loading movies dataset...")
        movies_list, movies_df = load_movies()
        logger.info(
            f"Dataset loaded successfully: {len(movies_list)} movies available."
        )

        logger.info("Initialize collaboration filtering recommendation engine...")
        cf_recommender = CFRecommender()

        rcmd_file = output.joinpath("recommendations.csv")
        mdeval_file = output.joinpath("model_evaluation.csv")
        mdhist_file = output.joinpath("cf_learn.csv")

        # mutexes
        rcmd_filelock = threading.Lock()
        mdeval_filelock = threading.Lock()
        mdhist_filelock = threading.Lock()

        if STEPS.RECOMMEND in args.only:
            recommend_step()

        if STEPS.ANALYSIS in args.only:
            logger.info("Starting recommendation analysis...")
            analysis_step()

        if STEPS.FIND in args.only:
            if environment.api_key is None:
                logger.warning("API_KEY not provided skipping FIND step.")
            else:
                find_step()

        logger.info("Finishing IO bound threads...")
        threads_join()
        end = time.time()
        logger.info(f"Execution time: {end - start:.2f}")
        logger.info("App finished with exit code 0")
        return 0

    except KeyboardInterrupt:
        logger.error(f"SIGINT interruption, finishing IO non daemon threads...")
        threads_join()
        end = time.time()
        logger.info(f"Execution time: {end - start:.2f}")
        logger.info(f"App finished with exit code 3")
        return 3


def main(argv: typing.List[str], logger=None, environment=None) -> int:
    if logger is None:
        logger = log.setup_logger()
    if environment is None:
        environment = common.Environment.from_env(os.environ)
    parser = argparse.ArgumentParser(
        description="Recommender System - machine learning based movies recommendation system.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-r",
        "--ratings",
        type=str,
        required=True,
        metavar="input_data",
        help="""specifies file, where user movies ratings are stored. Only .csv files supported, has to follow specific
format""",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="output_folder",
        default="results",
        help="specifies directory, where results should be saved. Has to be empty",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="*",
        choices=STEPS_CHOICES,
        default=STANDARD_STEPS,
        help="""specifies actions which should be performed on input data:
    'recommend' - find recommendations based on ratings pointed by --ratings
    'find' - find information about recommendations
    'similar' - find top matching movies to best rated one
    """,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display extended information about system execution.",
    )
    parser.epilog = get_help_epilog()
    return run_app(parser.parse_args(argv[1:]), argv, logger, environment)
