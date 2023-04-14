import argparse
import logging
import os.path
import pathlib
import sys

import typing

from src.application import common, log


def get_help_epilog():
    return """
Exit codes:
    0 - successful execution
    any other code indicated unrecoverable error - recommendations and statistics might be invalid

Environment variables:
                 
Examples:
    Run session:
    python3 app_runner.py
    Collect statistics from session:
    python3 app_runner.py --output results_dir

More info: <https://github.com/lukaszmichalskii/recommender-system>"""


def run_app(
    args: argparse.Namespace,
    argv: typing.List[str],
    logger: logging.Logger,
    environment: common.Environment,
) -> int:
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
            f"Path '{str(output)}' not exists, creating results storage space..."
        )
        output.mkdir()

    logger.info("App finished with exit code 0")
    return 0


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
        "-o",
        "--output",
        type=str,
        metavar="output_folder",
        default="results",
        help="specifies directory, where results should be saved. Has to be empty",
    )
    parser.epilog = get_help_epilog()
    return run_app(parser.parse_args(argv[1:]), argv, logger, environment)
