import pandas as pd


class MalformedFileFormat(Exception):
    pass


def get_ratings(file):
    ratings = []
    with open(file, "r") as fd:
        try:
            raw = fd.readline().strip()
            while raw:
                movie_id, rating, *_ = raw.split(",")
                ratings.append((int(movie_id), int(rating)))
                raw = fd.readline().strip()
        except Exception as e:
            raise MalformedFileFormat(
                f"File does not follow ratings convention: {str(e)}"
            )
    return ratings


def save_recommendations(
    file, predictions, recommendations, rated, movies_ls, limit, lock
):
    with lock:
        counter = 0
        iter = 0
        with open(file, "w") as fd:
            fd.write("Predicted rating,Movie\n")
            while counter != limit:
                j = recommendations[iter]
                if j not in rated:
                    record = (
                        f"{predictions[j]:0.2f},{movies_ls[j]}\n"
                        if "," not in movies_ls[j]
                        else f'{predictions[j]:0.2f},"{movies_ls[j]}"\n'
                    )
                    fd.write(record)
                    counter += 1
                iter += 1


def save_model_evaluation(file, predictions, ratings, movies_ls, lock):
    with lock:
        with open(file, "w") as fd:
            fd.write("Original prediction,Predicted rating,Movie\n")
            for i in range(len(ratings)):
                if ratings[i] > 0:
                    record = (
                        f"{ratings[i]},{predictions[i]:0.2f},{movies_ls[i]}\n"
                        if "," not in movies_ls[i]
                        else f'{ratings[i]},{predictions[i]:0.2f},"{movies_ls[i]}"\n'
                    )
                    fd.write(record)


def save_model_learn_history(file, history, lock):
    with lock:
        with open(file, "w") as fd:
            fd.write("Iteration,Loss\n")
            for iter, loss in history:
                fd.write(f"{iter},{int(loss)}\n")


def get_model_evaluation(file, lock):
    with lock:
        mdeval_data = []
        with open(file, "r") as fd:
            _ = fd.readline()  # header
            data = fd.readline().strip()
            while data:
                original, predicted, *other = data.split(",")
                mdeval_data.append((float(original), float(predicted)))
                data = fd.readline().strip()
        return mdeval_data


def get_model_learn_history(file, lock):
    with lock:
        history = []
        with open(file, "r") as fd:
            _ = fd.readline()  # header
            data = fd.readline().strip()
            while data:
                iteration, loss = data.split(",")
                history.append((int(iteration), int(loss)))
                data = fd.readline().strip()
        return history


def get_recommendations(file, lock):
    with lock:
        df = pd.read_csv(
            file,
            header=0,
            index_col=0,
            delimiter=",",
            quotechar='"',
        )
        return df["Movie"].to_list()
