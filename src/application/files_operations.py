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
    lock.acquire()
    counter = 0
    iter = 0
    try:
        with open(file, "w") as fd:
            fd.write("Predicted rating,Movie\n")
            while counter != limit:
                j = recommendations[iter]
                if j not in rated:
                    fd.write(f"{predictions[j]:0.2f},{movies_ls[j]}\n")
                    counter += 1
                iter += 1
    finally:
        lock.release()


def save_model_evaluation(file, predictions, ratings, movies_ls, lock):
    lock.acquire()
    try:
        with open(file, "w") as fd:
            fd.write("Original prediction,Predicted rating,Movie\n")
            for i in range(len(ratings)):
                if ratings[i] > 0:
                    fd.write(f"{ratings[i]},{predictions[i]:0.2f},{movies_ls[i]}\n")
    finally:
        lock.release()
