import numpy as np

from collaborative_filtering.recsys_utils import load_precalc, load_ratings


def predict():
    pass


if __name__ == "__main__":
    X, W, b, nm, nf, nu = load_precalc()
    Y, R = load_ratings()
    print("Y", Y.shape, "R", R.shape)
    print("X", X.shape)
    print("W", W.shape)
    print("b", b.shape)
    print("num_features", nf)
    print("num_movies", nm)
    print("num_users", nu)

    tsmean = np.mean(Y[0, R[0, :].astype(bool)])
    print(f"Average rating for movie 1 : {tsmean:0.3f} / 5")
