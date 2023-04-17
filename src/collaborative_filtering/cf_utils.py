import pandas as pd

from config import ROOT
from numpy import loadtxt

RESOURCES = ROOT.joinpath("docs/data")


def load_precalc():
    X = loadtxt(RESOURCES.joinpath("small_movies_X.csv"), delimiter=",")
    W = loadtxt(RESOURCES.joinpath("small_movies_W.csv"), delimiter=",")
    b = loadtxt(RESOURCES.joinpath("small_movies_b.csv"), delimiter=",")
    b = b.reshape(1, -1)
    num_movies, num_features = X.shape
    num_users, _ = W.shape
    return X, W, b, num_movies, num_features, num_users


def load_ratings():
    Y = loadtxt(RESOURCES.joinpath("small_movies_Y.csv"), delimiter=",")
    R = loadtxt(RESOURCES.joinpath("small_movies_R.csv"), delimiter=",")
    return Y, R


def load_movies():
    """
    Returns:
        returns df (pandas.DataFrame) with and index of movies in the order they are in the ratings matrix (Y)
    """
    df = pd.read_csv(
        RESOURCES.joinpath("small_movie_list.csv"),
        header=0,
        index_col=0,
        delimiter=",",
        quotechar='"',
    )
    movies = df["title"].to_list()
    return movies, df
