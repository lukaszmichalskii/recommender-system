import numpy as np


def normalize_ratings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).
    Only include real ratings R(i,j)=1.
    [Y_norm, Y_mean] = normalize_ratings(Y, R) normalized Y so that each movie
    has a rating of 0 on average. Unrated moves then have a mean rating (0)
    Args:
        Y: ratings matrix
        R:
    Returns:
        Normalized ratings and mean rating
    """
    Y_mean = (np.sum(Y * R, axis=1) / (np.sum(R, axis=1) + 1e-12)).reshape(-1, 1)
    Y_norm = Y - np.multiply(Y_mean, R)
    return Y_norm, Y_mean
