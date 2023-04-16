import numpy as np
import tensorflow as tf


def cf_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the collaborative filtering model, vectorized implementation
    Args:
        X (ndarray (num_movies,num_features)): matrix of item features
        W (ndarray (num_users,num_features)) : matrix of user parameters
        b (ndarray (1, num_users)            : vector of user parameters
        Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
        R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
        lambda_ (float): regularization parameter
    Returns:
        J (float) : cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_ / 2) * (
        tf.reduce_sum(X**2) + tf.reduce_sum(W**2)
    )
    return J


def cf_cost_func2(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the collaborative filtering model
    Args:
        X (ndarray (num_movies,num_features)): matrix of item features
        W (ndarray (num_users,num_features)) : matrix of user parameters
        b (ndarray (1, num_users)            : vector of user parameters
        Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
        R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
        lambda_ (float): regularization parameter
    Returns:
        J (float) : cost
    """
    nm, nu = Y.shape
    J = 0

    for j in range(nu):
        for i in range(nm):
            J += R[i, j] * (np.dot(W[j], X[i]) + b[0, j] - Y[i, j]) ** 2
    J = J / 2
    nf = X.shape[1]

    # regularization term
    w_reg = 0
    for j in range(nu):
        for k in range(nf):
            w_reg += W[j, k] ** 2

    x_reg = 0
    for i in range(nm):
        for k in range(nf):
            x_reg += X[i, k] ** 2

    J += lambda_ / 2 * (w_reg + x_reg)
    return J
