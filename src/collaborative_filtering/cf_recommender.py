import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.collaborative_filtering.cf_utils import load_ratings
from src.collaborative_filtering.cost_function import cf_cost_func
from src.collaborative_filtering.normalization import normalize_ratings

# tf.random.set_seed(1234)


class CFRecommender:
    def __init__(self, logger=None):
        self.logger = logger
        self.Y, self.R = load_ratings()
        self.num_movies, self.num_users = self.Y.shape
        self.num_features = 100

        self.ratings = None
        self.rated = None

        self.Y_norm, self.Y_mean = None, None

    def recommend(self, ratings, iterations, debug=False):
        self.collect_ratings(ratings)
        self.update_ratings()
        X, W, b, history = self.cf_learn(iterations=iterations, lambda_=1, debug=debug)
        predictions = self.predict(X, W, b)
        ix = tf.argsort(predictions, direction="DESCENDING")
        return ix, predictions, history

    def collect_ratings(self, user_ratings):
        self.ratings = np.zeros(self.num_movies)
        for user_rating in user_ratings:
            idx, rating = user_rating
            self.ratings[idx] = rating
        self.rated = [i for i in range(len(self.ratings)) if self.ratings[i] > 0]

    def cf_learn(self, iterations=200, lambda_=1, debug=False):
        num_movies, num_users = self.Y.shape
        num_features = 100
        W = tf.Variable(
            tf.random.normal((num_users, num_features), dtype=tf.float64), name="W"
        )
        X = tf.Variable(
            tf.random.normal((num_movies, num_features), dtype=tf.float64), name="X"
        )
        b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name="b")
        optimizer = keras.optimizers.Adam(learning_rate=1e-1)

        history = []
        for iter in range(1, iterations + 1):
            with tf.GradientTape() as tape:
                cost_value = cf_cost_func(X, W, b, self.Y_norm, self.R, lambda_)
            grads = tape.gradient(cost_value, [X, W, b])
            optimizer.apply_gradients(zip(grads, [X, W, b]))

            if iter % 20 == 0 or iter == 1:
                if debug and self.logger:
                    self.logger.info(
                        f"Training loss at iteration {iter}: {cost_value:0.1f}"
                    )
                history.append((iter, cost_value))

        return X, W, b, history

    def predict(self, X, W, b):
        predictions = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
        predictions = predictions + self.Y_mean
        return predictions[:, 0]

    def update_ratings(self):
        self.Y = np.c_[self.ratings, self.Y]
        self.R = np.c_[(self.ratings != 0).astype(int), self.R]
        self.Y_norm, self.Y_mean = normalize_ratings(self.Y, self.R)
