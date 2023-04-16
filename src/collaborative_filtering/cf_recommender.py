import numpy as np
import tensorflow as tf
from tensorflow import keras

from collaborative_filtering.cf_utils import load_movies, load_ratings, load_precalc
from collaborative_filtering.cost_function import cf_cost_func
from collaborative_filtering.normalization import normalize_ratings


tf.random.set_seed(1234)


class CFRecommender:
    def __init__(self):
        self.Y, self.R = load_ratings()
        self.num_movies, self.num_users = self.Y.shape
        self.num_features = 100
        self.movies_list, self.movies_df = load_movies()

        self.ratings = None
        self.rated = None

        self.Y_norm, self.Y_mean = None, None

    def recommend(self) -> str:
        self.collect_ratings()
        self.update_ratings()
        X, W, b = self.cf_learn(iterations=200, lambda_=1)
        predictions = self.predict(X, W, b)

        # sort predictions
        ix = tf.argsort(predictions, direction='DESCENDING')
        return ix

        # for i in range(17):
        #     j = ix[i]
        #     if j not in self.rated:
        #         print(f'Predicting rating {predictions[j]:0.2f} for movie {self.movies_list[j]}')
        #
        # print('\n\nOriginal vs Predicted ratings:\n')
        # for i in range(len(self.ratings)):
        #     if self.ratings[i] > 0:
        #         print(f'Original {self.ratings[i]}, Predicted {predictions[i]:0.2f} for {self.movies_list[i]}')
    def collect_ratings(self):
        self.ratings = np.zeros(self.num_movies)
        self.ratings[2700] = 5
        self.ratings[2609] = 2
        self.ratings[929] = 5  # Lord of the Rings: The Return of the King, The
        self.ratings[246] = 5  # Shrek (2001)
        self.ratings[2716] = 3  # Inception
        self.ratings[1150] = 5  # Incredibles, The (2004)
        self.ratings[382] = 2  # Amelie (Fabuleux destin d'Amélie Poulain, Le)
        self.ratings[366] = 5  # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
        self.ratings[622] = 5  # Harry Potter and the Chamber of Secrets (2002)
        self.ratings[988] = 3  # Eternal Sunshine of the Spotless Mind (2004)
        self.ratings[2925] = 1  # Louis Theroux: Law & Disorder (2008)
        self.ratings[2937] = 1  # Nothing to Declare (Rien à déclarer)
        self.ratings[793] = 5  # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
        self.rated = [i for i in range(len(self.ratings)) if self.ratings[i] > 0]

    def cf_learn(self, iterations=200, lambda_=1, debug=True):
        num_movies, num_users = self.Y.shape
        num_features = 100
        W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
        X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64), name='X')
        b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')
        optimizer = keras.optimizers.Adam(learning_rate=1e-1)

        for iter in range(iterations):
            with tf.GradientTape() as tape:
                cost_value = cf_cost_func(X, W, b, self.Y_norm, self.R, lambda_)
            grads = tape.gradient(cost_value, [X, W, b])
            optimizer.apply_gradients(zip(grads, [X, W, b]))

            if iter % 20 == 0 and debug:
                print(f"Training loss at iteration {iter}: {cost_value:0.1f}")

        return X, W, b

    def predict(self, X, W, b):
        predictions = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
        predictions = predictions + self.Y_mean
        return predictions[:, 0]

    def update_ratings(self):
        self.Y = np.c_[self.ratings, self.Y]
        self.R = np.c_[(self.ratings != 0).astype(int), self.R]
        self.Y_norm, self.Y_mean = normalize_ratings(self.Y, self.R)


if __name__ == '__main__':
    cf = CFRecommender()
    cf.recommend()
