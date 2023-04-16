import unittest

import numpy as np

from src.collaborative_filtering.cost_function import cf_cost_func, cf_cost_func2


class TestCostFunction(unittest.TestCase):
    def setUp(self) -> None:
        self.num_users_r = 4
        self.num_movies_r = 5
        self.num_features_r = 3

    def test_cf_cost_func(self):
        # num_users_r = 3
        # num_movies_r = 4
        # num_features_r = 4
        # np.random.seed(247)
        X_r = np.array(
            [
                [0.36618032, 0.9075415, 0.8310605, 0.08590986],
                [0.62634721, 0.38234325, 0.85624346, 0.55183039],
                [0.77458727, 0.35704147, 0.31003294, 0.20100006],
                [0.34420469, 0.46103436, 0.88638208, 0.36175401],
            ]
        )  # np.random.rand(num_movies_r, num_features_r)
        W_r = np.array(
            [
                [0.04786854, 0.61504665, 0.06633146, 0.38298908],
                [0.16515965, 0.22320207, 0.89826005, 0.14373251],
                [0.1274051, 0.22757303, 0.96865613, 0.70741111],
            ]
        )  # np.random.rand(num_users_r, num_features_r)
        b_r = np.array(
            [[0.14246472, 0.30110933, 0.56141144]]
        )  # np.random.rand(1, num_users_r)
        Y_r = np.array(
            [
                [0.20651685, 0.60767914, 0.86344527],
                [0.82665019, 0.00944765, 0.4376798],
                [0.81623732, 0.26776794, 0.03757507],
                [0.37232161, 0.19890823, 0.13026598],
            ]
        )  # np.random.rand(num_movies_r, num_users_r)
        R_r = np.array(
            [[1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]]
        )  # (np.random.rand(num_movies_r, num_users_r) > 0.4) * 1
        J = cf_cost_func(X_r, W_r, b_r, Y_r, R_r, 3)
        self.assertTrue(np.isclose(J, 13.621929978531858, atol=1e-8))

    def test_cf_cost_func_hc(self):
        X_r = np.ones((self.num_movies_r, self.num_features_r))
        W_r = np.ones((self.num_users_r, self.num_features_r))
        b_r = np.zeros((1, self.num_users_r))
        Y_r = np.zeros((self.num_movies_r, self.num_users_r))
        R_r = np.zeros((self.num_movies_r, self.num_users_r))

        J = cf_cost_func(X_r, W_r, b_r, Y_r, R_r, 2)
        self.assertTrue(np.isclose(J, 27))

        X_r = np.ones((self.num_movies_r, self.num_features_r))
        W_r = np.ones((self.num_users_r, self.num_features_r))
        b_r = np.ones((1, self.num_users_r))
        Y_r = np.ones((self.num_movies_r, self.num_users_r))
        R_r = np.ones((self.num_movies_r, self.num_users_r))

        # Evaluate cost function
        J = cf_cost_func(X_r, W_r, b_r, Y_r, R_r, 1)
        self.assertTrue(np.isclose(J, 103.5))

    def test_cf_cost_func_without_regularization(self):
        X_r = np.ones((self.num_movies_r, self.num_features_r))
        W_r = np.ones((self.num_users_r, self.num_features_r))
        b_r = np.ones((1, self.num_users_r))
        Y_r = np.ones((self.num_movies_r, self.num_users_r))
        R_r = np.ones((self.num_movies_r, self.num_users_r))

        J = cf_cost_func(X_r, W_r, b_r, Y_r, R_r, 0)
        self.assertTrue(np.isclose(J, 90))

        X_r = np.ones((self.num_movies_r, self.num_features_r))
        W_r = np.ones((self.num_users_r, self.num_features_r))
        b_r = np.ones((1, self.num_users_r))
        Y_r = np.zeros((self.num_movies_r, self.num_users_r))
        R_r = np.ones((self.num_movies_r, self.num_users_r))

        J = cf_cost_func(X_r, W_r, b_r, Y_r, R_r, 0)
        self.assertTrue(np.isclose(J, 160))

    def test_cf_cost2_func(self):
        # num_users_r = 3
        # num_movies_r = 4
        # num_features_r = 4

        # np.random.seed(247)
        X_r = np.array(
            [
                [0.36618032, 0.9075415, 0.8310605, 0.08590986],
                [0.62634721, 0.38234325, 0.85624346, 0.55183039],
                [0.77458727, 0.35704147, 0.31003294, 0.20100006],
                [0.34420469, 0.46103436, 0.88638208, 0.36175401],
            ]
        )  # np.random.rand(num_movies_r, num_features_r)
        W_r = np.array(
            [
                [0.04786854, 0.61504665, 0.06633146, 0.38298908],
                [0.16515965, 0.22320207, 0.89826005, 0.14373251],
                [0.1274051, 0.22757303, 0.96865613, 0.70741111],
            ]
        )  # np.random.rand(num_users_r, num_features_r)
        b_r = np.array(
            [[0.14246472, 0.30110933, 0.56141144]]
        )  # np.random.rand(1, num_users_r)
        Y_r = np.array(
            [
                [0.20651685, 0.60767914, 0.86344527],
                [0.82665019, 0.00944765, 0.4376798],
                [0.81623732, 0.26776794, 0.03757507],
                [0.37232161, 0.19890823, 0.13026598],
            ]
        )  # np.random.rand(num_movies_r, num_users_r)
        R_r = np.array(
            [[1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]]
        )  # (np.random.rand(num_movies_r, num_users_r) > 0.4) * 1
        J = cf_cost_func2(X_r, W_r, b_r, Y_r, R_r, 3)
        self.assertTrue(np.isclose(J, 13.621929978531858, atol=1e-8))

    def test_cf_cost2_func_hc(self):
        X_r = np.ones((self.num_movies_r, self.num_features_r))
        W_r = np.ones((self.num_users_r, self.num_features_r))
        b_r = np.zeros((1, self.num_users_r))
        Y_r = np.zeros((self.num_movies_r, self.num_users_r))
        R_r = np.zeros((self.num_movies_r, self.num_users_r))

        J = cf_cost_func2(X_r, W_r, b_r, Y_r, R_r, 2)
        self.assertTrue(np.isclose(J, 27))

        X_r = np.ones((self.num_movies_r, self.num_features_r))
        W_r = np.ones((self.num_users_r, self.num_features_r))
        b_r = np.ones((1, self.num_users_r))
        Y_r = np.ones((self.num_movies_r, self.num_users_r))
        R_r = np.ones((self.num_movies_r, self.num_users_r))

        # Evaluate cost function
        J = cf_cost_func2(X_r, W_r, b_r, Y_r, R_r, 1)
        self.assertTrue(np.isclose(J, 103.5))

    def test_cf_cost_func2_without_regularization(self):
        X_r = np.ones((self.num_movies_r, self.num_features_r))
        W_r = np.ones((self.num_users_r, self.num_features_r))
        b_r = np.ones((1, self.num_users_r))
        Y_r = np.ones((self.num_movies_r, self.num_users_r))
        R_r = np.ones((self.num_movies_r, self.num_users_r))

        J = cf_cost_func2(X_r, W_r, b_r, Y_r, R_r, 0)
        self.assertTrue(np.isclose(J, 90))

        X_r = np.ones((self.num_movies_r, self.num_features_r))
        W_r = np.ones((self.num_users_r, self.num_features_r))
        b_r = np.ones((1, self.num_users_r))
        Y_r = np.zeros((self.num_movies_r, self.num_users_r))
        R_r = np.ones((self.num_movies_r, self.num_users_r))

        J = cf_cost_func2(X_r, W_r, b_r, Y_r, R_r, 0)
        self.assertTrue(np.isclose(J, 160))
