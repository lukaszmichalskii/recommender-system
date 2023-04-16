class CollaborativeFilteringRecommender:
    def __init__(self, X, W, b, num_movies, num_features, num_users):
        self.X = X
        self.W = W
        self.b = b
        self.num_movies = num_movies
        self.num_features = num_features
        self.num_users = num_users
