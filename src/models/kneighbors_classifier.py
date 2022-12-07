from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import GridSearchCV

class KneighborsClassifier:
    """Implementation of KNN"""

    def __init__(self, n_neighbors):
        """Initialize the classifier"""
        self.knn=KNeighborsClassifier(n_neighbors)

    def train(self, X, y):
        """Training of the classifier"""
        self.knn.fit(X, y)

    def predict(self, X):
        """Prediction of the classifier"""
        return self.knn.predict(X)

    def score(self, X, y):
        """Score of the classifier"""
        return self.knn.score(X, y)
    
    def gridSearch(self, X, y, params):
        """Exhaustive search over specified parameter values for the classifier"""
        return 0