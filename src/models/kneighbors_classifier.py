from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import GridSearchCV

class KneighborsClassifier:
    """Implementation of KNN"""

    def __init__(self, n_neighbors):
        """Initialize the classifier"""
        self.clf=knn(n_neighbors=n_neighbors) #n_neighbors=3

    def train(self, X, y):
        """Training of the classifier"""
        self.clf.fit(X, y)

    def predict(self, X):
        """Prediction of the classifier"""
        return self.clf.predict(X)

    def score(self, X, y):
        """Score of the classifier"""
        return self.clf.score(X, y)
    
    def gridSearch(self, X, y, params):
        """Exhaustive search over specified parameter values for the classifier"""
        return 0