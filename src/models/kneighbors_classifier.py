from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import GridSearchCV

class KneighborsClassifier:
    """Implementation of KNN"""

    def __init__(self):
        """Initialize the classifier"""
        grid_parameters = {'n_neighbors': range(2, 15)}
        self.clf = GridSearchCV(knn(), grid_parameters, cv=3)

    def train(self, X, y):
        """Training of the classifier"""
        self.clf.fit(X, y)

    def predict(self, X):
        """Prediction of the classifier"""
        return self.clf.predict(X)

    def score(self, X, y):
        """Score of the classifier"""
        return self.clf.score(X, y)

    def get_classifier(self):
        """ Returns the classifier """
        return self.clf