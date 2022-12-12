from sklearn.neural_network import MLPClassifier as mlp
from sklearn.model_selection import GridSearchCV
import numpy as np

class MLPClassifier:
    """Implementation of MLPClassifier"""
    
    def __init__(self):
        """Initialize the classifier"""
        param_grid = {'alpha': 10.0 ** -np.arange(1, 6),
                'hidden_layer_sizes': [(99,), (99, 99), (99, 99, 99)]}
        self.clf = GridSearchCV(mlp(max_iter=1000), param_grid, cv=3)

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