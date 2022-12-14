from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
import numpy as np

class PerceptronClassifier:
    """Implementation of Perceptron"""
    
    def __init__(self):
        """Initialize the classifier"""
        param_grid = {'alpha': 10.0 ** -np.arange(1, 6)}
        self.clf = GridSearchCV(Perceptron(), param_grid, cv=3)

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