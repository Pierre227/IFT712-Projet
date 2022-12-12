from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.model_selection import GridSearchCV
import numpy as np

class GradientBoostingClassifier:
    """Implementation of GradientBoostingClassifier"""
    
    def __init__(self):
        """Initialize the classifier"""
        param_grid = {"loss": ["log_loss"],
            "learning_rate": [0.1],
            "subsample": [1.0]}
        self.clf = GridSearchCV(gbc(), param_grid, cv=3)

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