from sklearn.neural_network import MLPClassifier as mlp
from sklearn.model_selection import GridSearchCV

class MLPClassifier:
    """Implementation of MLPClassifier"""
    
    def __init__(self, random_state, max_iter):
        """Initialize the classifier"""
        self.clf= mlp(random_state=random_state, max_iter=max_iter) # random_state=1, max_iter=300

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