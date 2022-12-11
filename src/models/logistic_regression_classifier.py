from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import GridSearchCV

class LogisticRegressionClassifier:
    """Implementation of LogisticRegression"""
    
    def __init__(self, random_state):
        """Initialize the classifier"""
        self.lr= lr(random_state=random_state) #random_state=1

    def train(self, X, y):
        """Training of the classifier"""
        self.lr.fit(X, y)

    def predict(self, X):
        """Prediction of the classifier"""
        return self.lr.predict(X)

    def score(self, X, y):
        """Score of the classifier"""
        return self.lr.score(X, y)
    
    def gridSearch(self, X, y, params):
        """Exhaustive search over specified parameter values for the classifier"""
        return 0