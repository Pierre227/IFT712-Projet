from sklearn.linear_model import SGDClassifier as sgd
from sklearn.model_selection import GridSearchCV

class PerceptronClassifier:
    """Implementation of SGDClassifier"""
    
    def __init__(self, random_state, tol):
        """Initialize the classifier"""
        self.clf= sgd(random_state=random_state,tol=tol) # random_state=1,tol=1e-3

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