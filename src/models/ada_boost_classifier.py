from sklearn.ensemble import AdaBoostClassifier as adab
from sklearn.model_selection import GridSearchCV

class AdaBoostClassifier:
    """Implementation of AdaBoostClassifierr"""
    
    def __init__(self, random_state ,n_estimators):
        """Initialize the classifier"""
        self.clf= adab(random_state=random_state ,n_estimators=n_estimators) # random_state=0,n_estimators=100

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