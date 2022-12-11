from sklearn import svm
from sklearn.model_selection import GridSearchCV

class SVMClassifier:
    """Implementation of SVMClassifier"""
    
    def __init__(self,random_state,gamma):
        """Initialize the classifier"""
        self.clf= SVC(random_state=random_state, gamma=gamma) # random_state=1, gamma='auto'

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