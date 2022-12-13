from sklearn import svm
from sklearn.model_selection import GridSearchCV

class SVCClassifier:
    """Implementation of SVCClassifier"""
    
    def __init__(self):
        """Initialize the classifier"""
        param_grid = {'C': [0.1, 1, 10],
                    'loss': ['hinge', 'squared_hinge']}
        self.clf = GridSearchCV(svm.LinearSVC(max_iter=3000), param_grid, cv=3)

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