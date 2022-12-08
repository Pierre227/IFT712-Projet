import pandas as pd
import sklearn.utils

class DataOrganizer:

    def __init__(self):
        """Initialize the path to the directory of the project"""
        self.test=0

    def shuffle(self, dataset, columns):
        sklearn.utils.shuffle(dataset[columns])

    def standardize(self, dataset, columns):
        for column in columns:
            dataset[column] = (dataset[column] - dataset[column].mean()) / dataset[column].std()
    
    # def numerize(self, dataset, column):
        