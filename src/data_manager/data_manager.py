import os
import pandas as pd

class DatabaseManager:

    def __init__(self, path):
        """Initialize the path to the directory of the project"""
        self.directory = path
    
    def load(self):
        """Loads the content into dataframes"""
        train_dataframe = pd.read_csv(os.path.join(self.directory, '../data/raw/train.csv'))
        test_dataframe = pd.read_csv(os.path.join(self.directory, '../data/raw/test.csv'))
        self.train_dataset = train_dataframe
        self.test_dataset = test_dataframe
    
    def get_test_dataset(self):
        """Returns the test dataset"""
        return self.test_dataset
    
    def get_train_dataset(self):
        """Returns the train dataset"""
        return self.train_dataset