import pandas as pd
import sklearn.utils

class DataOrganizer:

    def shuffle(self, dataset, columns):
        """Randomly shufflle the dataset"""
        return sklearn.utils.shuffle(dataset[columns], random_state=1)

    def standardize(self, dataset, columns):
        """Standardize all given columns"""
        for column in columns:
            dataset[column] = (dataset[column] - dataset[column].mean()) / dataset[column].std()
    
    def numerize(self, dataset, column):
        """Numerize the attributes of a given column"""
        previous_number=-1
        values = dict()
        for i, value in enumerate(dataset[column]): #For each value of the selected column
            new_value = 0
            if value in values: #If the value is already in the dict
                new_value = values[value] #Assign the already defined numeric value
            else: #If the value is not in the dict
                previous_number += 1
                # Assign a new numeric value
                new_value = previous_number
                values[value] = new_value

            dataset.loc[i, column] = new_value
        dataset[column] = dataset[column].astype(int)