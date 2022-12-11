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
        values_dict = dict()
        last_number_used=-1
        for i, value in enumerate(dataset[column]):
            numeric_replacement = 0

            # Check if the dictionnary has the value already.
            if value in values_dict:
                numeric_replacement = values_dict[value]
            else:
                last_number_used += 1
                # Set as the numeric replacement for the current string.
                numeric_replacement = last_number_used
                # Register in the dictionnary.
                values_dict[value] = numeric_replacement

            # Replace the string by its numeric value.
            dataset.loc[i, column] = numeric_replacement
        dataset[column] = dataset[column].astype(int)