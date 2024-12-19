import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import joblib



class DataHandler:
    def __init__(self):
        pass
        

    def load_data(self, data_url: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        try:
            df = pd.read_csv(data_url)
            logging.info('Data loaded from %s', data_url)
            # Calculate the percentage of null values in each column
            null_percentage = df.isnull().mean() * 100
            logging.info('Null value percentages:\n%s', null_percentage)
            # Drop rows with null values if they are less than 5%
            if null_percentage.max() < 5:
                df = df.dropna()
                logging.info(
                    'Dropped rows with null values as they were less than 5% .'
                    )
            elif null_percentage.max() == 0:
                logging.info('No missing values in the data.')
            else:
                logging.warning(
                    'Null values exceed 5%, not dropping any rows.'
                    )
            return df
        except Exception as e:
            logging.info(
                'Unexpected error occurred while loading data: %s', e)
            raise CustomException(e, sys)


    def save_data(self, data: pd.DataFrame, file_path: str) -> None:
        """Save the train and test datasets."""
        try:
            
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            data.to_csv(file_path, index=False)
            logging.info(f'Data saved to {file_path}')
            
        except Exception as e:
            logging.error(f'Unexpected error occurred while saving the data: {e}')
            raise CustomException(e, sys)

    def save_object(self, object, file_path: str) :
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as file:
                joblib.dump(object, file)
        except Exception as e:
            logging.info(
                "unexpected error occured while saving object"
            )
            raise CustomException(e, sys)
        
        
    def save_array(self, array, file_path: str) -> None:
        try:

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if isinstance(array, np.ndarray):
                np.save(file_path, array)  
            else:
                raise TypeError("Only NumPy arrays can be saved using this method.")
          
        except Exception as e:
            logging.error(f"Unexpected error occurred while saving array: {e}")
            raise CustomException(e, sys)
    
 
    def load_array(self, file_path: str):
        try:
            
            if file_path.endswith(".npy"):
                return np.load(file_path)  
                raise ValueError("Unsupported file format. Only .npy files are supported.")
        except Exception as e:
            logging.error(f"Unexpected error occurred while loading array: {e}")
            raise CustomException(e, sys)