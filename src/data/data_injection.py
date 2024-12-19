import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import dataclasses
from src.logger import logging
from src.exception import CustomException
from src.utils import DataHandler

@dataclasses.dataclass
class DatainjectionConfig:
    raw_data_path: str = 'data/external/ML_Intern_Assignment_Data.csv'
    train_data_path: str = 'data/raw/train.csv'
    test_data_path: str = 'data/raw/test.csv'


class DataInjection:
    def __init__(self):
        self.data_injection_config = DatainjectionConfig()
        self.data_handler = DataHandler()

    def data_injection(self, data_path):
        try:
            df = pd.read_csv(data_path)
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            self.data_handler.save_data(train_data, self.data_injection_config.train_data_path)
            self.data_handler.save_data(test_data, self.data_injection_config.test_data_path)
            return train_data, test_data
        except Exception as e:
            logging.info(
                'Unexpected error occurred while performing data injection: %s', e)
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_config = DatainjectionConfig()
    object = DataInjection()
    object.data_injection(data_config.raw_data_path)
