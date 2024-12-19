import os
import sys
import pandas as pd
import numpy as np
import dataclasses
from src.logger import logging
from src.exception import CustomException
from src.exception import CustomException
from src.utils import DataHandler, Model


@dataclasses.dataclass
class Modelevaultionconfig:
    preprocessed_test_path = 'data/interim/test.csv'  # For model evaluation
    preprocessor_path = 'objects/preprosser/object.pkl'
    model_path = 'objects/model/model.pkl'

class Modelevaultion:
    def __init__(self):
        self.modelevalutionconfig = Modelevaultionconfig()
        self.data_handler = DataHandler()
        self.model = Model()
    def eval_mode(self):
        try:
            test_df = pd.read_csv(self.modelevalutionconfig.preprocessed_test_path)
            x_test = test_df.drop(columns='Churn')
            y_test = test_df['Churn'].map({'Yes': 1, 'No': 0})
            preprocessor = self.data_handler.load_object(file_path = self.modelevalutionconfig.preprocessor_path)
            model = self.data_handler.load_object(file_path = self.modelevalutionconfig.model_path)
            logging.info(f'{x_test.shape}')
            x_test_transformed = preprocessor.transform(x_test)
     
            logging.info(f'{preprocessor.n_features_in_}')
            report = self.model.evaluate_model(model, x_test_transformed,y_test)
            return report
        except Exception as e:
            logging.error(f"Unexpected error occurred while Evaulating the model: {str(e)}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    model_evalution_object = Modelevaultion()
    report  = model_evalution_object.eval_mode()
