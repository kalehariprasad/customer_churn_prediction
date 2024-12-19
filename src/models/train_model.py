import os
import sys
import pandas as pd
import numpy as np
import dataclasses
from src.logger import logging
from src.exception import CustomException
from src.utils import DataHandler, Model
import xgboost as xgb

@dataclasses.dataclass
class Modeltrainingconfig:
    preprocessor_path = 'objects/preprosser/object.pkl'
    x_train_features = 'data/processed/train_x.npy'
    y_train_feature = 'data/processed/train_y.npy'
    model_path = 'objects/model/model.pkl'
    

class Modeltraining:
    def __init__(self):
        self.modeltrainingconfig = Modeltrainingconfig()
        self.data_handler = DataHandler()
        self.model = Model()

    def train_model(self):
        try:
            model = xgb.XGBClassifier(random_state=42, scale_pos_weight=10)

            # Corrected method name and variable name
            train_x = self.data_handler.load_array(self.modeltrainingconfig.x_train_features)
            train_y = self.data_handler.load_array(self.modeltrainingconfig.y_train_feature)
            logging.info(f'{train_x.shape}')
            logging.info(f'{train_y.shape}')

        
            trained_model = self.model.train_model(model, train_x, train_y)
            
            # Saving the trained model
            self.data_handler.save_object(trained_model, self.modeltrainingconfig.model_path)
            
            logging.info(f"Model trained and saved at {self.modeltrainingconfig.model_path}")
            return trained_model
        
        except Exception as e:
            # Changed info to error for better error handling
            logging.error(f"Unexpected error occurred while training the model: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    model_training_object = Modeltraining()
    model = model_training_object.train_model()
