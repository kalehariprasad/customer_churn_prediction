import os
import sys
import dataclasses
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import dataclasses
from src.logger import logging
from src.exception import CustomException
from src.utils import DataHandler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline



@dataclasses.dataclass
class DatatransformationConfig:
    raw_train_data_path: str = 'data/raw/train.csv'
    raw_test_data_path: str = 'data/raw/test.csv'
    preprocessed_train_path = 'data/interim/train.csv'
    preprocessed_test_path = 'data/interim/test.csv'
    preprocessor_path = 'objects/preprosser/object.pkl'
    x_train_features = 'data/processed/train_x.npy'
    y_train_feature = 'data/processed/train_y.npy'



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DatatransformationConfig()
        self.data_handler = DataHandler()

    def get_preprocessor(self):
        try:
            logging.info('data preprocessing stated')
            train_df = pd.read_csv(self.data_transformation_config.raw_train_data_path)
            test_df =  pd.read_csv(self.data_transformation_config.raw_test_data_path)

            
            drop_columns = ['CustomerID']
            
            x_train = train_df.drop(columns='Churn')
            y_train = train_df['Churn'].map({'Yes': 1, 'No': 0})

            x_test = test_df.drop(columns='Churn')
            y_test = test_df['Churn'].map({'Yes': 1, 'No': 0})
            logging.info('dropped Churn column in both train and test data frames')
    
            self.data_handler.save_data(train_df,self.data_transformation_config.preprocessed_train_path)
      
            self.data_handler.save_data(test_df,self.data_transformation_config.preprocessed_test_path)

            # Define categorical and numerical columns
            cat_cols = ['Gender', 'PaymentMethod']
            numerical_cols_standard = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
            numerical_cols_minmax = ['ServiceUsage1', 'ServiceUsage2', 'ServiceUsage3']

            # Create scalers and encoder
            standard_scaler = StandardScaler()
            min_max_scaler = MinMaxScaler()
            one_hot_encoder = OneHotEncoder(sparse_output=True)
            smote = SMOTE()

            # Create the preprocessing pipeline (we won't apply it yet)
            IMB_preprocessor = ImbPipeline([
                ('preprocessing', ColumnTransformer(
                    transformers=[
                        ('standard', standard_scaler, numerical_cols_standard),
                        ('minmax', min_max_scaler, numerical_cols_minmax),
                        ('ohe', one_hot_encoder, cat_cols)
                    ]
                )),
                ('smote', smote)
            ])
            preprocessor = Pipeline([
                ('preprocessing', ColumnTransformer(
                    transformers=[
                        ('standard', standard_scaler, numerical_cols_standard),
                        ('minmax', min_max_scaler, numerical_cols_minmax),
                        ('ohe', one_hot_encoder, cat_cols)
                    ]
                )),
                
            ])
            logging.info(f"x_train shape before preprocessing: {x_train.shape}")
            transormed_x_train = preprocessor.fit_transform(x_train)
            self.data_handler.save_object(preprocessor, self.data_transformation_config.preprocessor_path)
            logging.info(f"x_train shape after transformation: {transormed_x_train.shape}")
            logging.info(f'preprocessor saved at {self.data_transformation_config.preprocessor_path}')


            x_train_resamp , y_train_resmp = IMB_preprocessor.fit_resample(x_train , y_train)
            x_train_resampled, y_train_resampled = IMB_preprocessor.fit_resample(x_train, y_train)
          

    

            #self.data_handler.save_array(x_train_resamp,self.data_transformation_config.x_train_features)
            #self.data_handler.save_array(y_train_resmp, self.data_transformation_config.y_train_feature)
            if isinstance(x_train_resamp, np.ndarray):
                self.data_handler.save_array(x_train_resamp, self.data_transformation_config.x_train_features)
            else:
                self.data_handler.save_array(np.array(x_train_resamp), self.data_transformation_config.x_train_features)

            if isinstance(y_train_resmp, np.ndarray):
                self.data_handler.save_array(y_train_resmp, self.data_transformation_config.y_train_feature)
            else:
                self.data_handler.save_array(np.array(y_train_resmp), self.data_transformation_config.y_train_feature)

            
            return preprocessor

         
        except Exception as e:
            logging.error(f"Unexpected error occurred while setting up preprocessor: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_transformation_object = DataTransformation()
    preprocessor = data_transformation_object.get_preprocessor()

