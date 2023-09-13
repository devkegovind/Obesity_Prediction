import os
import sys
import pickle
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler

from src.utils import save_object
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")

            # Define which columns should be ordinal encoded and which should be scaled

            cat_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',
            'SCC', 'CALC', 'MTRANS']
            
            num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

            # Define the custom ranking for each ordinal variable
            Gender_cat = ['Male', 'Female']
            family_history_with_overweight_cat = ['no', 'yes']
            FAVC_cat = ['no', 'yes']
            CAEC_cat = ['Frequently', 'Always', 'no', 'Sometimes']
            SMOKE_cat = ['no', 'yes']
            SCC_cat = ['no', 'yes']
            CALC_cat = ['Frequently', 'no', 'Always', 'Sometimes']
            MTRANS_cat = ['Public_Transportation', 'Motorbike', 'Bike', 'Automobile', 'Walking']

            logging.info("Pipeline Initiated")

            # Numerical Pipeline

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories = [Gender_cat, family_history_with_overweight_cat, FAVC_cat,
                                                                    CAEC_cat, SMOKE_cat, SCC_cat, CALC_cat, MTRANS_cat])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [            
                ('num_pipeline', num_pipeline, num_cols),
                ('cat_pipeline', cat_pipeline, cat_cols)
            ]
            )
            logging.info("Pipeline Completed")

            return preprocessor
            

        except Exception as e:
            logging.info("Exception Occured in the Data Transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:


            """Reading Train and Test Data"""

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test Data Completed")
            print()
            logging.info(f"Train Dataframe Head:\n{train_df.head().to_string()}")
            print()
            logging.info(f"Test Dataframe Head:\n{test_df.head().to_string()}")
            print()
            logging.info(f"Train Dataframe Tail:\n{train_df.tail().to_string()}")
            print()
            logging.info(f"Test Dataframe Tail:\n{test_df.tail().to_string()}")
            print()
            logging.info('Obtaining Preprocessing Object')
            
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'NObeyesdad'

            input_feature_train_df = train_df.drop(columns = target_column_name, axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = target_column_name, axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Input_Feature_Train_df :\n {input_feature_train_df.head().to_string()}")
            print()
            logging.info(f"Target_Feature_Train_df :\n {target_feature_train_df.head().to_string()}")
            print()
            logging.info(f"Input_Feature_Test_df :\n {input_feature_test_df.head().to_string()}")
            print()
            logging.info(f"Target_Feature_Test_df :\n {target_feature_test_df.head().to_string()}")
            print()

            # Transforming Using Preprocessing Object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info(f"Input_Feature_Train_Arr :\n {input_feature_train_arr}")
            print()
            logging.info(f"Input_Feature_Test_Arr :\n{input_feature_test_arr}")
            print()
            logging.info("Applying Preprocessing Object on Training & Testing Dataset")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr =  np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Train_arr : \n{train_arr}")
            print()
            logging.info(f"Test arr :\n {test_arr}")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Preprocessor Pickle File Saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.info("Exception Occured in the initiate_data_Transformation")
            raise CustomException(e, sys)
        

















            




         