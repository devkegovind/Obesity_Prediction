import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception Occured in Prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 Gender:str,
                 Age:float,
                 Height:float,
                 Weight:float,
                 family_history_with_overweight:str,
                 FAVC:str,
                 FCVC:float,
                 NCP:float,
                 CAEC:str,
                 SMOKE:str,
                 CH2O:float,
                 SCC:str,
                 FAF:float,
                 TUE:float,
                 CALC:str,
                 MTRANS:str                
                 ):
                self.Gender = Gender
                self.Age = Age
                self.Height = Height
                self.Weight = Weight
                self.family_history_with_overweight = family_history_with_overweight
                self.FAVC = FAVC
                self.FCVC = FCVC
                self.NCP = NCP
                self.CAEC = CAEC
                self.SMOKE = SMOKE
                self.CH2O = CH2O
                self.SCC = SCC
                self.FAF = FAF
                self.TUE = TUE
                self.CALC = CALC
                self.MTRANS = MTRANS

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                [self.Gender] : 'Gender',
                [self.Age] : 'Age',
                [self.Height] : 'Height',
                [self.Weight] : 'Weight',
                [self.family_history_with_overweight] : 'family_history_with_overweight',
                [self.FAVC] : 'FAVC',
                [self.FCVC] : 'FCVC',
                [self.NCP] : 'NCP',
                [self.CAEC] : 'CAEC',
                [self.SMOKE] : 'SMOKE',
                [self.CH2O] : 'CH2O',
                [self.SCC] : 'SCC',
                [self.FAF] : 'FAF',
                [self.TUE] : 'TUE',
                [self.CALC] : 'CALC',
                [self.MTRANS] : 'MTRANS'
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Gathered")
            return df
        except Exception as e:
             logging.info('Exception Occured in Prediction Pipeline')
             raise CustomException(e, sys)

