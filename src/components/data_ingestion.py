import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:

    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts')

class DataIngestion:
    def __init__(self):
        self.ingestion_co
        


