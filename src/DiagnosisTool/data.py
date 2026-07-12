import os
from kaggle.api.kaggle_api_extended import KaggleApi

import pandas as pd
from pathlib import Path

RAW_DATA_DIR = "data/raw"

def download_diabetes_file():
    api = KaggleApi()

    api.authenticate()
    

    target_dir = os.path.join(RAW_DATA_DIR, "diabetes-data-set")
    os.makedirs(target_dir, exist_ok=True)

    api.dataset_download_files(
        "mathchi/diabetes-data-set",
        path=target_dir,
        unzip=True)
    
def load_diabetes_data():
    download_diabetes_file()
    file_path = os.path.join(RAW_DATA_DIR,"diabetes-data-set/diabetes.csv")
    return pd.read_csv(file_path)
    
def download_breast_cancer_file():
    api = KaggleApi()

    api.authenticate()

    target_dir = os.path.join(RAW_DATA_DIR, "breast-cancer-wisconsin-data")
    os.makedirs(target_dir, exist_ok=True)

    api.dataset_download_files(
        "uciml/breast-cancer-wisconsin-data",
        path=target_dir,
        unzip=True)
    
def load_breast_cancer_data():
    download_breast_cancer_file()
    file_path = os.path.join(RAW_DATA_DIR, "breast-cancer-wisconsin-data/data.csv")
    return pd.read_csv(file_path)
    
