import pandas as pd
import numpy as np

def clean_diabetes_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.fillna(df.median(numeric_only=True))
    df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median(numeric_only=True))
    return df

def clean_breast_cancer_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove colunas irrelevantes
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Converte diagnosis para numérico (M=1, B=0)
    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    # Seleciona apenas colunas numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns

    # Substitui infinitos por NaN
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # Imputa NaN com a mediana de cada coluna
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Se ainda restar algum NaN (caso extremo), substitui por 0
    df[num_cols] = df[num_cols].fillna(0)

    return df



def split_diabetes_data(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def split_breast_cancer_data(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
