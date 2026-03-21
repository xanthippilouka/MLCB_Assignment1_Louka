#Import required libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import seaborn as sns


#Function for the Split of the data based on age 
def split_data(data, target='age'):
    df = data.copy()
    #Stratify by age, develope age-groups
    df['age_group'] = pd.qcut(df[target], q=4, labels=False)
    
    #Split the data
    X = df.drop(columns = [target])
    y = df[target]
    stratify_col = df['age_group']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_col
    )

    #Remove the additional column
    X_train = X_train.drop(columns=['age_group'])
    X_val = X_val.drop(columns=['age_group'])

    return X_train, X_val, y_train, y_val

#Function for handling the missing values, standardise scales, convert categories into 0s and 1s (create columns for the different strings)
def preprocessing_pipeline(numeric_features, categorical_features):
    #For numerical values
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputation_median", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    #For categorical values, there are not missing data na here
    categorical_preprocessor =  OneHotEncoder(drop='first', sparse_output=False) 

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_preprocessor, numeric_features),
            ("categorical", categorical_preprocessor, categorical_features)
        ]
    )

    return preprocessor




