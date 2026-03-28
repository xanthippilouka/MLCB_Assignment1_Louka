#Import required libraries

#Core scientific libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#Skicit-learn preprocessing libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


#Regression algorithms and evaluation metrics
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix, 
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    ConfusionMatrixDisplay)

from scipy.stats import pearsonr


#Function for the Split of the data based on age 
def split_data(data, target='age'):
    df = data.copy()
    #Stratify by age, develope age-groups as some age values are unique is the dataset
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

#Function for Bootstrap
import random

def bootstrap_apply(y_pred, y_val, n_resamples=1000, seed=42):
    np.random.seed(seed) #starting point
    resamples = n_resamples
    n = len(y_val)

    #Initialize the empty statistics matrix
    stats = []

    for i in range(resamples):
        #Pick random indices with replacement
        idx = np.random.choice(np.arange(n), size=n, replace=True)

        y_val_resample = y_val.iloc[idx]

        if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
            y_pred_resample= y_pred.iloc[idx]
        else:
            y_pred_resample = y_pred[idx]
       
        #Metrics
        RMSE = np.sqrt(mean_squared_error(y_val_resample, y_pred_resample))
        MAE = mean_absolute_error(y_val_resample, y_pred_resample)
        R2 = r2_score(y_val_resample, y_pred_resample)
        r, _ = pearsonr(y_val_resample, y_pred_resample) 

        stats.append({'RMSE':RMSE, 'MAE':MAE, 'R2':R2, 'Pearson_r':r})
        
    return pd.DataFrame(stats, columns=['RMSE', 'MAE', 'R2', 'Pearson_r'])


#Function for stability selection method for feature selection

from scipy import stats

def stability_selection(X_train, y_train, n_subsamples=50, sample_size=0.8, top=200, seed=42):
    np.random.seed(seed) #set the starting point
    n = len(X_train)
    subsample_size = int(n* sample_size)

    #Initialize an empty list to store the 200 top features
    top200_features = []

    for i in range(n_subsamples):

        #Pick random indices without replacement to generate the subsample
        idx = np.random.choice(np.arange(len(X_train)), size=subsample_size, replace=False)
        X_subsample = X_train.iloc[idx]
        y_subsample = y_train.iloc[idx]

        #Calculate Spearman correlation(all CpG with age)
        corr_list = [] #Initialize an empty list to store the correlation results for the specific subsample

        for col_name in X_subsample.columns:
            col_spearman , pvalue = stats.spearmanr(X_subsample[col_name], y_subsample)

            corr_list.append(col_spearman)

        #Convert to Pandas
        series_corr_list = pd.Series(corr_list, index=X_subsample.columns)

        #Take the absolute values
        abs_corr_list = series_corr_list.abs()

        #Sort the values to take the top200
        abs_corr_list_sorted = abs_corr_list.sort_values(ascending=False)
        top200 = abs_corr_list_sorted.head(top).index.tolist()

        top200_features.extend(top200) #this list in the end will have all the 200 top200 features from each subsample (total 10,000)

    return top200_features










