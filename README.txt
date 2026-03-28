Goal of the Assignment: a machine learning pipeline for epigenetic age prediction from DNA methylation data. 
Development of regression models (epigenetic clocks) that predict chronological age from methylation profiles.

Dataset GSE40279 (Hannum et al., 2013). Part of the initial Dataset analyzed (1000 CpG sites).
*Data not commited to GitHub (.gitignore)

Organization of the repository:

requirements.txt : include the libraries used in the present study (intall in every python notebook)

>data/ : contains the 2 initial dataset .csv files (development_data.csv, evaluation_data.csv)
    -processed/ : contains generated feature matrices (.pkl), table with statistics for training and validation sets
    -baseline/ : metrics from the OLS baseline models (task2_ols_baseline_metrics.csv), metrics from the regression models on CpG feature matrix(task2_cpg_models_metrics.csv)
    -selected_features/ : 
>figures/ : contains all the figures generated for the different Tasks
    -Task1_Figures/ : Figures for Task 1
    -Task2_Figures/ : Figures for Task 2
    -Task3_Figures/ : Figures for Task 3
    -Task4_Figures/ : Figures for Task 4
>models/ : saved models from different steps of the pipeline
    -OLS_Baseline_Models: OLS baseline models (.joblib) for the feature matrices (Task 2)
    -CpG_Baseline_Models: ElasticNet, SVR, BayesianRidge models (.joblib) for the CpG feature matrix (Task 2)
>notebooks/ : Include all the notebooks in which the pipeline for the assignment Tasks is executed
    -data_exploration : Explore the development data (dimensions, type, distribution of target variable), Preprocessing pipeline, Generation of Feature Matrices
    -baseline_models : Establishment of performance baselines, OLS LinearRegression across all the feature sets, ElasticNet/SVR/BayesianRidge for the CpG feature matrix
    -feature_selection : Implement Stability selection for CpG feature matrix, Implement mRMR feature selection algorithm (train OLS models to find the optimum k number of features based on RMSE), venn diagram comparing those 2 feature selection methods
>src/ : include the functions.py file with all the functions developed in the present study