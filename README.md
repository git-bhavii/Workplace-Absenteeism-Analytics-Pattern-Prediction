# Workplace-Absenteeism-Analytics-Pattern-Prediction
This repository contains exploratory data analysis, data cleaning, feature selection, preprocessing pipelines, and regression modeling for the Absenteeism at Work dataset (Absenteeism_at_work.csv). The goal is to predict absenteeism_time_in_hours (renamed from "Absenteeism time in hours") from employee, work, and absence-related attributes.
​Dataset Overview
​File: Absenteeism_at_work.csv (semicolon-separated ; with fallback to ,)
​Shape: 740 rows and 21 columns
​Target Variable: absenteeism_time_in_hours (continuous / hourly duration)
​Key Predictors:
​Continuous / Numeric Features: Transportation expense, Distance from Residence to Work, Service time, Age, Work load Average/day, Weight, Height, Body mass index
​Categorical / Coded Features: Reason for absence, Month of absence, Day of the week, Seasons, Disciplinary failure, Education, Son, Social drinker, Social smoker, Pet
​Identifier (Dropped from Features): ID
​Workflow & Methodology
​Data Loading & Cleaning
​Strip whitespace from column headers and normalize "Absenteeism time in hours" to absenteeism_time_in_hours.
​Convert numeric columns safely (handling comma decimals and non-numeric strings via pd.to_numeric(..., errors="coerce") and replacing ±inf with NaN).
​Apply IQR outlier capping on continuous numeric features while excluding coded categorical columns.
​Exploratory Data Analysis (EDA) & Visualization
​Distribution and summary statistics of absenteeism_time_in_hours (noting strong right-skewness with most absences falling in shorter hour ranges and occasional high-hour spikes).
​Bar plots for Seasons and Education (with categories explicitly ordered 1–4 so categories do not collapse into a single bar).
​Correlation heatmap across continuous features and absenteeism_time_in_hours (ensuring Work load Average/day remains numeric and present in the correlation matrix).
​Feature Selection
​Use SelectKBest with f_regression (suited for a continuous target) on clean, imputed numeric predictors to rank feature importance and select top features (k = 10).
​Train/Test Split & Preprocessing Pipeline
​Split data using train_test_split (test_size=0.20, random_state=42) without stratify=y because rare hourly values in absenteeism_time_in_hours have very few instances.
​Build a ColumnTransformer pipeline:
​Numeric features: SimpleImputer(strategy="median") + StandardScaler()
​Categorical features: SimpleImputer(strategy="most_frequent") + OneHotEncoder(handle_unknown="ignore")
​Fit transformers on training data (fit_transform) and apply only transform to test data to prevent data leakage.
​Regression Modeling & Evaluation
​Trained and evaluated multiple regression models on the preprocessed features:
​LinearRegression
​Ridge
​DecisionTreeRegressor
​RandomForestRegressor
​Evaluated using regression metrics: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and R² (Coefficient of Determination).
