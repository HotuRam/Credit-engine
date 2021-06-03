"""
Make sure you have 'xgboost' and 'bayesian-optimization' installed using 'pip' before running the following code
"""
# Import packages, read credit_data.csv, drop column of row numbers
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from xgboost import XGBRegressor
from scipy import interp
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split

dataset = '../data/data_fs.csv'
print("dataset : ", dataset)
df = pd.read_csv(dataset)

df.drop('Unnamed: 0', axis=1, inplace=True)
print(df.head())


# One hot encoding function
def one_hot(df, nan = False):
    original = list(df.columns)
    category = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns = category, dummy_na = nan, drop_first = True)
    new_columns = [c for c in df.columns if c not in original]
    return df, new_columns

# Feature extraction
df = df.merge(pd.get_dummies(df['Sex'], drop_first=True, prefix='Sex'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df['Housing'], drop_first=False, prefix='Housing'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df["Saving accounts"], drop_first=False, prefix='Saving'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df["Checking account"], drop_first=False, prefix='Checking'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df['Purpose'], drop_first=False, prefix='Purpose'), left_index=True, right_index=True)

# Group age into categories
interval = (18, 25, 40, 65, 100)
categories = ['Student', 'Younger', 'Older', 'Senior']
df["Age_cat"] = pd.cut(df.Age, interval, labels=categories)
df = df.merge(pd.get_dummies(df["Age_cat"], drop_first=False, prefix='Age_cat'), left_index=True, right_index=True)

del df['Sex']
del df['Housing']
del df['Saving accounts']
del df['Checking account']
del df['Purpose']
del df['Age']
del df['Age_cat']

# Scale credit amount by natural log function
df['Credit amount'] = np.log(df['Credit amount'])



from bayes_opt import BayesianOptimization

# Separate X and y of dataset
X = np.array(df.drop(['Credit amount'], axis=1))
y = np.array(df['Credit amount'])

# Split train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Test on unseen data
xgboosted = XGBRegressor(learning_rate=0.2, 
                          gamma=5, 
                          n_estimators=1000, # int
                          max_depth=3, # int
                          min_child_weight=20, # int
                          subsample=0.94, 
                          colsample_bytree=0.95, 
                          scale_pos_weight=0.94)
print("XGBoosted :", xgboosted)

xgboosted.fit(X_train, y_train)
y_pred = xgboosted.predict(X_test)

# Evaluate with mae and mse
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("mae :", mae)
print("mse :", mse)


all_mae = []
all_mse = []

def xgb_function(learning_rate, gamma, min_child_weight, subsample, colsample_bytree, scale_pos_weight):
    """
    Function with XGBoost parameters that returns mae on train and test set
    """
    xgbclf = XGBRegressor(learning_rate=learning_rate, 
                           gamma=gamma, 
                           n_estimators=1000, 
                           max_depth=3, 
                           min_child_weight=min_child_weight, 
                           subsample=subsample, 
                           colsample_bytree=colsample_bytree, 
                           scale_pos_weight=scale_pos_weight)
    
    
    for train_index, val_index in [[np.arange(len(X_train)), np.arange(len(y_train))]]:
        X_train_fun = X_train[train_index]
        y_train_fun = y_train[train_index]
        X_val = X_train[val_index]
        y_val = y_train[val_index]

        xgbclf.fit(X_train_fun, y_train_fun)
        y_pred = xgbclf.predict(X_val)
    
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)

        all_mae.append(mae)
        all_mse.append(mse)
    
    
    mean_mae = np.mean(np.array(all_mae))
    mean_mse = np.mean(np.array(all_mse))
    return mean_mse
    
# Parameter bounds
pbounds = {'learning_rate': (0.01, 0.2), 
           'gamma': (1.0, 5.0), 
           'min_child_weight': (0, 20), 
           'subsample': (0.8, 1.0), 
           'colsample_bytree': (0.7, 1.0), 
           'scale_pos_weight': (0.5, 1.0)}
optimizer = BayesianOptimization(f=xgb_function, pbounds=pbounds, verbose=2)
optimizer.maximize(init_points=2, n_iter=10)
print("Optimizer :", optimizer.max)
