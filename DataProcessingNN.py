import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, max_error, mean_absolute_percentage_error)
from sklearn.metrics import r2_score
import xgboost as xgb 

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics.functional as tmf
from sklearn.preprocessing import StandardScaler

def DataSplit(raw_data): 
    raw_data = pd.read_csv(raw_data, encoding="latin1") 
    raw_data["Date"] = pd.to_datetime(raw_data["Date"])
    train_df = raw_data[raw_data["Date"].dt.year < 2023].copy()
    val_df   = raw_data[raw_data["Date"].dt.year == 2023].copy()  #    val_df = raw_data[(raw_data["Date"].dt.year >= 2022) & (raw_data["Date"].dt.year <= 2023)].copy()
    test_df  = raw_data[raw_data["Date"].dt.year >= 2024].copy()
    return train_df, val_df, test_df
def SlidingWindowWithTarget(df, window_size, shift_step):
    data = []
    i = 0
    while i + window_size < df.shape[0]:
        window = df.iloc[i:i+window_size]  
        target_value = df.iloc[i+window_size]["[Filt] Mean Turbidity [NTU]"]  # [Filt] Max Turbidity [NTU] #[Filt] Mean Turbidity [NTU] [TW] Al [mg/L]  # window_size +2
        data.append((window, target_value))
        i += shift_step
    return data
def WindowsToNmp(data):
    X = np.stack([window.to_numpy(dtype=np.float32) for window, target in data], axis=0)
    y = np.array([target for window, target in data], dtype=np.float32)
    return X, y
def FlattenData(data):
    X = np.vstack([window.values.reshape(-1) for window, target in data])
    y = np.array([target for window, target in data])
    return X, y
def DataProcessing(df):
    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"])
    start_date = pd.to_datetime("2015-03-01")
    start_idx = df_copy[df_copy["Date"] >= start_date].index[0]
    df_copy = df_copy.loc[start_idx:].copy()
    days_in_year = np.where(df_copy["Date"].dt.is_leap_year, 366, 365)
    theta = 2 * np.pi * df_copy["Date"].dt.dayofyear / days_in_year
    df_copy.insert((df_copy.columns.get_loc("Date")+1), "Date_cos", np.cos(theta))  
    df_copy.insert((df_copy.columns.get_loc("Date")+2), "Date_sin", np.sin(theta))  
    df_copy.drop(columns = "[Chem] Alum Dose [mg/L]", inplace = True) 
    downstream_feaures = df_copy.columns.get_loc("[TW] Turbidity [NTU]") # [TW] Turbidity [NTU]  [Filt] Total Runtime [h]
    initial_features = df_copy.columns.get_loc("Date_cos")
    df_copy = df_copy.iloc[:, initial_features:downstream_feaures].copy()
    raw = df_copy.copy()   
    original_cols = list(raw.columns)
    i = 0
    while i < len(original_cols):
        col = original_cols[i]
        insert_pos = df_copy.columns.get_loc(col) + 1
        mask = df_copy[col].notna().astype(int)  

        pos = pd.Series(np.arange(len(df_copy)), index=df_copy.index)
        last_pos = pos.where(df_copy[col].notna()).ffill()
        prev = (pos - last_pos).where(~df_copy[col].notna(), 0).fillna(0).astype(int)

        df_copy.insert(insert_pos, f"{col}_mask", mask)
        df_copy.insert(insert_pos + 1, f"{col}_previouslymeasured", prev)
        i += 1
    df_copy[original_cols] = df_copy[original_cols].ffill().bfill()
    print("done")
    df_copy.to_csv("output_file1.csv", index=False)
    return df_copy 
def NormalizeStd(X_train, X_val, X_test):
    N_train, T, F = X_train.shape
    N_val = X_val.shape[0]
    N_test = X_test.shape[0]

    X_train_2d = X_train.reshape(-1, F)
    X_val_2d   = X_val.reshape(-1, F)
    X_test_2d  = X_test.reshape(-1, F)

    scaler = StandardScaler()
    scaler.fit(X_train_2d)

    X_train_scaled = scaler.transform(X_train_2d)
    X_val_scaled   = scaler.transform(X_val_2d)
    X_test_scaled  = scaler.transform(X_test_2d)

    X_train_scaled = X_train_scaled.reshape(N_train, T, F)
    X_val_scaled   = X_val_scaled.reshape(N_val, T, F)
    X_test_scaled  = X_test_scaled.reshape(N_test, T, F)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def zscoreN(X_train, X_val, X_test):
    # Compute feature-wise mean/std across all samples and timesteps
    mean = X_train.mean(axis=(0,1), keepdims=True)
    std  = X_train.std(axis=(0,1), keepdims=True) + 1e-8

    # Apply normalization
    X_train_norm = (X_train - mean) / std
    X_val_norm   = (X_val   - mean) / std
    X_test_norm  = (X_test  - mean) / std

    return X_train_norm, X_val_norm, X_test_norm, mean, std
def RegressionMetrics(model, X, y_true):

    # Get predictions from the model using the provided features X
    y_predicted = model.predict(X)

    # Calculate the regression metrics using sklearn.metrics functions
    mae = mean_absolute_error(y_true, y_predicted)
    mse = mean_squared_error(y_true, y_predicted)
    maximum_error = max_error(y_true, y_predicted)
    mape =  mean_absolute_percentage_error(y_true, y_predicted)

    # Store the calculated metrics in a dictionary
    metrics_dict = {
        'mae': mae,
        'mse': mse,
        'max_error': maximum_error,
        'mape': mape
    }

    return metrics_dict

# def nan_percentage(df):
#     nan_percent = df.isna().mean() * 100
#     return nan_percent.sort_values(ascending=False)

# print("Train NaN percentage:")
# print(nan_percentage(train_raw))

# print("\nValidation NaN percentage:")
# print(nan_percentage(val_raw))

# print("\nTest NaN percentage:")
# print(nan_percentage(test_raw))