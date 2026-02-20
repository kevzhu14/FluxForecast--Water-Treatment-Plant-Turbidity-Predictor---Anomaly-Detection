import numpy as np
#import torch
import pandas as pd

# [1] Take original data → [2] start March 1st 2015 → [3] make sure all features are numerical (e.g., cos(date)) → [4] take only features of interest (i.e., remove the TW data) + 
#  → [5] add masked + prev mesured features for each column/feature → [6] forward filling → [7] split train/test → [8] set up tuple (window, prediction)


def DataSplit(raw_data):
    # [7] 
    raw_data = pd.read_csv(raw_data, encoding="latin1") 
    raw_data["Date"] = pd.to_datetime(raw_data["Date"])
    train_df = raw_data[raw_data["Date"].dt.year < 2025].copy()
    test_df  = raw_data[raw_data["Date"].dt.year == 2025].copy()
    return train_df, test_df
    

def DataProcessing(input_file):
    # [1]
    #df = pd.read_csv(input_file, encoding="latin1") 
    df_copy = input_file.copy()

    # [2] 
    df_copy["Date"] = pd.to_datetime(df_copy["Date"])
    start_date = pd.to_datetime("2015-03-01")
    start_idx = df_copy[df_copy["Date"] >= start_date].index[0]
    df_copy = df_copy.loc[start_idx:].copy()
    
    # [3]
    days_in_year = np.where(df_copy["Date"].dt.is_leap_year, 366, 365)
    theta = 2 * np.pi * df_copy["Date"].dt.dayofyear / days_in_year
    df_copy.insert((df_copy.columns.get_loc("Date")+1), "Date_cos", np.cos(theta))
    df_copy.insert((df_copy.columns.get_loc("Date")+2), "Date_sin", np.sin(theta))

    # [4]
    downstream_feaures = df_copy.columns.get_loc("[Filt] Total Runtime [h]")
    initial_features = df_copy.columns.get_loc("Date_cos")
    df_copy = df_copy.iloc[:, initial_features:downstream_feaures].copy()

    # [5]
    original_cols = list(df_copy.columns)
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

    # [6]
    df_copy[original_cols] = df_copy[original_cols].ffill()
    
    ######
    print("done")
    df_copy.to_csv("output_file1.csv", index=False)

    return df_copy


def SlidingWindowWithTarget(df, window_size, shift_step, target_col):
    # [8]

    # X = []# y = []
    data = []

    i = 0
    while i + window_size < df.shape[0]:
        window = df.iloc[i:i+window_size, :]
        target_value = df.iloc[i+window_size][target_col]  # next timestep
        
        # X.append(window) # y.append(target_value)
        data.append((window, target_value))
        
        i += shift_step

    return data




train_data = DataSplit("Raw data.csv")[0]
test_data = DataSplit("Raw data.csv")[1]

#print(len(DataProcessing(train_data)))
print((SlidingWindowWithTarget(DataProcessing(train_data),7,1,"[Filt] Mean Turbidity [NTU]"))[0]) # [(window_df, target), ...]

# Flattening for xgboost 
# X = np.stack([w for (w, y) in data])   # shape: (N, 7, 72)
# y = np.array([y for (w, y) in data])   # shape: (N,)