import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, max_error, mean_absolute_percentage_error)
from sklearn.metrics import r2_score
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt


def DataSplit(raw_data):
    # [7] 
    raw_data = pd.read_csv(raw_data, encoding="latin1") 
    raw_data["Date"] = pd.to_datetime(raw_data["Date"])
    train_df = raw_data[raw_data["Date"].dt.year < 2023].copy()
    val_df   = raw_data[raw_data["Date"].dt.year == 2023].copy()
    test_df  = raw_data[raw_data["Date"].dt.year >= 2024].copy()
    return train_df, val_df, test_df
    

def DataProcessing(df):
    # [1]
    #df = pd.read_csv(df, encoding="latin1") 
    df_copy = df.copy()

    # [2] 
    df_copy["Date"] = pd.to_datetime(df_copy["Date"])
    start_date = pd.to_datetime("2015-03-01")
    start_idx = df_copy[df_copy["Date"] >= start_date].index[0]
    df_copy = df_copy.loc[start_idx:].copy()
    # y = df_copy["[Filt] Mean Turbidity [NTU]"].copy()
    # df_copy.drop(columns=["[Filt] Mean Turbidity [NTU]"], inplace=True)
    
    # [3]
    days_in_year = np.where(df_copy["Date"].dt.is_leap_year, 366, 365)
    theta = 2 * np.pi * df_copy["Date"].dt.dayofyear / days_in_year
    df_copy.insert((df_copy.columns.get_loc("Date")+1), "Date_cos", np.cos(theta))  
    df_copy.insert((df_copy.columns.get_loc("Date")+2), "Date_sin", np.sin(theta))  

    # [4]
    df_copy.drop(columns = "[Chem] Alum Dose [mg/L]", inplace = True) #old aluminum input
    downstream_feaures = df_copy.columns.get_loc("[TW] Turbidity [NTU]") # [TW] Turbidity [NTU]  [Filt] Total Runtime [h]
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

    return df_copy #,y


def SlidingWindowWithTarget(df, window_size, shift_step):
    data = []

    i = 0
    while i + window_size< df.shape[0]:  # i + window_size +2
        window = df.iloc[i:i+window_size]  # 
        #window = df.iloc[i:i+window_size][["[Filt] Mean Turbidity [NTU]"]]
        #target_value = y.iloc[i+window_size]  # next timestep target from y
        target_value = df.iloc[i+window_size]["[Filt] Mean Turbidity [NTU]"]  # [Filt] Max Turbidity [NTU] #[Filt] Mean Turbidity [NTU] [TW] Al [mg/L]  # window_size +2
        
        data.append((window, target_value))
        
        i += shift_step

    return data


def get_regression_metrics(model, X, y_true):

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


def optimization_objective(space):
   # Create an XGBoost regressor with the hyperparameters from the search space
   model = xgb.XGBRegressor(
                   n_estimators = int(space['n_estimators']),
                   max_depth = int(space['max_depth']),
                   learning_rate = space['learning_rate'],
                   subsample = space['subsample'],
                   reg_lambda = space['reg_lambda'],
                   gamma = space['gamma'],
                   reg_alpha = int(space['reg_alpha']),
                   min_child_weight = int(space['min_child_weight']),
                   colsample_bytree = space['colsample_bytree'],
                   random_state = random_state,
                   eval_metric="rmse",
                   early_stopping_rounds=15)

   # Fit the model on the training data and evaluate on the validation set
   model.fit(X_train, y_train,
           eval_set= [(X_val, y_val)],
           verbose=False)

   y_val_pred = model.predict(X_val)
   
   val_r2 = r2_score(y_val, y_val_pred)
   val_mse = mean_squared_error(y_val, y_val_pred)
   val_mae = mean_absolute_error(y_val, y_val_pred)
  
   return {'loss': val_mse, 'status': STATUS_OK, 'metrics': {'r2': val_r2, 'mse': val_mse, 'mae': val_mae}}

'''
Main Execution
'''

threshold = 0.10
window_size = 7
shift_step = 7
random_state = 42
train_raw, val_raw, test_raw = DataSplit("data/WTP_raw_data.csv")

X_train_df = DataProcessing(train_raw)
X_val_df   = DataProcessing(val_raw)
X_test_df  = DataProcessing(test_raw)

train_data = SlidingWindowWithTarget(X_train_df, window_size, shift_step)
val_data   = SlidingWindowWithTarget(X_val_df,   window_size, shift_step)
test_data  = SlidingWindowWithTarget(X_test_df,  window_size, shift_step)

X_train = np.vstack([window.values.reshape(-1) for window, target in train_data])
y_train = np.array([target for window, target in train_data])

X_val = np.vstack([window.values.reshape(-1) for window, target in val_data])
y_val = np.array([target for window, target in val_data])

X_test = np.vstack([window.values.reshape(-1) for window, target in test_data])
y_test = np.array([target for window, target in test_data])

print(train_data[0])

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.01,
    subsample=1,
    colsample_bytree=1.0,
    reg_lambda=10,
    reg_aplha=1,
    gamma=0.0,
    random_state=random_state,
    eval_metric="rmse",
    early_stopping_rounds = 50
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

xgb_train_results = get_regression_metrics(xgb_model, X_train, y_train)
xgb_test_results  = get_regression_metrics(xgb_model, X_test, y_test)

train_metrics = get_regression_metrics(xgb_model, X_train, y_train)
val_metrics   = get_regression_metrics(xgb_model, X_val, y_val)
test_metrics  = get_regression_metrics(xgb_model, X_test, y_test)

print("train:", train_metrics)
print("val:  ", val_metrics)
print("test: ", test_metrics)

y_train_pred = xgb_model.predict(X_train)
y_val_pred   = xgb_model.predict(X_val)
y_test_pred  = xgb_model.predict(X_test)

print(f"Training R2 Score:   {r2_score(y_train, y_train_pred):.6f}")
print(f"Validation R2 Score: {r2_score(y_val,   y_val_pred):.6f}")
print(f"Test R2 Score:       {r2_score(y_test,  y_test_pred):.6f}")


turb_col = "[Filt] Mean Turbidity [NTU]"
naive_pred_test = np.array([window[turb_col].iloc[-1] for window, target in test_data])
print("Naive TEST R2:", r2_score(y_test, naive_pred_test))
print("Model TEST R2:", r2_score(y_test, y_test_pred))

print("Train y mean/std:", y_train.mean(), y_train.std())
print("Val   y mean/std:", y_val.mean(), y_val.std())
print("Test  y mean/std:", y_test.mean(), y_test.std())

'''
Hyperparameter Optimization with Hyperopt
'''

space={'max_depth': hp.quniform("max_depth", 3, 8, 1),
       'gamma': hp.uniform('gamma', 0, 2),
       'subsample': hp.uniform('subsample', 0.6, 1),
       'reg_alpha' : hp.uniform('reg_alpha', 0.1, 5),
       'reg_lambda' : hp.uniform('reg_lambda', 5, 50),
       'learning_rate' : hp.uniform('learning_rate', 0.001, 0.05),
       'colsample_bytree' : hp.uniform('colsample_bytree', 0.6, 1),
       'min_child_weight' : hp.quniform('min_child_weight', 1, 5, 1),
       'n_estimators': hp.quniform('n_estimators', 100, 300, 10)}

# fix RNG so optimization is deterministic
trials = Trials()

# Run the hyperparameter optimization and get the best hyperparameters
best_hyperparams = fmin(fn = optimization_objective,
                       space = space,
                       algo = tpe.suggest,
                       max_evals = 2000,
                       trials = trials,
                       rstate = np.random.default_rng(random_state))
                      
print("\nBest Hyperparameters:", best_hyperparams)

# Train the final model with the best hyperparameters
xgb_optimized_model = xgb.XGBRegressor(
   n_estimators=int(best_hyperparams['n_estimators']),
   max_depth=int(best_hyperparams['max_depth']),
   learning_rate=best_hyperparams['learning_rate'],
   subsample=best_hyperparams['subsample'],
   colsample_bytree=best_hyperparams['colsample_bytree'],
   reg_lambda=best_hyperparams['reg_lambda'],
   reg_alpha=int(best_hyperparams['reg_alpha']),
   gamma=best_hyperparams['gamma'],
   random_state=random_state,
   eval_metric="rmse",
   early_stopping_rounds = 15
)

xgb_optimized_model.fit(
   X_train, y_train,
   eval_set=[(X_val, y_val)],
   verbose=False
)

y_train_pred_opt = xgb_optimized_model.predict(X_train)
y_val_pred_opt   = xgb_optimized_model.predict(X_val)
y_test_pred_opt  = xgb_optimized_model.predict(X_test)

print(f"Training R2 Score for Optimized Model:   {r2_score(y_train, y_train_pred_opt):.6f}")
print(f"Validation R2 Score for Optimized Model: {r2_score(y_val,   y_val_pred_opt):.6f}")
print(f"Test R2 Score for Optimized Model:       {r2_score(y_test,  y_test_pred_opt):.6f}")
test_metrics_opt  = get_regression_metrics(xgb_optimized_model, X_test, y_test)
print("Test Metrics for Optimized Model: ", test_metrics_opt)

saved_metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'Max Error', 'MAPE'],
    'Baseline Model': [test_metrics['mae'], test_metrics['mse'], test_metrics['max_error'], test_metrics['mape']],
    'Optimized Model': [test_metrics_opt['mae'], test_metrics_opt['mse'], test_metrics_opt['max_error'], test_metrics_opt['mape']]
})

saved_hparameters_df = pd.DataFrame({
    'Hyperparameter': ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'reg_lambda', 'reg_alpha', 'gamma', 'min_child_weight', 'early_stopping_rounds'],
    'Baseline Model': [300, 3, 0.01, 1, 1.0, 10, 1, 0.0, 1, 50],
    'Bayesian Optimization': [int(best_hyperparams['n_estimators']), int(best_hyperparams['max_depth']), best_hyperparams['learning_rate'], best_hyperparams['subsample'], best_hyperparams['colsample_bytree'], best_hyperparams['reg_lambda'], int(best_hyperparams['reg_alpha']), best_hyperparams['gamma'], int(best_hyperparams['min_child_weight']), 15]
})

# Save the metrics and hyperparameters to CSV files
saved_metrics_df.to_csv("xgb_test_metrics_comparison.csv", index=False)
saved_hparameters_df.to_csv("xgb_hyperparameters_comparison.csv", index=False)