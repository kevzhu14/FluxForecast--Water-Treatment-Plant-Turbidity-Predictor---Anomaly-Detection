import pandas as pd
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, r2_score,
                             mean_absolute_error, mean_squared_error, max_error, mean_absolute_percentage_error)
import torch
from torch.utils.data import DataLoader
import xgboost as xgb
from DataProcessingFinal import (DataSplit, SlidingWindowWithTarget, WindowsToNmp, DataProcessing, NormalizeStd, 
                                 DropNaNCols, ApplyNaNDrop, DataPrep, WindowDataset)

# Import models and modules
from Naive import NaivePredRegression
from CNN_Regression_Tuning import CNN1DRegressor, EvaluateCNN
from TCN_Regression_Tuning import TCNRegressor, EvaluateTCN


SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_folder = 'data'
figures_folder = os.path.join('figures', 'comparisons')
models_folder = 'saved_models'

target_col = "[Filt] Mean Turbidity [NTU]"

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    maximum_error = max_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "MaxError": maximum_error,
        "R2": r2
    }

def model_metrics(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, model_name):

    # Calculate the regression metrics for each set
    train_metrics = regression_metrics(y_train, y_train_pred)
    val_metrics = regression_metrics(y_val, y_val_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)

    # Store the calculated metrics in a dictionary
    metrics_dict = {
        'model': model_name,

        "train_MAE": train_metrics["MAE"],
        "train_MSE": train_metrics["MSE"],
        "train_RMSE": train_metrics["RMSE"],
        "train_MAPE": train_metrics["MAPE"],
        "train_MaxError": train_metrics["MaxError"],
        "train_R2": train_metrics["R2"],

        "val_MAE": val_metrics["MAE"],
        "val_MSE": val_metrics["MSE"],
        "val_RMSE": val_metrics["RMSE"],
        "val_MAPE": val_metrics["MAPE"],
        "val_MaxError": val_metrics["MaxError"],
        "val_R2": val_metrics["R2"],

        "test_MAE": test_metrics["MAE"],
        "test_MSE": test_metrics["MSE"],
        "test_RMSE": test_metrics["RMSE"],
        "test_MAPE": test_metrics["MAPE"],
        "test_MaxError": test_metrics["MaxError"],
        "test_R2": test_metrics["R2"]
    }

    return metrics_dict

def load_saved_models(root_dir="saved_models"):
    
    loaded_models = {}

    # Search directory for saved models
    for model_name in os.listdir(root_dir):
        model_dir = os.path.join(root_dir, model_name)

        if not os.path.isdir(model_dir):
            continue

        # Load metadata
        with open(os.path.join(model_dir, "metadata.json")) as f:
            metadata = json.load(f)

        framework = metadata["framework"]
        extra = metadata.get("extra", {})

        window = extra.get("window")
        shift = extra.get("shift")

        model = None

        # Load PyTorch model
        if framework == "pytorch":
            if "CNN" in model_name:
                model = load_CNN_model(model_dir)
            if "TCN" in model_name:
                model = load_TCN_model(model_dir)

        # Load XGBoost model
        elif framework == "xgboost":
            model = xgb.XGBRegressor()
            model.load_model(os.path.join(model_dir, "model.json"))

        loaded_models[model_name] = {
            "model": model,
            "framework": framework,
            "window": window,
            "shift": shift,
            "metadata": metadata
        }

    return loaded_models

def load_CNN_model(load_dir):
    # Load config to initialize model architecture of CNN model
    with open(os.path.join(load_dir, "config.json")) as f:
        config = json.load(f)

    model = CNN1DRegressor(**config)

    state_dict = torch.load(os.path.join(load_dir, "model.pth"), map_location="cpu")
    model.load_state_dict(state_dict)

    return model

def load_TCN_model(load_dir):
    # Load config to initialize model architecture of TCN model
    with open(os.path.join(load_dir, "config.json")) as f:
        config = json.load(f)

    model = TCNRegressor(**config)

    state_dict = torch.load(os.path.join(load_dir, "model.pth"), map_location="cpu")
    model.load_state_dict(state_dict)

    return model

models = load_saved_models()


def main():
    for name, data in models.items():
        print(name)
        print("Framework:", data["framework"])
        print("Window:", data["window"])
        print("Shift:", data["shift"])

    # Initialize raw data
    train_raw, val_raw, test_raw = DataSplit(os.path.join(data_folder, "WTP_raw_data.csv"))

    '''
    =================================================================================
    Model 1: Naive Baseline (Last Value)
    '''
    # Initialize using same window, shift as optimal XGBoost for fair comparison
    train_data, val_data, test_data, X_train, y_train, X_val, y_val, X_test, y_test = DataPrep(train_raw, val_raw, test_raw, models['Optimized_XGBoost']["window"], models['Optimized_XGBoost']["shift"])

    naive_train = NaivePredRegression(train_data, target_col)
    naive_val = NaivePredRegression(val_data, target_col)
    naive_test = NaivePredRegression(test_data, target_col)

    # Compute metrics for the naive baseline
    naive_metrics = model_metrics(y_train, naive_train, y_val, naive_val, y_test, naive_test, "Naive Baseline")
    print("Naive Baseline Metrics calculation complete.")


    '''
    =================================================================================
    Model 2: Optimized XGBoost
    '''
    # Make predictions with the best XGBoost model
    y_train_pred = models['Optimized_XGBoost']["model"].predict(X_train)
    y_val_pred = models['Optimized_XGBoost']["model"].predict(X_val)
    y_test_pred = models['Optimized_XGBoost']["model"].predict(X_test)

    # Compute metrics for the optimized XGBoost model
    XGBoost_metrics = model_metrics(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, "Optimized XGBoost")
    print("Optimized XGBoost Metrics calculation complete.")


    '''
    =================================================================================
    Model 3: Optimized 1-D CNN
    '''
    # Apply same preprocessing steps as training
    drop_cols = DropNaNCols(train_raw)
    train_raw = ApplyNaNDrop(train_raw, drop_cols)
    val_raw   = ApplyNaNDrop(val_raw, drop_cols)
    test_raw  = ApplyNaNDrop(test_raw, drop_cols)

    X_train_df = DataProcessing(train_raw)
    X_val_df   = DataProcessing(val_raw)
    X_test_df  = DataProcessing(test_raw)

    train_data = SlidingWindowWithTarget(X_train_df, models['Optimized_CNN']["window"], models['Optimized_CNN']["shift"])
    val_data = SlidingWindowWithTarget(X_val_df, models['Optimized_CNN']["window"], models['Optimized_CNN']["shift"])
    test_data = SlidingWindowWithTarget(X_test_df, models['Optimized_CNN']["window"], models['Optimized_CNN']["shift"])

    X_nn_train_seq, y_nn_train = WindowsToNmp(train_data)
    X_nn_val_seq,   y_nn_val   = WindowsToNmp(val_data)
    X_nn_test_seq,  y_nn_test  = WindowsToNmp(test_data)
    X_nn_train_seq, X_nn_val_seq, X_nn_test_seq, scaler = NormalizeStd(X_nn_train_seq,X_nn_val_seq,X_nn_test_seq)

    # Load data into PyTorch DataLoaders
    train_loader = DataLoader(WindowDataset(X_nn_train_seq, y_nn_train), batch_size=64, shuffle=False)
    val_loader = DataLoader(WindowDataset(X_nn_val_seq, y_nn_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(WindowDataset(X_nn_test_seq, y_nn_test), batch_size=64, shuffle=False)

    # Evaluate the best model on each set
    y_CNN_train, y_CNN_pred_train = EvaluateCNN(models['Optimized_CNN']["model"], train_loader, device)
    y_CNN_val, y_CNN_pred_val = EvaluateCNN(models['Optimized_CNN']["model"], val_loader, device)
    y_CNN_test, y_CNN_pred_test = EvaluateCNN(models['Optimized_CNN']["model"], test_loader, device)

    # Compute model metrics
    CNN_metrics = model_metrics(y_CNN_train, y_CNN_pred_train, y_CNN_val, y_CNN_pred_val, y_CNN_test, y_CNN_pred_test, "Optimized CNN")
    print("Optimized 1-D CNN Metrics calculation complete.")

    '''
    =================================================================================
    Model 4: TCN
    '''
    # Apply same preprocessing steps as training
    train_data = SlidingWindowWithTarget(X_train_df, models['Optimized_TCN']["window"], models['Optimized_TCN']["shift"])
    val_data = SlidingWindowWithTarget(X_val_df, models['Optimized_TCN']["window"], models['Optimized_TCN']["shift"])
    test_data = SlidingWindowWithTarget(X_test_df, models['Optimized_TCN']["window"], models['Optimized_TCN']["shift"])

    X_nn_train_seq, y_nn_train = WindowsToNmp(train_data)
    X_nn_val_seq,   y_nn_val   = WindowsToNmp(val_data)
    X_nn_test_seq,  y_nn_test  = WindowsToNmp(test_data)
    X_nn_train_seq, X_nn_val_seq, X_nn_test_seq, scaler = NormalizeStd(X_nn_train_seq,X_nn_val_seq,X_nn_test_seq)

    # Load data into PyTorch DataLoaders
    train_loader = DataLoader(WindowDataset(X_nn_train_seq, y_nn_train), batch_size=64, shuffle=False)
    val_loader = DataLoader(WindowDataset(X_nn_val_seq, y_nn_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(WindowDataset(X_nn_test_seq, y_nn_test), batch_size=64, shuffle=False)

    # Evaluate the best model on each set
    y_TCN_train, y_TCN_pred_train = EvaluateTCN(models['Optimized_TCN']["model"], train_loader, device)
    y_TCN_val, y_TCN_pred_val = EvaluateTCN(models['Optimized_TCN']["model"], val_loader, device)
    y_TCN_test, y_TCN_pred_test = EvaluateTCN(models['Optimized_TCN']["model"], test_loader, device)

    # Compute model metrics
    TCN_metrics = model_metrics(y_TCN_train, y_TCN_pred_train, y_TCN_val, y_TCN_pred_val, y_TCN_test, y_TCN_pred_test, "Optimized TCN")
    print("Optimized TCN Metrics calculation complete.")


    '''
    =================================================================================
    Plotting and Summary
    '''
    # Combine all metrics into a single DataFrame
    all_metrics = [naive_metrics, XGBoost_metrics, CNN_metrics, TCN_metrics]
    models_performance_df = pd.DataFrame(all_metrics)

    # Save the combined metrics to a CSV file

    models_performance_df.to_csv(os.path.join(data_folder, "model_comparison_metrics.csv"), index=False)
    print(f"Saved model comparison metrics to '{os.path.join(data_folder, 'model_comparison_metrics.csv')}'")

    # Plot Predicted vs Actual for each model
    min_val = min(y_test.min(), y_test_pred.min(), naive_test.min(), y_CNN_test.min(), y_CNN_pred_test.min(),  y_TCN_test.min(), y_TCN_pred_test.min())
    max_val = max(y_test.max(), y_test_pred.max(), naive_test.max(), y_CNN_test.max(), y_CNN_pred_test.max(),  y_TCN_test.max(), y_TCN_pred_test.max())

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, naive_test, alpha=0.6, label="Naive") # Naive baseline
    plt.scatter(y_test, y_test_pred, alpha=0.6, label="XGBoost") # Optimized XGBoost
    plt.scatter(y_CNN_test, y_CNN_pred_test, alpha=0.6, label="Optimized CNN") # Optimized CNN
    plt.scatter(y_TCN_test, y_TCN_pred_test, alpha=0.6, label="Optimized TCN") # Optimized TCN

    plt.plot([min_val, max_val], [min_val, max_val], "k--", label="Ideal") # Ideal Line

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(
        f"Predicted vs Actual For All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, 'best_pred_vs_actual_all_models.png'), dpi=300)
    plt.close()
    print(f"Predicted vs Actual plot saved to '{os.path.join(figures_folder, 'best_pred_vs_actual_all_models.png')}'.")

    # Plot Bar Graph of key eval metrics
    def eval_metric_bar_plot(df, metric, filename):

        metric_cols = [f"train_{metric}", f"val_{metric}", f"test_{metric}"]
        metric_df = df[["model"] + metric_cols].copy()

        metric_long = metric_df.melt(
            id_vars="model",
            var_name="dataset",
            value_name=metric,
        )

        metric_long["dataset"] = metric_long["dataset"].str.replace(f"_{metric}", "")

        # Force dataset order
        desired_order = ["train", "val", "test"]
        metric_long["dataset"] = pd.Categorical(metric_long["dataset"], categories=desired_order, ordered=True)

        pivot = metric_long.pivot(index="model", columns="dataset", values=metric)

        # Reorder columns so train/val/test appears in correct order
        pivot = pivot.reindex(columns=desired_order)

        plt.figure(figsize=(8, 6))
        pivot.plot(kind="bar")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison Across Models")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_folder, filename), dpi=300)
        plt.close()

        print(f"Bar plot of {metric} saved as '{filename}'.")

        return

    eval_metrics = ["MAE", "MSE", "RMSE", "MAPE", "MaxError", "R2"]
    for metric in eval_metrics:
        eval_metric_bar_plot(models_performance_df, metric, f"model_comparison_{metric}.png")

    print("\n======================================\nAll evaluation metric bar plots saved.\n======================================")

if __name__ == "__main__":
    main()