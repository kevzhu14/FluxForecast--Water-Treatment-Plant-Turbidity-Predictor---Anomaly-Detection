import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import torch
from src.pipeline.DataProcessing import DataSplit, DataPrep,RegressionMetrics
from src.models.Naive import NaivePredRegression


SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

data_folder = 'data'
figures_folder = os.path.join('figures', 'xgb_results')
models_folder = 'saved_models'

window_sizes = [3, 7, 14, 21]   
shift_steps = [1, 3, 7]         
target_col = "[Filt] Mean Turbidity [NTU]"


def BaseModel():
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.01,
        subsample=1,
        colsample_bytree=1.0,
        reg_lambda=10,  
        gamma=0.0,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric=["mae", "rmse"],
    )

def EvaluateXG(model,train_raw, val_raw, test_raw, window_size, shift_step):
    train_data, val_data, test_data, X_train, y_train, X_val, y_val, X_test, y_test = DataPrep(
        train_raw, val_raw, test_raw, window_size, shift_step
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    results = model.evals_result()

    train_metrics = RegressionMetrics(model, X_train, y_train)
    val_metrics = RegressionMetrics(model, X_val, y_val)
    test_metrics = RegressionMetrics(model, X_test, y_test)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    naive_train = NaivePredRegression(train_data, target_col)
    naive_val = NaivePredRegression(val_data, target_col)
    naive_test = NaivePredRegression(test_data, target_col)

    summary = {
        "window_size": window_size,
        "shift_step": shift_step,
        "best_iteration": model.best_iteration,
        "best_score": model.best_score,
        "train_rmse": np.sqrt(train_metrics["mse"]),
        "val_rmse": np.sqrt(val_metrics["mse"]),
        "test_rmse": np.sqrt(test_metrics["mse"]),
        "train_r2": r2_score(y_train, y_train_pred),
        "val_r2": r2_score(y_val, y_val_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "naive_val_rmse": np.sqrt(mean_squared_error(y_val, naive_val)),
        "naive_test_rmse": np.sqrt(mean_squared_error(y_test, naive_test)),
        "naive_val_r2": r2_score(y_val, naive_val),
        "naive_test_r2": r2_score(y_test, naive_test),
        "model": model,
        "results": results,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "naive_test": naive_test,
    }

    return summary


def plot_search_results(results_df):
    plt.figure(figsize=(10, 5))
    plt.bar(
        [f"w={w}, s={s}" for w, s in zip(results_df["window_size"], results_df["shift_step"])],
        results_df["val_rmse"]
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Validation RMSE")
    plt.title("Validation RMSE by Window/Shift Combination")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, "xgb_gridsearch_val_rmse.png"), dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(
        [f"w={w}, s={s}" for w, s in zip(results_df["window_size"], results_df["shift_step"])],
        results_df["val_r2"]
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Validation R²")
    plt.title("Validation R² by Window/Shift Combination")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, "xgb_gridsearch_val_r2.png"), dpi=300)
    print(f"Saved grid search results plots to {figures_folder}")


def plot_best_model(best_result, model_name="baseline"):
    results = best_result["results"]
    train_rmse_hist = results["validation_0"]["rmse"]
    val_rmse_hist = results["validation_1"]["rmse"]
    val_mae_hist = results["validation_1"]["mae"]

    plt.figure(figsize=(8, 5))
    plt.plot(train_rmse_hist, label="Train RMSE")
    plt.plot(val_rmse_hist, label="Val RMSE")
    plt.plot(val_mae_hist, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title(
        f"Training Error Metrics for {model_name.capitalize()} (w={best_result['window_size']}, s={best_result['shift_step']})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, f"xgb_{model_name}_best_training_errors.png"), dpi=300)
    print(f"Saved best model training error plot to {os.path.join(figures_folder, f'xgb_{model_name}_best_training_errors.png')}")

def save_xgb_model(model, config, extra_params, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model.save_model(os.path.join(save_dir, "model.json"))

    # Save config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f)

    # Save metadata + extra params
    metadata = {
        "framework": "xgboost",
        "model_class": model.__class__.__name__,
        "extra": extra_params or {}
    }

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

def main():
    train_raw, val_raw, test_raw = DataSplit(os.path.join(data_folder, "WTP_raw_data.csv"))
    all_results = []

    for window_size in window_sizes:
        for shift_step in shift_steps:
            if shift_step > window_size:
                continue
            print(f"Running window_size={window_size}, shift_step={shift_step} ...")
            result = EvaluateXG(BaseModel(), train_raw, val_raw, test_raw, window_size, shift_step)
            all_results.append(result)

    results_df = pd.DataFrame([
        {
            "window_size": r["window_size"],
            "shift_step": r["shift_step"],
            "best_iteration": r["best_iteration"],
            "best_score": r["best_score"],
            "train_rmse": r["train_rmse"],
            "val_rmse": r["val_rmse"],
            "test_rmse": r["test_rmse"],
            "train_r2": r["train_r2"],
            "val_r2": r["val_r2"],
            "test_r2": r["test_r2"],
            "naive_val_rmse": r["naive_val_rmse"],
            "naive_test_rmse": r["naive_test_rmse"],
            "naive_val_r2": r["naive_val_r2"],
            "naive_test_r2": r["naive_test_r2"],
        }
        for r in all_results
    ])

    results_df["val_rmse_gain_vs_naive"] = results_df["naive_val_rmse"] - results_df["val_rmse"]
    results_df["val_r2_gain_vs_naive"] = results_df["val_r2"] - results_df["naive_val_r2"]

    results_df = results_df.sort_values("val_rmse", ascending=True).reset_index(drop=True)

    print("\nAll combinations ranked by validation RMSE:")
    print(results_df[[
        "window_size", "shift_step",
        "val_rmse", "naive_val_rmse", "val_rmse_gain_vs_naive",
        "val_r2", "naive_val_r2", "val_r2_gain_vs_naive"
    ]])

    best_row = results_df.iloc[0]
    best_window = int(best_row["window_size"])
    best_shift = int(best_row["shift_step"])

    print("\nBest combination:")
    print(f"window_size = {best_window}")
    print(f"shift_step  = {best_shift}")
    print(f"val_rmse    = {best_row['val_rmse']:.6f}")
    print(f"val_r2      = {best_row['val_r2']:.6f}")
    print(f"naive_val_rmse = {best_row['naive_val_rmse']:.6f}")
    print(f"naive_val_r2   = {best_row['naive_val_r2']:.6f}")

    best_result = next(
        r for r in all_results
        if r["window_size"] == best_window and r["shift_step"] == best_shift
    )

    print("\nBest model test performance:")
    print(f"test_rmse       = {best_result['test_rmse']:.6f}")
    print(f"test_r2         = {best_result['test_r2']:.6f}")
    print(f"naive_test_rmse = {best_result['naive_test_rmse']:.6f}")
    print(f"naive_test_r2   = {best_result['naive_test_r2']:.6f}")

    results_df.to_csv(os.path.join(data_folder, "xgboost_window_shift_results.csv"), index=False)
    print(f"Saved grid search results to {os.path.join(data_folder, 'xgboost_window_shift_results.csv')}")

    plot_search_results(results_df)
    plot_best_model(best_result)
    plt.close()

if __name__ == "__main__":
    main()