import pandas as pd
import numpy as np
import random
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
from DataProcessing import (DataSplit, SlidingWindowWithTarget, WindowsToNmp, 
                              DataProcessing, NormalizeStd, FlattenData, DropNaNCols, ApplyNaNDrop, ClassificationMetrics)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

threshold=0.07
window_sizes = [3, 7, 14, 21]   
shift_steps = [1, 3, 7]         
target_col = "[Filt] Mean Turbidity [NTU]"  # [TW] Al [mg/L]  [Filt] Mean Turbidity [NTU]

def NaivePredClassifier(data, target_col, threshold=threshold):
    return np.array([
        1.0 if window[target_col].iloc[-1] > threshold else 0.0
        for window, target in data
    ])

def DataPrep(train_raw, val_raw, test_raw, window_size, shift_step, threshold=threshold, horizon=1):
    train_data = SlidingWindowWithTarget(DataProcessing(train_raw), window_size, shift_step,threshold=threshold, horizon=horizon)
    val_data = SlidingWindowWithTarget(
        DataProcessing(val_raw), window_size, shift_step,
        threshold=threshold, horizon=horizon
    )
    test_data = SlidingWindowWithTarget(
        DataProcessing(test_raw), window_size, shift_step,
        threshold=threshold, horizon=horizon
    )

    X_train, y_train = FlattenData(train_data)
    X_val, y_val = FlattenData(val_data)
    X_test, y_test = FlattenData(test_data)

    return train_data, val_data, test_data, X_train, y_train, X_val, y_val, X_test, y_test

def Model():
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.01,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=10,
        gamma=0.0,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric=["logloss", "auc"],
        objective="binary:logistic"
    )

def EvaluateXG(train_raw, val_raw, test_raw, window_size, shift_step, threshold=0.07, horizon=3):
    train_data, val_data, test_data, X_train, y_train, X_val, y_val, X_test, y_test = DataPrep(
        train_raw, val_raw, test_raw, window_size, shift_step, threshold=threshold, horizon=horizon
    )

    model = Model()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    results = model.evals_result()

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    train_metrics = ClassificationMetrics(y_train, y_train_prob)
    val_metrics = ClassificationMetrics(y_val, y_val_prob)
    test_metrics = ClassificationMetrics(y_test, y_test_prob)

    naive_train = NaivePredClassifier(train_data, target_col, threshold=threshold)
    naive_val = NaivePredClassifier(val_data, target_col, threshold=threshold)
    naive_test = NaivePredClassifier(test_data, target_col, threshold=threshold)

    naive_train_metrics = {
        "accuracy": accuracy_score(y_train, naive_train),
        "precision": precision_score(y_train, naive_train, zero_division=0),
        "recall": recall_score(y_train, naive_train, zero_division=0),
        "f1": f1_score(y_train, naive_train, zero_division=0)
    }

    naive_val_metrics = {
        "accuracy": accuracy_score(y_val, naive_val),
        "precision": precision_score(y_val, naive_val, zero_division=0),
        "recall": recall_score(y_val, naive_val, zero_division=0),
        "f1": f1_score(y_val, naive_val, zero_division=0)
    }

    naive_test_metrics = {
        "accuracy": accuracy_score(y_test, naive_test),
        "precision": precision_score(y_test, naive_test, zero_division=0),
        "recall": recall_score(y_test, naive_test, zero_division=0),
        "f1": f1_score(y_test, naive_test, zero_division=0)
    }

    summary = {
        "window_size": window_size,
        "shift_step": shift_step,
        "best_iteration": model.best_iteration,
        "best_score": model.best_score,

        "train_auc": train_metrics["roc_auc"],
        "val_auc": val_metrics["roc_auc"],
        "test_auc": test_metrics["roc_auc"],

        "train_f1": train_metrics["f1"],
        "val_f1": val_metrics["f1"],
        "test_f1": test_metrics["f1"],

        "train_recall": train_metrics["recall"],
        "val_recall": val_metrics["recall"],
        "test_recall": test_metrics["recall"],

        "naive_val_f1": naive_val_metrics["f1"],
        "naive_test_f1": naive_test_metrics["f1"],
        "naive_val_recall": naive_val_metrics["recall"],
        "naive_test_recall": naive_test_metrics["recall"],

        "model": model,
        "results": results,
        "y_test": y_test,
        "y_test_prob": y_test_prob,
        "naive_test": naive_test,
    }

    return summary


def plot_search_results(results_df):
    plt.figure(figsize=(10, 5))
    plt.bar(
        [f"w={w}, s={s}" for w, s in zip(results_df["window_size"], results_df["shift_step"])],
        results_df["val_f1"]
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Validation F1")
    plt.title("Validation F1 by Window/Shift Combination")
    plt.tight_layout()
    plt.savefig("gridsearch_val_f1.png", dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(
        [f"w={w}, s={s}" for w, s in zip(results_df["window_size"], results_df["shift_step"])],
        results_df["val_auc"]
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Validation ROC-AUC")
    plt.title("Validation ROC-AUC by Window/Shift Combination")
    plt.tight_layout()
    plt.savefig("gridsearch_val_auc.png", dpi=300)
    plt.show()
    plt.close()


def plot_best_model(best_result, prob_threshold=0.2):
    results = best_result["results"]

    # training history from XGBoost classifier
    train_logloss_hist = results["validation_0"]["logloss"]
    val_logloss_hist = results["validation_1"]["logloss"]
    val_auc_hist = results["validation_1"]["auc"]

    plt.figure(figsize=(8, 5))
    plt.plot(train_logloss_hist, label="Train Logloss")
    plt.plot(val_logloss_hist, label="Val Logloss")
    plt.xlabel("Epoch")
    plt.ylabel("Logloss")
    plt.title(
        f"Training Logloss (w={best_result['window_size']}, s={best_result['shift_step']})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("best_training_logloss_xgboost.png", dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(val_auc_hist, label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(
        f"Validation AUC (w={best_result['window_size']}, s={best_result['shift_step']})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("best_validation_auc_xgboost.png", dpi=300)
    plt.show()
    plt.close()

    # test predictions
    y_test = best_result["y_test"].astype(int)
    y_test_prob = best_result["y_test_prob"]
    naive_test = best_result["naive_test"].astype(int)

    y_test_pred = (y_test_prob >= prob_threshold).astype(int)

    # confusion matrix for XGBoost
    cm_xgb = confusion_matrix(y_test, y_test_pred)
    disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb)

    plt.figure(figsize=(6, 6))
    disp_xgb.plot(values_format="d")
    plt.title(
        f"XGBoost Confusion Matrix (w={best_result['window_size']}, s={best_result['shift_step']})"
    )
    plt.tight_layout()
    plt.savefig("best_confusion_matrix_xgboost.png", dpi=300)
    plt.show()
    plt.close()

    # confusion matrix for naive baseline
    cm_naive = confusion_matrix(y_test, naive_test)
    disp_naive = ConfusionMatrixDisplay(confusion_matrix=cm_naive)

    plt.figure(figsize=(6, 6))
    disp_naive.plot(values_format="d")
    plt.title(
        f"Naive Confusion Matrix (w={best_result['window_size']}, s={best_result['shift_step']})"
    )
    plt.tight_layout()
    plt.savefig("best_confusion_matrix_naive.png", dpi=300)
    plt.show()
    plt.close()

train_raw, val_raw, test_raw = DataSplit("Raw data.csv")
# drop_cols = DropNaNCols(train_raw)
# train_raw = ApplyNaNDrop(train_raw, drop_cols)
# val_raw   = ApplyNaNDrop(val_raw, drop_cols)
# test_raw  = ApplyNaNDrop(val_raw, drop_cols)

all_results = []

for window_size in window_sizes:
    for shift_step in shift_steps:
        if shift_step > window_size:
            continue
        print(f"Running window_size={window_size}, shift_step={shift_step} ...")
        result = EvaluateXG(train_raw, val_raw, test_raw, window_size, shift_step)
        all_results.append(result)

results_df = pd.DataFrame([
    {
        "window_size": r["window_size"],
        "shift_step": r["shift_step"],
        "best_iteration": r["best_iteration"],
        "best_score": r["best_score"],
        "train_auc": r["train_auc"],
        "val_auc": r["val_auc"],
        "test_auc": r["test_auc"],
        "train_f1": r["train_f1"],
        "val_f1": r["val_f1"],
        "test_f1": r["test_f1"],
        "train_recall": r["train_recall"],
        "val_recall": r["val_recall"],
        "test_recall": r["test_recall"],
        "naive_val_f1": r["naive_val_f1"],
        "naive_test_f1": r["naive_test_f1"],
        "naive_val_recall": r["naive_val_recall"],
        "naive_test_recall": r["naive_test_recall"],
    }
    for r in all_results
])

results_df["val_f1_gain_vs_naive"] = results_df["val_f1"] - results_df["naive_val_f1"]
results_df["val_recall_gain_vs_naive"] = results_df["val_recall"] - results_df["naive_val_recall"]

results_df = results_df.sort_values("val_f1", ascending=False).reset_index(drop=True)

print("\nAll combinations ranked by validation F1:")
print(results_df[[
    "window_size", "shift_step",
    "val_f1", "naive_val_f1", "val_f1_gain_vs_naive",
    "val_recall", "naive_val_recall", "val_recall_gain_vs_naive",
    "val_auc"
]])

best_row = results_df.iloc[0]
best_window = int(best_row["window_size"])
best_shift = int(best_row["shift_step"])

print("\nBest combination:")
print(f"window_size      = {best_window}")
print(f"shift_step       = {best_shift}")
print(f"val_f1           = {best_row['val_f1']:.6f}")
print(f"val_recall       = {best_row['val_recall']:.6f}")
print(f"val_auc          = {best_row['val_auc']:.6f}")
print(f"naive_val_f1     = {best_row['naive_val_f1']:.6f}")
print(f"naive_val_recall = {best_row['naive_val_recall']:.6f}")

best_result = next(
    r for r in all_results
    if r["window_size"] == best_window and r["shift_step"] == best_shift
)

print("\nBest model test performance:")
print(f"test_f1           = {best_result['test_f1']:.6f}")
print(f"test_recall       = {best_result['test_recall']:.6f}")
print(f"test_auc          = {best_result['test_auc']:.6f}")
print(f"naive_test_f1     = {best_result['naive_test_f1']:.6f}")
print(f"naive_test_recall = {best_result['naive_test_recall']:.6f}")

results_df.to_csv("xgboost_window_shift_results.csv", index=False)

plot_search_results(results_df)
plot_best_model(best_result)
plt.close()

