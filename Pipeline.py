import pandas as pd
import numpy as np
import random
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, max_error, mean_absolute_percentage_error)
from sklearn.metrics import r2_score
import xgboost as xgb 
import os 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics.functional as tmf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from DataProcessing import (DataSplit, SlidingWindowWithTarget, WindowsToNmp, 
                              DataProcessing, NormalizeStd, FlattenData, DropNaNCols, ApplyNaNDrop)
from CNNClassifier import (CNN1DClassifier, TrainModelCNNClassifier, EvaluateCNN, TorchPredict)
from CNNRegression import (CNN1DRegressor, TrainModelCNNRegression)
from Naive import (NaivePredClassifier, NaivePredRegression)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve
from TCNClassifier import (TCNClassifier, TrainModelTCNClassifier)
from TCNRegression import (TCNRegressor, TrainModelTCNRegression)
import json
from Model_Comparison import load_CNN_model, load_TCN_model

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_FOLDER = 'data'
FIGURES_FOLDER = os.path.join('figures', 'comparisons')
MODELS_FOLDER = 'saved_models'

TARGET_COL = "[Filt] Mean Turbidity [NTU]"

def PrepareData(config, df):
    model_type = config["model_type"]
    window_size = config["window_size"]
    shift_step = config["shift_step"]
    threshold = config.get("threshold", None)
    horizon = config.get("horizon", 1)
    drop_nans = config.get("drop_nans", True)
    task = config.get("task", "classification")
    target_col = config["target_col"]

    train_raw, val_raw, test_raw = DataSplit(df)

    if config["use_pretrained"]:
        window_size = 14
        shift_step = 7
        if model_type in ["cnn", "tcn"]:
            drop_nans = True
        else: drop_nans = False

    if drop_nans:
        drop_cols = DropNaNCols(train_raw)
        train_raw, val_raw, test_raw = [ApplyNaNDrop(x, drop_cols) for x in (train_raw, val_raw, test_raw)]
    X_train_df, X_val_df, X_test_df = [DataProcessing(x) for x in (train_raw, val_raw, test_raw)]
    train_data, val_data, test_data = [SlidingWindowWithTarget(x, window_size, shift_step, task=task, threshold=threshold, horizon=horizon, target_col = target_col) for x in (X_train_df, X_val_df, X_test_df)]

    if model_type in ["xgboost", "naive"]:
        X_train, y_train = FlattenData(train_data)
        X_val,   y_val   = FlattenData(val_data)
        X_test,  y_test  = FlattenData(test_data)
        scaler = None 
    elif model_type in ["cnn", "tcn"]:
        X_train, y_train = WindowsToNmp(train_data)
        X_val,   y_val   = WindowsToNmp(val_data)
        X_test,  y_test  = WindowsToNmp(test_data)
        X_train, X_val, X_test, scaler = NormalizeStd(X_train, X_val, X_test) 
    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)
    return {
    "train_data": train_data,
    "val_data": val_data,
    "test_data": test_data,
    "X_train": X_train,
    "y_train": y_train,
    "X_val": X_val,
    "y_val": y_val,
    "X_test": X_test,
    "y_test": y_test,
    "scaler": scaler
} 


def BuildModel(config, input_shape=None):
    model_type = config["model_type"]
    task = config.get("task", "classification")
    use_pretrained = config.get("use_pretrained", False)

    if model_type == "naive":
        return None 
    elif model_type == "xgboost":
        if task == "classification":
            return xgb.XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.01,
                subsample=1.0,
                colsample_bytree=1.0,
                reg_lambda=10,
                gamma=0.0,
                random_state=42,
                objective="binary:logistic",
                eval_metric=["logloss", "auc"]
            )
        elif task == "regression":
            if use_pretrained:
                # Load optimized model for w14_s7
                model_dir = os.path.join(MODELS_FOLDER, "Optimized_XGBoost")
                model = xgb.XGBRegressor()
                model.load_model(os.path.join(model_dir, "model.json"))
            else:
                model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.01,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    reg_lambda=10,
                    gamma=0.0,
                    random_state=42,
                    objective="reg:squarederror",
                    eval_metric="rmse"
                )
            return model
                        
    elif model_type == "cnn":
        if input_shape is None:
            raise ValueError("input_shape is required for cnn")
        if task == "classification":
            return CNN1DClassifier(
                num_features=input_shape[2],
                hidden_channels=64,
                kernel_size=3,
                dropout=0.2
            )
        elif task == "regression":
            if use_pretrained:
                # Load optimized model for w14_s7
                model_dir = os.path.join(MODELS_FOLDER, "Optimized_CNN")
                model = load_CNN_model(model_dir)
            else:
                model = CNN1DRegressor(
                    num_features=input_shape[2],
                    hidden_channels=64,
                    kernel_size=3,
                    dropout=0.2
                )
            return model
    elif model_type == "tcn":
        if input_shape is None:
            raise ValueError("input_shape is required for tcn")
        if task == "classification":
            return TCNClassifier(
                num_features=input_shape[2],
                channels=config.get("channels", [32, 32, 32]),
                kernel_size=config.get("kernel_size", 3),
                dropout=config.get("dropout", 0.2)
            )
        elif task == "regression":
            if use_pretrained:
                # Load optimized model for w14_s7
                model_dir = os.path.join(MODELS_FOLDER, "Optimized_TCN")
                model = load_TCN_model(model_dir)
            else:
                model = TCNRegressor(
                num_features=input_shape[2],
                channels=config.get("channels", [32, 32, 32]),
                kernel_size=config.get("kernel_size", 3),
                dropout=config.get("dropout", 0.2)
                )
            return model
    else:
        raise ValueError(f"Unknown model_type: {model_type}")    

def TrainAndPredict(model, data, config):
    model_type = config["model_type"]
    task = config.get("task", "classification")
    target_col = config["target_col"]
    threshold = config.get("threshold", None)
    use_pretrained = config.get("use_pretrained", False)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val     = data["X_val"], data["y_val"]
    X_test, y_test   = data["X_test"], data["y_test"]

    train_data = data["train_data"]
    val_data   = data["val_data"]
    test_data  = data["test_data"]

    if model_type == "naive":
        if task == "classification":
            y_pred = NaivePredClassifier(test_data, target_col, threshold)
            return {
                "model": None,
                "y_true": y_test,
                "y_pred": y_pred
            }
        elif task == "regression":
            y_pred = NaivePredRegression(test_data, target_col)
            return {
                "model": None,
                "y_true": y_test,
                "y_pred": y_pred
            }

        else:
            raise ValueError("task must be 'classification' or 'regression'")
    elif model_type == "xgboost":
        if not use_pretrained:
            # Train model if not loading optimized version
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        if task == "classification":
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= config.get("prob_threshold", 0.5)).astype(np.float32)
            return {
                "model": model,
                "y_true": y_test,
                "y_prob": y_prob,
                "y_pred": y_pred
            }
        elif task == "regression":
            y_pred = model.predict(X_test)
            return {
                "model": model,
                "y_true": y_test,
                "y_pred": y_pred
            }
        else:
            raise ValueError("task must be 'classification' or 'regression'")

    elif model_type == "cnn":
        if task == "classification":
            model = TrainModelCNNClassifier(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                batch_size=config.get("batch_size", 64),
                lr=config.get("lr", 1e-3),
                epochs=config.get("epochs", 100),
                patience=config.get("patience", 15),
                hidden_channels=config.get("hidden_channels", 64),
                kernel_size=config.get("kernel_size", 3),
                dropout=config.get("dropout", 0.2),
                seed=config.get("seed", 42)
            )

            y_prob = TorchPredict(model, X_test, task="classification")
            y_pred = (y_prob >= config.get("prob_threshold", 0.5)).astype(np.float32)

            return {
                "model": model,
                "y_true": y_test,
                "y_prob": y_prob,
                "y_pred": y_pred
            }
        elif task == "regression":
            if not use_pretrained:
                # Train model if not loading optimized version
                model = TrainModelCNNRegression(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    batch_size=config.get("batch_size", 64),
                    lr=config.get("lr", 1e-3),
                    epochs=config.get("epochs", 100),
                    patience=config.get("patience", 15),
                    hidden_channels=config.get("hidden_channels", 64),
                    kernel_size=config.get("kernel_size", 3),
                    dropout=config.get("dropout", 0.2),
                    seed=config.get("seed", 42)
                )
            y_pred = TorchPredict(model, X_test, task="regression")
            return {
                "model": model,
                "y_true": y_test,
                "y_pred": y_pred
            }
    elif model_type == "tcn":
        if task == "classification":
            model = TrainModelTCNClassifier(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                batch_size=config.get("batch_size", 64),
                lr=config.get("lr", 1e-3),
                epochs=config.get("epochs", 100),
                patience=config.get("patience", 15),
                channels=config.get("channels", [32, 32, 32]),
                kernel_size=config.get("kernel_size", 3),
                dropout=config.get("dropout", 0.2),
                seed=config.get("seed", 42)
            )
            y_prob = TorchPredict(model, X_test, task="classification")
            y_pred = (y_prob >= config.get("prob_threshold", 0.5)).astype(np.float32)

            return {
                "model": model,
                "y_true": y_test,
                "y_prob": y_prob,
                "y_pred": y_pred
            }
        elif task == "regression":
            if not use_pretrained:
                # Train model if not loading optimized version
                model = TrainModelTCNRegression(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    batch_size=config.get("batch_size", 64),
                    lr=config.get("lr", 1e-3),
                    epochs=config.get("epochs", 100),
                    patience=config.get("patience", 15),
                    channels=config.get("channels", [32, 32, 32]),
                    kernel_size=config.get("kernel_size", 3),
                    dropout=config.get("dropout", 0.2),
                    fc_hidden=config.get("fc_hidden", 32),
                    dilation_reset=config.get("dilation_reset", None),
                    use_norm=config.get("use_norm", "weight_norm"),
                    weight_decay=config.get("weight_decay", 0.0),
                    seed=config.get("seed", 42)
                )
            y_pred = TorchPredict(model, X_test, task="regression")

            return {
                "model": model,
                "y_true": y_test,
                "y_pred": y_pred
            }

    else:
        raise ValueError("task must be 'classification' or 'regression'")
        

def EvaluateResults(results, config):
    task = config.get("task", "classification")

    y_true = results["y_true"]
    y_pred = results["y_pred"]

    if task == "classification":
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

        if "y_prob" in results:
            y_prob = results["y_prob"]
            if len(np.unique(y_true)) > 1:
                from sklearn.metrics import roc_auc_score
                metrics["auc"] = roc_auc_score(y_true, y_prob)
            else:
                metrics["auc"] = np.nan

        return metrics

    elif task == "regression":
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": rmse,
            "r2": r2_score(y_true, y_pred),
            "max_error": max_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred)
        }

        return metrics

    else:
        raise ValueError("task must be 'classification' or 'regression'")



def MakeRunDirs(config, base_dir="results"):
    model_type = config["model_type"]
    task = config.get("task", "classification")
    window_size = config.get("window_size", "na")
    shift_step = config.get("shift_step", "na")
    horizon = config.get("horizon", "na")
    threshold = config.get("threshold", "na")

    run_name = f"{model_type}_{task}_w{window_size}_s{shift_step}_h{horizon}_thr{threshold}"
    run_dir = os.path.join(base_dir, run_name)
    plots_dir = os.path.join(run_dir, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    return {
        "run_dir": run_dir,
        "plots_dir": plots_dir,
        "run_name": run_name
    }

def SaveMetricsCSV(metrics, config, run_paths):
    row = {
        "model_type": config["model_type"],
        "task": config.get("task", "classification"),
        "window_size": config.get("window_size"),
        "shift_step": config.get("shift_step"),
        "horizon": config.get("horizon"),
        "threshold": config.get("threshold"),
        "prob_threshold": config.get("prob_threshold")
    }
    row.update(metrics)

    df = pd.DataFrame([row])
    save_path = os.path.join(run_paths["run_dir"], "metrics.csv")
    df.to_csv(save_path, index=False)
    return save_path
def SavePredictionsCSV(results, run_paths):
    out = {
        "y_true": results["y_true"],
        "y_pred": results["y_pred"]
    }

    if "y_prob" in results:
        out["y_prob"] = results["y_prob"]

    df = pd.DataFrame(out)
    save_path = os.path.join(run_paths["run_dir"], "predictions.csv")
    df.to_csv(save_path, index=False)
    return save_path

def SaveConfig(config, run_paths):
    save_path = os.path.join(run_paths["run_dir"], "config.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    return save_path

def DisplayResults(results, metrics, config, run_paths=None):
    model_type = config["model_type"]
    task = config.get("task", "classification")

    lines = []
    lines.append("==============================")
    lines.append(f"Model: {model_type}")
    lines.append(f"Task: {task}")

    if task == "classification":
        lines.append(f"Accuracy : {metrics['accuracy']:.4f}")
        lines.append(f"Precision: {metrics['precision']:.4f}")
        lines.append(f"Recall   : {metrics['recall']:.4f}")
        lines.append(f"F1       : {metrics['f1']:.4f}")
        if "auc" in metrics:
            lines.append(f"AUC      : {metrics['auc']:.4f}")

    elif task == "regression":
        lines.append(f"MAE      : {metrics['mae']:.4f}")
        lines.append(f"MSE      : {metrics['mse']:.4f}")
        lines.append(f"RMSE     : {metrics['rmse']:.4f}")
        lines.append(f"R2       : {metrics['r2']:.4f}")
        lines.append(f"Max Error: {metrics['max_error']:.4f}")
        lines.append(f"MAPE     : {metrics['mape']:.4f}")

    for line in lines:
        print(line)

    if run_paths is not None:
        txt_path = os.path.join(run_paths["run_dir"], "summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            
def PlotResults(results, metrics, config, run_paths):
    model_type = config["model_type"]
    task = config.get("task", "classification")
    plots_dir = run_paths["plots_dir"]

    y_true = np.asarray(results["y_true"])
    y_pred = np.asarray(results["y_pred"])

    if task == "classification":
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(values_format="d", ax=ax)
        ax.set_title(f"Confusion Matrix - {model_type}")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=300)
        plt.close(fig)

        if "y_prob" in results and len(np.unique(y_true)) > 1:
            y_prob = np.asarray(results["y_prob"])

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f"AUC = {metrics.get('auc', np.nan):.3f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve - {model_type}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "roc_curve.png"), dpi=300)
            plt.close(fig)

            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(recall, precision)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Precision-Recall Curve - {model_type}")
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "precision_recall_curve.png"), dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="True 0")
            ax.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="True 1")
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Predicted Probability Histogram - {model_type}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "probability_histogram.png"), dpi=300)
            plt.close(fig)

    elif task == "regression":
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.plot([min_val, max_val], [min_val, max_val], "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Predicted vs Actual - {model_type}")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "predicted_vs_actual.png"), dpi=300)
        plt.close(fig)

        residuals = y_true - y_pred

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title(f"Residual Plot - {model_type}")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "residual_plot.png"), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(residuals, bins=30)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Residual Histogram - {model_type}")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "residual_histogram.png"), dpi=300)
        plt.close(fig)

