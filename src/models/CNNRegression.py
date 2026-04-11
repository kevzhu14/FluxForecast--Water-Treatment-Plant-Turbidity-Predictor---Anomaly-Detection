import pandas as pd
import numpy as np
import random
import os
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
from src.pipeline.DataProcessing import WindowDataset, DataSplit, SlidingWindowWithTarget, WindowsToNmp, DataProcessing, NormalizeStd, DropNaNCols, ApplyNaNDrop, save_NN_model
import matplotlib.pyplot as plt


SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
    

class CNN1DRegressor(nn.Module):       
    def __init__(self, num_features: int, hidden_channels: int = 64, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()

        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),                 
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)              
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        x = self.head(x)
        return x.squeeze(-1)
    
    def get_config(self):
        return {
            "num_features": self.net[0].in_channels,
            "hidden_channels": self.net[0].out_channels,
            "kernel_size": self.net[0].kernel_size[0],
            "dropout": self.net[4].p
        }


def RegressionMetrics(y_true, y_pred, split_name, model_name, epoch=None):
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()

    mae  = tmf.mean_absolute_error(y_pred, y_true).item()
    mse  = tmf.mean_squared_error(y_pred, y_true).item()
    rmse = tmf.mean_squared_error(y_pred, y_true, squared=False).item()
    mape = tmf.mean_absolute_percentage_error(y_pred, y_true).item()
    r2   = tmf.r2_score(y_pred, y_true).item()

    max_error = torch.max(torch.abs(y_true - y_pred)).item()

    res = {
        "Model": model_name,
        "Split": split_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "MaxError": max_error
    }
    if epoch is not None:
        res["Epoch"] = epoch
    return res


@torch.no_grad()
def EvaluateCNN(model, loader, device):
    model.eval()
    ys, preds = [], []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        pb = model(Xb)
        ys.append(yb.cpu())
        preds.append(pb.cpu())
    y_all = torch.cat(ys)
    p_all = torch.cat(preds)
    return y_all, p_all


def TrainModelCNNRegression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 15,
    hidden_channels: int = 64,
    kernel_size=3,
    dropout: float = 0.2,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = WindowDataset(X_train, y_train)
    val_ds   = WindowDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    num_features = X_train.shape[2]
    model = CNN1DRegressor(
        num_features=num_features,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    wait = 0

    train_rmse_hist = []
    val_rmse_hist = []
    val_mae_hist = []
    val_r2_hist = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n = 0.0, 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.numel()
            n += yb.numel()

        train_mse = running_loss / max(n, 1)
        train_rmse = float(np.sqrt(train_mse))


        y_true_val, y_pred_val = EvaluateCNN(model, val_loader, device)
        val_metrics = RegressionMetrics(
            y_true_val,
            y_pred_val,
            "Val",
            "CNN",
            epoch
        )
        val_rmse = val_metrics["RMSE"]

        train_rmse_hist.append(train_rmse)
        val_rmse_hist.append(val_rmse)
        val_mae_hist.append(val_metrics["MAE"])
        val_r2_hist.append(val_metrics["R2"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_rmse={train_rmse:.4f} | "
            f"val_rmse={val_rmse:.4f} | val_mae={val_metrics['MAE']:.4f} | val_r2={val_metrics['R2']:.4f}"
        )

        # Early stopping
        if val_rmse + 1e-6 < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}. Best val_rmse={best_val_rmse:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    plt.figure(figsize=(8,5))

    plt.plot(train_rmse_hist, label="Train RMSE")
    plt.plot(val_rmse_hist, label="Val RMSE")
    plt.plot(val_mae_hist, label="Val MAE")

    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Training Error Metrics")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_errors.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8,5))

    plt.plot(val_r2_hist, label="Val R²", color="red")

    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("Validation R²")

    plt.legend()

    plt.tight_layout()
    plt.savefig("training_r2.png", dpi=300)
    plt.close()

    plt.scatter(y_true_val, y_pred_val, alpha=0.4)
    plt.plot([min(y_true_val), max(y_true_val)],
            [min(y_true_val), max(y_true_val)], 'r--')
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("Validation R²")

    plt.legend()

    plt.tight_layout()
    plt.savefig("not", dpi=300)
    plt.close()
    return model






def TuneCNN(
    X_train,
    y_train,
    X_val,
    y_val,
    search_space,
    n_trials=20,
    epochs=200,
    patience=20,
    seed=42
):
    best_rmse = float("inf")
    best_params = None
    best_model = None
    all_trials = []

    for trial in range(n_trials):
        params = {
            "hidden_channels": random.choice(search_space["hidden_channels"]),
            "kernel_size": random.choice(search_space["kernel_size"]),
            "dropout": random.choice(search_space["dropout"]),
            "lr": random.choice(search_space["lr"]),
            "batch_size": random.choice(search_space["batch_size"]),
        }

        print("\n==============================")
        print(f"Trial {trial+1}/{n_trials}")
        print(params)

        model = TrainModelCNNRegression(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=params["batch_size"],
            lr=params["lr"],
            hidden_channels=params["hidden_channels"],
            kernel_size=params["kernel_size"],
            dropout=params["dropout"],
            epochs=epochs,
            patience=patience,
            seed=seed
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=128, shuffle=False)

        y_true_val, y_pred_val = EvaluateCNN(model, val_loader, device)
        metrics = RegressionMetrics(y_true_val, y_pred_val, "Val", "CNN")

        rmse = metrics["RMSE"]
        mae = metrics["MAE"]

        all_trials.append({
            "trial": trial + 1,
            **params,
            "val_rmse": rmse,
            "val_mae": mae
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            best_model = model

    print("\n==============================")
    print("BEST RESULT")
    print("Best RMSE:", best_rmse)
    print("Best parameters:", best_params)

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_rmse": best_rmse,
        "trials_df": pd.DataFrame(all_trials)
    }




def main():
    window_size=14
    shift_step=7

    train_raw, val_raw, test_raw = DataSplit(os.path.join("data", "WTP_raw_data.csv"))

    drop_cols = DropNaNCols(train_raw)
    train_raw = ApplyNaNDrop(train_raw, drop_cols)
    val_raw   = ApplyNaNDrop(val_raw, drop_cols)
    test_raw  = ApplyNaNDrop(test_raw, drop_cols)

    X_train_df = DataProcessing(train_raw)
    X_val_df   = DataProcessing(val_raw)
    X_test_df  = DataProcessing(test_raw)


    train_data = SlidingWindowWithTarget(X_train_df, window_size, shift_step)
    val_data = SlidingWindowWithTarget(X_val_df, window_size, shift_step)
    test_data = SlidingWindowWithTarget(X_test_df, window_size, shift_step)

    X_train_seq, y_train = WindowsToNmp(train_data)
    X_val_seq,   y_val   = WindowsToNmp(val_data)
    X_test_seq,  y_test  = WindowsToNmp(test_data)
    X_train_seq, X_val_seq, X_test_seq, scaler = NormalizeStd(X_train_seq,X_val_seq,X_test_seq)

    print("Train:", X_train_seq.shape, y_train.shape)
    print("Val:  ", X_val_seq.shape, y_val.shape)


    search_space = {
        "hidden_channels": [16, 32, 64, 128], # "hidden_channels": [16, 32, 64, 128]
        "kernel_size": [3, 5, 7],  # "kernel_size": [3, 5, 7],
        "dropout": [0.0, 0.1, 0.2, 0.4],  # "dropout": [0.0, 0.1, 0.2, 0.4]
        "lr": [1e-4, 3e-4, 1e-3],  # "lr": [1e-4, 3e-4, 1e-3]
        "batch_size": [16, 32, 64]  # "batch_size": [16, 32, 64]
    }

    # results = TuneCNN(
    #     X_train_seq,
    #     y_train,
    #     X_val_seq,
    #     y_val,
    #     search_space,
    #     n_trials=50,
    #     epochs=200,
    #     patience=20,
    #     seed=42
    # )
    # print(results["best_params"])
    # print(results["trials_df"].head())
    # best_model = results["best_model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model = TrainModelCNNRegression(
        X_train=X_train_seq,
        y_train=y_train,
        X_val=X_val_seq,
        y_val=y_val,
        batch_size=16,
        lr=0.001,
        hidden_channels=64,
        kernel_size=7,
        dropout=0.1,
        epochs=100,
        patience=15,
        seed=SEED
    ).to(device)

    test_loader = DataLoader(WindowDataset(X_test_seq, y_test), batch_size=128, shuffle=False)

    y_true_test, y_pred_test = EvaluateCNN(best_model, test_loader, device)
    test_metrics = RegressionMetrics(y_true_test, y_pred_test, "Test", "Best_Tuned_TCN")

    print("\n==============================")
    print("BEST TUNED MODEL - TEST METRICS")
    print(f"Test MAE:  {test_metrics['MAE']:.4f}")
    print(f"Test MSE:  {test_metrics['MSE']:.4f}")
    print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"Test MAPE: {test_metrics['MAPE']:.4f}")
    print(f"Test R2:   {test_metrics['R2']:.4f}")
    print(f"Test Max Error: {test_metrics['MaxError']:.4f}")

    save_NN_model(
        model = best_model,
        config = best_model.get_config(),
        save_dir = os.path.join("saved_models", "Optimized_CNN"),
        extra_params={
            "window": 14,
            "shift": 7
        })
    print(f"Best model saved to {os.path.join('saved_models', 'Optimized_CNN')}")

if __name__ == "__main__":
    main()