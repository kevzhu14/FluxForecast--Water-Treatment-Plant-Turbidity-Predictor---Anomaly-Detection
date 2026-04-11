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
import matplotlib.pyplot as plt
from src.pipeline.DataProcessing import (DataSplit, SlidingWindowWithTarget, WindowsToNmp, 
                              DataProcessing, NormalizeStd, FlattenData, DropNaNCols, ApplyNaNDrop)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3
        assert y.ndim == 1
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class CNN1DClassifier(nn.Module):       
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


def ClassificationMetrics(y_true, y_logits, split_name, model_name, epoch=None, prob_threshold=0.5):
    y_true = y_true.detach().cpu()
    y_logits = y_logits.detach().cpu()

    y_prob = torch.sigmoid(y_logits)
    y_pred = (y_prob >= prob_threshold).int()
    y_true_int = y_true.int()

    tp = ((y_pred == 1) & (y_true_int == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true_int == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true_int == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true_int == 1)).sum().item()

    accuracy = accuracy_score(y_true_int.numpy(), y_pred.numpy())
    precision = precision_score(y_true_int.numpy(), y_pred.numpy(), zero_division=0)
    recall = recall_score(y_true_int.numpy(), y_pred.numpy(), zero_division=0)
    f1 = f1_score(y_true_int.numpy(), y_pred.numpy(), zero_division=0)

    res = {
        "Model": model_name,
        "Split": split_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
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


@torch.no_grad()
def TorchPredict(model, X, task="classification", batch_size=256):
    device = next(model.parameters()).device
    dummy_y = np.zeros(len(X), dtype=np.float32)
    ds = WindowDataset(X, dummy_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    preds = []
    model.eval()
    for Xb, _ in loader:
        Xb = Xb.to(device)
        out = model(Xb)
        if task == "classification":
            out = torch.sigmoid(out)
        preds.append(out.cpu())

    return torch.cat(preds).numpy()

def TrainModelCNNClassifier(
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
    model = CNN1DClassifier(
        num_features=num_features,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = -float("inf")
    best_state = None
    wait = 0

    train_loss_hist = []
    val_loss_hist = []
    val_f1_hist = []
    val_recall_hist = []
    val_precision_hist = []

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

        train_loss = running_loss / max(n, 1)


        y_true_val, y_pred_val = EvaluateCNN(model, val_loader, device)
        val_metrics = ClassificationMetrics(
            y_true_val,
            y_pred_val,
            "Val",
            "CNN",
            epoch,
            prob_threshold=0.5
        )
        val_f1 = val_metrics["F1"]

        train_loss_hist.append(train_loss)
        val_f1_hist.append(val_metrics["F1"])
        val_recall_hist.append(val_metrics["Recall"])
        val_precision_hist.append(val_metrics["Precision"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_f1={val_metrics['F1']:.4f} | "
            f"val_recall={val_metrics['Recall']:.4f} | "
            f"val_precision={val_metrics['Precision']:.4f}"
        )

        # Early stopping
        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}. Best val_f1={best_val_f1:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    plt.figure(figsize=(8,5))
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(val_f1_hist, label="Val F1")
    plt.plot(val_recall_hist, label="Val Recall")
    plt.plot(val_precision_hist, label="Val Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Classification Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_classification_metrics.png", dpi=300)
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
    best_f1 = -float("inf")
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

        model = TrainModelCNNClassifier(
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
        metrics = ClassificationMetrics(y_true_val, y_pred_val, "Val", "CNN", prob_threshold=0.2)

        f1 = metrics["F1"]
        recall = metrics["Recall"]
        precision = metrics["Precision"]
        accuracy = metrics["Accuracy"]

        all_trials.append({
            "trial": trial + 1,
            **params,
            "val_f1": f1,
            "val_recall": recall,
            "val_precision": precision,
            "val_accuracy": accuracy
        })

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_model = model

    print("\n==============================")
    print("BEST RESULT")
    print("Best F1:", best_f1)
    print("Best parameters:", best_params)

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_f1": best_f1,
        "trials_df": pd.DataFrame(all_trials)
    }


def main():
    window_size = 14
    shift_step = 3
    threshold = 0.07
    horizon = 3

    train_raw, val_raw, test_raw = DataSplit("Raw data.csv")

    drop_cols = DropNaNCols(train_raw)
    train_raw = ApplyNaNDrop(train_raw, drop_cols)
    val_raw = ApplyNaNDrop(val_raw, drop_cols)
    test_raw = ApplyNaNDrop(test_raw, drop_cols)

    X_train_df = DataProcessing(train_raw)
    X_val_df = DataProcessing(val_raw)
    X_test_df = DataProcessing(test_raw)

    train_data = SlidingWindowWithTarget(X_train_df, window_size, shift_step, threshold=threshold, horizon=horizon)
    val_data = SlidingWindowWithTarget(X_val_df, window_size, shift_step, threshold=threshold, horizon=horizon)
    test_data = SlidingWindowWithTarget(X_test_df, window_size, shift_step, threshold=threshold, horizon=horizon)

    X_train_seq, y_train = WindowsToNmp(train_data)
    X_val_seq, y_val = WindowsToNmp(val_data)
    X_test_seq, y_test = WindowsToNmp(test_data)
    X_train_seq, X_val_seq, X_test_seq, scaler = NormalizeStd(X_train_seq, X_val_seq, X_test_seq)

    print("Train:", X_train_seq.shape, y_train.shape)
    print("Val:  ", X_val_seq.shape, y_val.shape)

    search_space = {
        "hidden_channels": [16, 32, 64, 128],
        "kernel_size": [3, 5, 7],
        "dropout": [0.0, 0.1, 0.2, 0.4],
        "lr": [1e-4, 3e-4, 1e-3],
        "batch_size": [16, 32, 64],
    }

    results = TuneCNN(
        X_train_seq,
        y_train,
        X_val_seq,
        y_val,
        search_space,
        n_trials=20,
        epochs=200,
        patience=20,
        seed=42,
    )

    print(results["best_params"])
    print(results["trials_df"].head())


if __name__ == "__main__":
    main()
