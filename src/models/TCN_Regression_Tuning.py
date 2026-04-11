import os
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics.functional as tmf
from src.models.TCNRegression import TCNRegressor, RegressionMetrics, EvaluateTCN
from src.pipeline.DataProcessing import DataSplit, SlidingWindowWithTarget, WindowsToNmp, DataProcessing, NormalizeStd, DropNaNCols, ApplyNaNDrop, WindowDataset, save_NN_model
import matplotlib.pyplot as plt
from pytorch_tcn import TCN


SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

data_folder = 'data'
figures_folder = os.path.join('figures', 'tcn_results')
models_folder = 'saved_models'


def TrainModelTCNRegression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 15,
    channels=(32, 32, 32),
    kernel_size=3,
    dropout: float = 0.2,
    fc_hidden: int = 32,
    dilations=None,
    dilation_reset=None,
    use_norm: str = "weight_norm",
    weight_decay: float = 0.0,
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
    model = TCNRegressor(
        num_features=num_features,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
        fc_hidden=fc_hidden,
        dilations=dilations,
        dilation_reset=dilation_reset,
        use_norm=use_norm
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    wait = 0

    train_rmse_hist = []
    val_mae_hist = []
    val_mse_hist = []
    val_rmse_hist = []
    val_mape_hist = []
    val_r2_hist = []
    val_maxerr_hist = []

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


        y_true_val, y_pred_val = EvaluateTCN(model, val_loader, device)
        val_metrics = RegressionMetrics(
            y_true_val,
            y_pred_val,
            "Val",
            "TCN",
            epoch
        )
        val_rmse = val_metrics["RMSE"]

        train_rmse_hist.append(train_rmse)
        val_rmse_hist.append(val_rmse)
        val_mae_hist.append(val_metrics["MAE"])
        val_mse_hist.append(val_metrics["MSE"])
        val_mape_hist.append(val_metrics["MAPE"])
        val_r2_hist.append(val_metrics["R2"])
        val_maxerr_hist.append(val_metrics["MaxError"])

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

    history = {
        "train_rmse": train_rmse_hist,
        "val_rmse": val_rmse_hist,
        "val_mae": val_mae_hist,
        "val_mse": val_mse_hist,
        "val_mape": val_mape_hist,
        "val_r2": val_r2_hist,
        "val_max_error": val_maxerr_hist,
    }

    return model, history


def TuneTCN(
        X_train, y_train, X_val, y_val,
        search_space, n_trials=1, epochs=200, patience=20, seed=42
    ):
        best_rmse = float("inf")
        best_r2 = None
        best_params = None
        best_model = None
        best_history = None
        all_trials = []

        for trial in range(n_trials):
            params = {
                "channels": random.choice(search_space["channels"]),
                "kernel_size": random.choice(search_space["kernel_size"]),
                "dropout": random.choice(search_space["dropout"]),
                "lr": random.choice(search_space["lr"]),
                "batch_size": random.choice(search_space["batch_size"]),
                "fc_hidden": random.choice(search_space["fc_hidden"]),
                "weight_decay": random.choice(search_space["weight_decay"]),
                "dilation_reset": random.choice(search_space["dilation_reset"]),
                "use_norm": random.choice(search_space["use_norm"]),
            }

            print(f"\n==============================")
            print(f"Trial {trial+1}/{n_trials}")
            print(params)

            model, history = TrainModelTCNRegression(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                batch_size=params["batch_size"],
                lr=params["lr"],
                channels=params["channels"],
                kernel_size=params["kernel_size"],
                dropout=params["dropout"],
                fc_hidden=params["fc_hidden"],
                weight_decay=params["weight_decay"],
                dilation_reset=params["dilation_reset"],
                use_norm=params["use_norm"],
                epochs=epochs,
                patience=patience,
                seed=seed
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=128, shuffle=False)
            y_true_val, y_pred_val = EvaluateTCN(model, val_loader, device)
            metrics = RegressionMetrics(y_true_val, y_pred_val, "Val", "TCN")

            rmse = metrics["RMSE"]
            r2 = metrics["R2"]

            all_trials.append({
                "trial": trial + 1,
                **params,
                "val_rmse": rmse,
                "val_mae": metrics["MAE"],
                "val_mse": metrics["MSE"],
                "val_mape": metrics["MAPE"],
                "val_r2": r2
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_r2 = r2
                best_params = params
                best_model = model
                best_history = history

        print("\n==============================")
        print("BEST RESULT")
        print("Best RMSE:", best_rmse)
        print("Best R2:", best_r2)
        print("Best parameters:", best_params)

        return {
            "best_model": best_model,
            "best_params": best_params,
            "best_rmse": best_rmse,
            "best_r2": best_r2,
            "best_history": best_history,
            "trials_df": pd.DataFrame(all_trials)
        }

def main():
    # Find the best hyperparameters for TCN, varying window sizes and shift steps
    window_sizes = [3, 7, 14, 21]
    shift_steps = [1, 3, 7]

    search_space = {
        "channels": [
            (16, 16),
            (16, 16, 16),
            (32, 32),
            (32, 32, 32),
            (64, 64),
            (64, 64, 64),
            (16, 32, 64),
            (32, 32, 64),
            (32, 64, 64),
            (32, 64, 128),
        ],
        "kernel_size": [2, 3, 5, 7],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
        "lr": [1e-4, 3e-4, 1e-3, 3e-3],
        "batch_size": [16, 32, 64, 128],
        "fc_hidden": [16, 32, 64, 128],
        "weight_decay": [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
        "use_norm": ["weight_norm", "batch_norm", "layer_norm", None],
        "dilation_reset": [None, 4, 8, 16],
    }

    train_raw, val_raw, test_raw = DataSplit(os.path.join(data_folder, "WTP_raw_data.csv"))

    drop_cols = DropNaNCols(train_raw)
    train_raw = ApplyNaNDrop(train_raw, drop_cols)
    val_raw   = ApplyNaNDrop(val_raw, drop_cols)
    test_raw  = ApplyNaNDrop(test_raw, drop_cols)

    X_train_df = DataProcessing(train_raw)
    X_val_df   = DataProcessing(val_raw)
    X_test_df  = DataProcessing(test_raw)

    all_results = []

    # Iterate over window sizes and shift steps, train and evaluate TCN for each combination
    for window_size in window_sizes:
        for shift_step in shift_steps:
            print(f"\n=== Window Size: {window_size}, Shift Step: {shift_step} ===")
            train_data = SlidingWindowWithTarget(X_train_df, window_size, shift_step)
            val_data = SlidingWindowWithTarget(X_val_df, window_size, shift_step)
            test_data = SlidingWindowWithTarget(X_test_df, window_size, shift_step)

            X_train_seq, y_train = WindowsToNmp(train_data)
            X_val_seq,   y_val   = WindowsToNmp(val_data)
            X_test_seq,  y_test  = WindowsToNmp(test_data)
            X_train_seq, X_val_seq, X_test_seq, scaler = NormalizeStd(X_train_seq,X_val_seq,X_test_seq)
            
            TCN_results = TuneTCN(
                    X_train_seq,
                    y_train,
                    X_val_seq,
                    y_val,
                    search_space,
                    n_trials=20,
                    epochs=200,
                    patience=20,
                    seed=SEED
            )

            all_results.append({
                "window_size": window_size,
                "shift_step": shift_step,
                "scaler": scaler,
                "best_model": TCN_results["best_model"],
                "best_rmse": TCN_results["best_rmse"],
                "best_r2": TCN_results["best_r2"],
                "best_history": TCN_results["best_history"],
                **TCN_results["best_params"]
            })

    # Print best results for all window sizes and shift steps
    print("\n=== Summary of Best Results ===")
    for res in all_results:
        print(f"Window Size: {res['window_size']}, Shift Step: {res['shift_step']}, "
            f"Best RMSE: {res['best_rmse']:.4f}, Best R2: {res['best_r2']:.4f}, "
            f"Params: channels={res['channels']}, batch_size={res['batch_size']}, kernel_size={res['kernel_size']}, "
            f"dropout={res['dropout']}, lr={res['lr']}, fc_hidden={res['fc_hidden']}")

    # Sort results by best_rmse to find optimal window size and shift step
    all_results_sorted = sorted(all_results, key=lambda x: x["best_rmse"])

    # Get the best model from the best window size and shift step
    best_result = all_results_sorted[0]
    print(f"\nBest Window Size: {best_result['window_size']}, Best Shift Step: {best_result['shift_step']}")

    # Save all results to a CSV for further analysis
    results_df = pd.DataFrame(all_results_sorted)
    results_df.drop(columns=["best_model", "best_history", "scaler"]).to_csv(os.path.join(data_folder, "tcn_tuning_results.csv"), index=False)
    print(f"All tuning results saved to {os.path.join(data_folder, 'tcn_tuning_results.csv')}")

    #Plot best RMSE for each window size and shift step
    plt.figure(figsize=(10, 5))
    plt.bar(
        [f"w={w}, s={s}" for w, s in zip(results_df["window_size"], results_df["shift_step"])],
        results_df["best_rmse"]
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Validation RMSE")
    plt.title("Validation RMSE by Window/Shift Combination")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder,"TCN_gridsearch_val_rmse.png"), dpi=300)
    plt.close()

    #Plot best R2 for each window size and shift step
    plt.figure(figsize=(10, 5))
    plt.bar(
        [f"w={w}, s={s}" for w, s in zip(results_df["window_size"], results_df["shift_step"])],
        results_df["best_r2"]
        )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Validation R²")
    plt.title("Validation R² by Window/Shift Combination")
    plt.ylim(-0.5, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder,"TCN_gridsearch_val_r2.png"), dpi=300)
    plt.close()

    #Plot history of the best model
    best_history = best_result["best_history"]

    #Plot train rmse, val rmse and mae history of the best model
    plt.figure(figsize=(8,6))
    plt.plot(best_history["train_rmse"], label="Train RMSE")
    plt.plot(best_history["val_rmse"], label="Validation RMSE")
    plt.plot(best_history["val_mae"], label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Training Error Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, "training_errors_TCN.png"), dpi=300)
    plt.close()

    #Plot val r2 history of the best model
    plt.figure(figsize=(8,6))
    plt.plot(best_history["val_r2"], label="Validation R²", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("Validation R²")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, "training_r2_TCN.png"), dpi=300)
    plt.close()
    print(f"Training history plots saved to {figures_folder}")

    # Save the best model for future assessment
    model_name = "Optimized_TCN"

    save_NN_model(
        model = all_results_sorted[0]["best_model"],
        config = all_results_sorted[0]["best_model"].get_config(),
        save_dir = os.path.join(models_folder, model_name),
        extra_params={
            "window": all_results_sorted[0]["window_size"],
            "shift": all_results_sorted[0]["shift_step"]
        })
    print(f"Best model saved to {os.path.join(models_folder, model_name)}")

if __name__ == "__main__":
    main()