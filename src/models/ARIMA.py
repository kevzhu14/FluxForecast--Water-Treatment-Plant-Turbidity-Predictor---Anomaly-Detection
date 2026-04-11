import sys
import os


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Import required functions from your existing processing file
from src.pipeline.DataProcessing import DataSplit, DataProcessing, SlidingWindowWithTarget

def evaluate_arima(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

def main():
    window_size = 14
    shift_step = 7
    target_col = "[Filt] Mean Turbidity [NTU]"
    data_path = os.path.join(repo_root, "data", "WTP_raw_data.csv")

    arima_order = (1, 1, 0) # Example parameters: (p, d, q)
    
    print("Loading and splitting data...")
    train_raw, val_raw, test_raw = DataSplit(data_path)
    
    # We only technically need to process the test slice if we are only forecasting on the test split, 
    # but processing all ensures consistency
    print("Running DataProcessing on subsets...")
    test_processed = DataProcessing(test_raw)
    
    print("Creating sliding windows...")
    test_data = SlidingWindowWithTarget(
        test_processed, 
        window_size, 
        shift_step, 
        task="regression", 
        target_col=target_col
    )
    
    y_true = []
    y_pred = []
    
    print(f"Running ARIMA on {len(test_data)} test windows...")
    for idx, (window, target) in enumerate(test_data):
        # Extract purely the sequence of target variable history within the sliding window
        ts = window[target_col].values
        y_true.append(target)
        
        try:
            # Fit an ARIMA model individually onto the historical horizon
            arima_model = ARIMA(ts, order=arima_order)
            fitted_model = arima_model.fit()
            # Predict the next step forward beyond the window
            forecast = fitted_model.forecast(steps=1)[0]
        except Exception:
            # Fallback to naive prediction (last measured value) if ARIMA iterative solver fails to converge
            forecast = ts[-1]
            
        y_pred.append(forecast)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(test_data)} windows...")
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = evaluate_arima(y_true, y_pred)
    
    print("\n=== ARIMA Evaluation on Test Set ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Plot results against the real values
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Real", color="black", linewidth=2)
    plt.plot(y_pred, label="ARIMA Predicted", color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Test Sample")
    plt.ylabel("Turbidity [NTU]")
    plt.title(f"ARIMA{arima_order} Univariate Predictions vs Real Data")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # PREPEND REPO ROOT TO FIGURES PATH
    figures_folder = os.path.join(repo_root, "figures")
    os.makedirs(figures_folder, exist_ok=True)
    save_path = os.path.join(figures_folder, "arima_predictions.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")

if __name__ == "__main__":
    main()