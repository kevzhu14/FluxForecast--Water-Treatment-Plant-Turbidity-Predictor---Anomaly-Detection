import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error 
import xgboost as xgb
import torch
from src.pipeline.DataProcessing import DataSplit
from XGBoost import EvaluateXG, plot_best_model, DataPrep
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

data_folder = 'data'
figures_folder = os.path.join('figures', 'xgb_results')
models_folder = 'saved_models'

target_col = "[Filt] Mean Turbidity [NTU]"

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
    # Extract best result from baseline to compare against optimized model
    baseline = pd.read_csv(os.path.join(data_folder, "xgboost_window_shift_results.csv"))
    best_baseline = baseline.iloc[0].to_dict()

    window_size = int(best_baseline["window_size"])
    shift_step = int(best_baseline["shift_step"])

    # Initialize data split and preprocess
    train_raw, val_raw, test_raw = DataSplit(os.path.join(data_folder, "WTP_raw_data.csv"))

    train_data, val_data, test_data, X_train, y_train, X_val, y_val, X_test, y_test = DataPrep(
            train_raw, val_raw, test_raw, window_size, shift_step)
    
    # Define hyperparameter search space for XGBoost
    space={'max_depth': hp.quniform("max_depth", 3, 8, 1),
        'gamma': hp.uniform('gamma', 0, 2),
        'subsample': hp.uniform('subsample', 0.6, 1),
        'reg_alpha' : hp.uniform('reg_alpha', 0.1, 5),
        'reg_lambda' : hp.uniform('reg_lambda', 5, 50),
        'learning_rate' : hp.uniform('learning_rate', 0.001, 0.05),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.6, 1),
        'min_child_weight' : hp.quniform('min_child_weight', 1, 5, 1),
        'n_estimators': hp.quniform('n_estimators', 100, 300, 10)}

    # Define optimization objective for Hyperopt
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
                        random_state = SEED,
                        eval_metric = ["mae", "rmse"],
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


    trials = Trials()

    # Run the hyperparameter optimization and get the best hyperparameters
    best_hyperparams = fmin(fn = optimization_objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 2000,
                        trials = trials,
                        rstate = np.random.default_rng(SEED))
                        
    print("\nBest Hyperparameters:", best_hyperparams)

    # Get the per-trial losses (validation MSE)
    losses = trials.losses()

    # Compute the cumulative best-so-far loss (monotonically non-increasing)
    best_so_far = np.minimum.accumulate(losses)

    # Plot the trial losses and best-so-far curve
    plt.subplots(figsize=(8, 6))
    plt.plot(losses, label="Trial loss (Val MSE)", alpha=0.6)
    plt.plot(best_so_far, label="Best loss so far", linewidth=2, color="tab:red")
    plt.xlabel("Trial")
    plt.ylabel("Loss (Val MSE)")
    plt.title("Hyperopt XGBoost Bayesian Optimization: Best Loss vs Trial")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(figures_folder, "bayesian_opt_loss_vs_trial.png"), dpi=300)
    print(f"Saved Bayesian optimization loss plot to {os.path.join(figures_folder, 'bayesian_opt_loss_vs_trial.png')}.")

    xgb_optimized_model = xgb.XGBRegressor(
        n_estimators=int(best_hyperparams['n_estimators']),
        max_depth=int(best_hyperparams['max_depth']),
        learning_rate=best_hyperparams['learning_rate'],
        subsample=best_hyperparams['subsample'],
        colsample_bytree=best_hyperparams['colsample_bytree'],
        reg_lambda=best_hyperparams['reg_lambda'],
        reg_alpha=int(best_hyperparams['reg_alpha']),
        gamma=best_hyperparams['gamma'],
        random_state=SEED,
        eval_metric=["mae", "rmse"],
        early_stopping_rounds=15
    )

    optimized_results = EvaluateXG(xgb_optimized_model, train_raw, val_raw, test_raw, window_size, shift_step)

    saved_hparameters_df = pd.DataFrame({
        'Hyperparameter': ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'reg_lambda', 'reg_alpha', 'gamma', 'min_child_weight', 'early_stopping_rounds'],
        'Baseline Model': [300, 3, 0.01, 1, 1.0, 10, 1, 0.0, 1, 50],
        'Bayesian Optimization': [int(best_hyperparams['n_estimators']), int(best_hyperparams['max_depth']), best_hyperparams['learning_rate'], best_hyperparams['subsample'], best_hyperparams['colsample_bytree'], best_hyperparams['reg_lambda'], int(best_hyperparams['reg_alpha']), best_hyperparams['gamma'], int(best_hyperparams['min_child_weight']), 15]
    })

    saved_opt_results_df = pd.DataFrame([{
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
        for r in [optimized_results, best_baseline]
    ])

    # Add a method column and make it the first column
    saved_opt_results_df["method"] = ["bayesian_optimization", "baseline"]
    cols = ["method"] + [c for c in saved_opt_results_df.columns if c != "method"]
    saved_opt_results_df = saved_opt_results_df[cols]

    saved_opt_results_df["val_rmse_gain_vs_naive"] = saved_opt_results_df["naive_val_rmse"] - saved_opt_results_df["val_rmse"]
    saved_opt_results_df["val_r2_gain_vs_naive"] = saved_opt_results_df["val_r2"] - saved_opt_results_df["naive_val_r2"]

    saved_opt_results_df.to_csv(os.path.join(data_folder, "xgb_test_metrics_comparison.csv"), index=False)
    saved_hparameters_df.to_csv(os.path.join(data_folder, "xgb_hyperparameters_comparison.csv"), index=False)
    print(f"Saved model comparison and hyperparameters to {data_folder}.")

    plot_best_model(optimized_results, model_name="optimized")

    # Save the optimized model
    model_name = "Optimized_XGBoost"
    save_dir = os.path.join(models_folder, model_name)
    save_xgb_model(
        xgb_optimized_model,
        xgb_optimized_model.get_params(),
        extra_params={
            "window": window_size,
            "shift": shift_step,
        },
        save_dir=save_dir
    )
    print(f"Best model saved to {save_dir}")

if __name__ == "__main__":
    main()