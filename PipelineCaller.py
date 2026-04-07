from Pipeline import (
    PrepareData,
    BuildModel,
    TrainAndPredict,
    EvaluateResults,
    MakeRunDirs,
    DisplayResults,
    PlotResults,
    SaveMetricsCSV,
    SavePredictionsCSV,
    SaveConfig
)
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from matplotlib import cm

SEED = 42

WINDOW_SIZE = 14
SHIFT_STEP = 7
TARGET_COL = "[Filt] Mean Turbidity [NTU]"

DATA_FOLDER = 'data'
FIGURES_FOLDER = os.path.join('figures', 'comparisons')
MODELS_FOLDER = 'saved_models'

CONFIG = {
    "task": "classification",          # "regression" or "classification"
    "target_col": TARGET_COL,
    "threshold": 0.07,
    "horizon": 3,
    "window_size": WINDOW_SIZE,
    "shift_step": SHIFT_STEP,
    "model_type": "xgboost",          # "naive", "xgboost", "cnn", "tcn"
    "prob_threshold": 0.5,
    "use_pretrained": False,
    "seed": SEED
}

COLOUR_SET = plt.get_cmap("Dark2").colors

MODEL_STYLES = {
    "naive":   {"color": COLOUR_SET[0], "marker": "o", "linestyle": "--"},
    "xgboost": {"color": COLOUR_SET[1], "marker": "s", "linestyle": "--"},
    "cnn":     {"color": COLOUR_SET[2], "marker": "^", "linestyle": "--"},
    "tcn":     {"color": COLOUR_SET[3], "marker": "D", "linestyle": "--"},
    "real":    {"color": "black", "marker": None, "linestyle": "-"}
}
order = ["naive", "xgboost", "cnn", "tcn"]

def RunPipeline(config, df):
    run_paths = MakeRunDirs(config)
    data = PrepareData(config, df)
    input_shape = data["X_train"].shape if config["model_type"] in ["cnn", "tcn"] else None
    model = BuildModel(config, input_shape)
    results = TrainAndPredict(model, data, config)
    metrics = EvaluateResults(results, config)
    DisplayResults(results, metrics, config, run_paths=run_paths)
    PlotResults(results, metrics, config, run_paths=run_paths)
    SaveMetricsCSV(metrics, config, run_paths)
    SavePredictionsCSV(results, run_paths)
    SaveConfig(config, run_paths)
    return {
        "data": data,
        "model": model,
        "results": results,
        "metrics": metrics,
        "paths": run_paths
    }

def SaveModelMetricFigure(pipeline_outputs, task, save_dir=FIGURES_FOLDER):
    os.makedirs(save_dir, exist_ok=True)
    rows = []

    if task == "regression":
        metric_info = ["R2", "RMSE"]
        for model_name, out in pipeline_outputs.items():
            m = out["metrics"]
            rows.append({
                "Model": model_name,
                "R2": m["r2"],
                "MAE": m["mae"],
                "MAPE": m["mape"],
                "RMSE": m["rmse"]
            })
    elif task == "classification":
        metric_info = ["Recall", "F1"]
        for model_name, out in pipeline_outputs.items():
            m = out["metrics"]
            rows.append({
                "Model": model_name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"]
            })
    df = pd.DataFrame(rows)
    df["Model"] = pd.Categorical(df["Model"], categories=order, ordered=True)
    df = df.sort_values("Model")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    for ax, metric in zip(axes, metric_info):
        bar_colors = [MODEL_STYLES[m]["color"] for m in df["Model"]]
        ax.barh(df["Model"], df[metric], color=bar_colors)
        ax.set_title(f"{metric} (Test)")
        ax.set_xlabel(metric)
        ax.grid(axis="x", linestyle="--", alpha=0.6)

    fig.suptitle(f"{task.capitalize()} Comparison on Test Set", fontsize=16)
    fig.text(0.512, 0.85, f"Window Size: {CONFIG['window_size']}, Shift Step: {CONFIG['shift_step']}", ha='center', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(save_dir, f"{task}_model_metric_comparison.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure to: {save_path}")

def PlotPredictedVsActual(all_results, save_dir=FIGURES_FOLDER):
    os.makedirs(save_dir, exist_ok=True)
    y_true = None
    preds = {}
    for model_name, out in all_results.items():
        y_true_model = np.asarray(out["results"]["y_true"]).flatten()
        y_pred_model = np.asarray(out["results"]["y_pred"]).flatten()

        if y_true is None:
            y_true = y_true_model

        preds[model_name] = y_pred_model
    colors = {
        "real": "black",
        "naive": "#55A868",
        "xgboost": "#4C72B0",
        "cnn": "#C44E52",
        "tcn": "#64B5F6"
    }
    min_val = min(y_true.min(), *(p.min() for p in preds.values()))
    max_val = max(y_true.max(), *(p.max() for p in preds.values()))
    plt.figure(figsize=(8, 6))
    for model_name in order:
        if model_name not in preds:
            continue
        y_pred = preds[model_name]
        style = MODEL_STYLES[model_name]
        plt.scatter(
            y_true,
            y_pred,
            alpha=0.7,
            label=model_name,
            marker=style["marker"],
            color=style["color"]
        )

    plt.plot([min_val, max_val], [min_val, max_val], "k--", label="Ideal")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual Comparison")
    plt.legend()
    plt.tight_layout()
    save_path_1 = os.path.join(save_dir, "predicted_vs_actual_models.png")
    plt.savefig(save_path_1, dpi=300)
    plt.close()
    print("Saved:", save_path_1)

    x = np.arange(1, len(y_true) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(
        x,
        y_true,
        label="real",
        color=MODEL_STYLES["real"]["color"],
        linestyle=MODEL_STYLES["real"]["linestyle"],
        linewidth=2.5
    )

    for model_name in order:
        if model_name not in preds:
            continue
        y_pred = preds[model_name]
        style = MODEL_STYLES[model_name]
        plt.plot(
            x,
            y_pred,
            label=model_name,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2
        )

    plt.xlabel("Sample")
    plt.ylabel("Turbidity")
    plt.title("Real vs Predicted Values for All Models")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_path_2 = os.path.join(save_dir, "real_vs_predicted_all_models.png")
    plt.savefig(save_path_2, dpi=300)
    plt.close()
    print("Saved:", save_path_2)

    for model_name in order:
        if model_name not in preds:
            continue

        y_pred = preds[model_name]
        style = MODEL_STYLES[model_name]

        plt.figure(figsize=(10, 5))
        plt.plot(
            x,
            y_true,
            label="real",
            color=MODEL_STYLES["real"]["color"],
            linestyle=MODEL_STYLES["real"]["linestyle"],
            linewidth=2.5
        )
        plt.plot(
            x,
            y_pred,
            label=model_name,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2
        )

        plt.xlabel("Sample")
        plt.ylabel("Turbidity")
        plt.title(f"Real vs Predicted ({model_name})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"real_vs_predicted_{model_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print("Saved:", save_path)
def PlotClassificationComparison(all_results, save_dir=FIGURES_FOLDER):
    os.makedirs(save_dir, exist_ok=True)

    models = [m for m in order if m in all_results]
    accuracy = [all_results[m]["metrics"]["accuracy"] for m in models]
    precision = [all_results[m]["metrics"]["precision"] for m in models]
    recall = [all_results[m]["metrics"]["recall"] for m in models]
    f1 = [all_results[m]["metrics"]["f1"] for m in models]

    metrics = [accuracy, precision, recall, f1]
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        plt.bar(
            x + i * width,
            metric,
            width,
            label=metric_names[i],
            color=COLOUR_SET[i % len(COLOUR_SET)]
        )

    plt.xticks(x + width * 1.5, models)
    plt.ylabel("Score")
    plt.title("Classification Model Comparison")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, "classification_model_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)


models = ["naive", "xgboost", "cnn", "tcn"]
outputs = {}

for m in models:

    config = CONFIG.copy()
    config["model_type"] = m

    if m == "cnn":
        config.update({ "hidden_channels": 64, "kernel_size": 7, "dropout": 0.1, "lr": 0.001, "batch_size": 16 })
    if m == "tcn":
        config.update({'channels': (32, 64, 128), 'kernel_size': 5, 'dropout': 0.0, 'lr': 0.003, 'batch_size': 16, 'fc_hidden': 128, 'weight_decay': 0.0, 'dilation_reset': 8, 'use_norm': 'layer_norm'})

    outputs[m] = RunPipeline(config, os.path.join(DATA_FOLDER, 'WTP_raw_data.csv'))

if config["task"] == "regression":
    PlotPredictedVsActual(outputs)
elif config["task"] == "classification":
    PlotClassificationComparison(outputs)
SaveModelMetricFigure(outputs, config["task"])