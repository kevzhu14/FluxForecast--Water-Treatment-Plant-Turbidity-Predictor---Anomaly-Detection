import numpy as np


def NaivePredClassifier(data, target_col, threshold=0.07):
    return np.array([
        1.0 if window[target_col].iloc[-1] > threshold else 0.0
        for window, target in data
    ])
def NaivePredRegression(data, target_col):
    return np.array([window[target_col].iloc[-1] for window, target in data])