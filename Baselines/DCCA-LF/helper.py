import torch
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_results_to_csv(predictions, actuals, output_file='results.csv'):
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)

    if predictions.ndim == 1:
        df = pd.DataFrame({'Actual': actuals, 'Predicted': predictions})
    elif predictions.ndim == 2:
        k = predictions.shape[1]
        cols = {}
        for i in range(k):
            cols[f'Actual_t+{i+1}'] = actuals[:, i]
            cols[f'Pred_t+{i+1}'] = predictions[:, i]
        df = pd.DataFrame(cols)
    else:
        raise ValueError(f"predictions shape not supported: {predictions.shape}")

    df.to_csv(output_file, index=False)
    print(f'Results saved to {output_file}')


def compute_mape(predictions, actuals, scaler_y):
    predictions = scaler_y * predictions
    actuals = scaler_y * actuals
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    return mape


def compute_maermse(predictions, actuals, scaler_y):
    predictions = scaler_y * predictions
    actuals = scaler_y * actuals
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mae = np.mean(np.abs(actuals - predictions))
    return rmse, mae
