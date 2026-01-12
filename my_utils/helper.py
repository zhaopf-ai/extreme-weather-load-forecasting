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
        K = predictions.shape[1]
        cols = {}
        for k in range(K):
            cols[f'Actual_t+{k+1}'] = actuals[:, k]
            cols[f'Pred_t+{k+1}'] = predictions[:, k]
        df = pd.DataFrame(cols)
    else:
        raise ValueError(f"predictions shape not supported: {predictions.shape}")

    df.to_csv(output_file, index=False)
    print(f'Results saved to {output_file}')


def compute_mape(predictions, actuals, scaler_y):
    predictions = scaler_y*predictions
    actuals = scaler_y*actuals

    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    return mape

def compute_maermse(predictions, actuals, scaler_y):
    predictions = scaler_y*predictions
    actuals = scaler_y*actuals

    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mae = np.mean(np.abs(actuals - predictions))
    return rmse, mae
