import pandas as pd
import numpy as np

# 定义误差指标计算函数
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
mode = "offline"
if mode == "online":
    print("在线模型评估结果：")
elif mode == "offline":
    print("离线模型评估结果：")
else:
    raise ValueError("mode 必须是 'online' 或 'offline'！")

base_dir = "results"

csv_paths = {}

for site in ["Gympie", "Coolum", "Noosaville", "Tewantin"]:
    if mode == "online":
        csv_paths[site] = f"{base_dir}/predictions_{site}_online.csv"
    else:  # offline
        csv_paths[site] = f"{base_dir}/predictions_{site}.csv"
for name, path in csv_paths.items():
    df = pd.read_csv(path)

    actual = df.iloc[:, 0]
    predicted = df.iloc[:, 1]

    mae_val = mae(actual, predicted)
    rmse_val = rmse(actual, predicted)
    mape_val = mape(actual, predicted)

    print(f"===== {name} =====")
    print(f"MAE:  {mae_val:.4f}")
    print(f"RMSE: {rmse_val:.4f}")
    print(f"MAPE: {mape_val:.2f}%\n")
