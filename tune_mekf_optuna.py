import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

import optuna

from models.MultiModalFusion import MultiModalFusion
from dataprocess.dataprocess import load_data_from_csv, WeatherImageLoader
from my_utils.online_mekf import MEKFOnlineAdapter
from my_utils.helper import compute_mape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMAGE_SEQ_LEN = 6
STATIONS = [
    dict(
        name="Gympie",
        img_dir="/data/modified",
        file_path="/data/processed_Gympie_north1.csv",
        processed_dir="/data/processed_image",
        model_path="/results/model_saved/best_model_Gympie.pth",
    ),
    dict(
        name="Coolum",
        img_dir="/data/modified_other",
        file_path="/data/processed_Coolum1.csv",
        processed_dir="/data/processed_image_other",
        model_path="/results/model_saved/best_model_Coolum.pth",
    ),
    dict(
        name="Noosaville",
        img_dir="/data/modified_other",
        file_path="/data/processed_Noosaville1.csv",
        processed_dir="/data/processed_image_other",
        model_path="/results/model_saved/best_model_Noosaville.pth",
    ),
    dict(
        name="Tewantin",
        img_dir="/data/modified_other",
        file_path="/data/processed_Tewantin1.csv",
        processed_dir="/data/processed_image_other",
        model_path="/results/model_saved/best_model_Tewantin.pth",
    ),
]

def run_station_once(station_cfg, mekf_params):
    """
    station_cfg: STATIONS 里的一个 dict
    mekf_params: dict, 传给 MEKFOnlineAdapter 的超参数
    """
    name = station_cfg["name"]
    img_dir = station_cfg["img_dir"]
    file_path = station_cfg["file_path"]
    processed_dir = station_cfg["processed_dir"]
    model_path = station_cfg["model_path"]

    print(f"\n[Station: {name}] Loading data & model...")
    weather_loader = WeatherImageLoader(img_dir, processed_dir)
    train_loader, test_loader, scaler_y = load_data_from_csv(file_path)

    model = MultiModalFusion(
        weather_loader=weather_loader,
        dim=64,
        depth=5,
        image_seq_len=IMAGE_SEQ_LEN,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    online_test_loader = DataLoader(
        test_loader.dataset,
        batch_size=1,
        shuffle=False,
    )
    mekf = MEKFOnlineAdapter(
        model=model,
        R=mekf_params["R"],
        Q0=mekf_params["Q0"],
        mu_v=mekf_params["mu_v"],
        mu_p=mekf_params["mu_p"],
        lamb=mekf_params["lamb"],
        delta=mekf_params["delta"],
    )
    online_predictions = []
    online_actuals = []

    for weather, past, time, targets, year, month, day, hour in online_test_loader:
        y_hat, y_true = mekf.step(
            weather, past, time,
            year, month, day, hour,
            targets,
        )
        online_predictions.append(y_hat.item())
        online_actuals.append(y_true.item())
    online_predictions = np.array(online_predictions).reshape(-1, 1)
    online_actuals = np.array(online_actuals).reshape(-1, 1)

    mape = compute_mape(online_predictions, online_actuals, scaler_y)

    print(f"[Station: {name}] Online MAPE = {mape:.3f}%")
    return float(mape)

def objective(trial: optuna.Trial) -> float:
    start_time = time.time()

    mekf_params = {
        "R": trial.suggest_float("R", 1e-4, 1e-1, log=True),
        "Q0": trial.suggest_float("Q0", 1e-7, 1e-3, log=True),
        "delta": trial.suggest_float("delta", 1e-8, 1e-4, log=True),
        "mu_v": trial.suggest_float("mu_v", 0.5, 0.99),
        "mu_p": trial.suggest_float("mu_p", 0.5, 0.99),
        "lamb": trial.suggest_float("lamb", 0.90, 0.999),
    }

    print(f"\n========== Trial {trial.number} ==========")
    print("MEKF hyperparameters:", mekf_params)

    mape_list = []

    try:
        for station_cfg in STATIONS:
            mape = run_station_once(station_cfg, mekf_params)
            mape_list.append(mape)
        avg_mape = float(np.mean(mape_list))
        print(f"[Trial {trial.number}] Avg MAPE over 4 stations = {avg_mape:.3f}%")

    except Exception as e:
        print(f"[Trial {trial.number}] FAILED with error: {repr(e)}")
        avg_mape = 1e6
    elapsed = time.time() - start_time
    print(f"[Trial {trial.number}] Elapsed time: {elapsed:.2f} seconds")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_mape

if __name__ == "__main__":
    N_TRIALS = 150
    study = optuna.create_study(
        study_name="mekf_hparam_tuning",
        direction="minimize",
    )
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        n_jobs=1,
    )

    print("\n========== OPTUNA RESULT ==========")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best Avg MAPE: {study.best_value:.6f}%")
    print("Best hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

