import torch
from torch.utils.data import DataLoader
import numpy as np

from models.MultiModalFusion import MultiModalFusion
from dataprocess.dataprocess import load_data_from_csv, WeatherImageLoader
from my_utils.online_mekf import MEKFOnlineAdapter
from my_utils.helper import compute_mape, compute_maermse, save_results_to_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMAGE_SEQ_LEN = 6


def run_station(name, img_dir, file_path, processed_dir, model_path, result_path):
    print(f"\n========== Online MEKF test for {name} ==========")
    weather_loader = WeatherImageLoader(img_dir, processed_dir)
    train_loader, test_loader, scaler_y = load_data_from_csv(file_path)
    model = MultiModalFusion(
        weather_loader=weather_loader,
        dim=64,
        depth=5,
        image_seq_len=IMAGE_SEQ_LEN
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    online_test_loader = DataLoader(
        test_loader.dataset,
        batch_size=1,
        shuffle=False
    )
    mekf = MEKFOnlineAdapter(
        model=model,
        R=0.08662668812005468,
        Q0=1.2380651884969045e-07,
        mu_v= 0.7246679759335559,
        mu_p=0.8716768561208147,
        lamb=0.9988935922348762,
        delta=1.1761089966153553e-06
    )

    online_predictions = []
    online_actuals = []
    for weather, past, time, targets, year, month, day, hour in online_test_loader:
        y_hat, y_true = mekf.step(
            weather, past, time,
            year, month, day, hour,
            targets
        )
        online_predictions.append(y_hat.item())
        online_actuals.append(y_true.item())
    online_predictions = np.array(online_predictions).reshape(-1, 1)
    online_actuals = np.array(online_actuals).reshape(-1, 1)

    online_predictions_denorm = scaler_y * online_predictions
    online_actuals_denorm = scaler_y * online_actuals

    mape = compute_mape(online_predictions, online_actuals, scaler_y)
    rmse, mae = compute_maermse(online_predictions, online_actuals, scaler_y)

    print(f"[{name} Online] MAPE: {mape:.3f}%, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    online_result_path = result_path.replace(".csv", "_online.csv")
    save_results_to_csv(
        online_predictions_denorm,
        online_actuals_denorm,
        output_file=online_result_path
    )
    print(f"[{name}] Online predictions saved to: {online_result_path}")


if __name__ == "__main__":
    run_station(
        name="Gympie",
        img_dir="/data/modified",
        file_path = '/data/processed_Gympie_north1.csv',
        processed_dir = '/data/processed_image',
        model_dir = '/results/model_saved/best_model_Gympie.pth',
        results_dir = '/results/predictions_Gympie.csv',
    )

    run_station(
        name="Coolum",
        img_dir="/data/modified_other",
        file_path="/data/processed_Coolum1.csv",
        processed_dir="/data/processed_image_other",
        model_path="/results/model_saved/best_model_Coolum.pth",
        result_path="/results/predictions_Coolum.csv",
    )

    run_station(
        name="Noosaville",
        img_dir="/data/modified_other",
        file_path="/data/processed_Noosaville1.csv",
        processed_dir="/data/processed_image_other",
        model_path="/results/model_saved/best_model_Noosaville.pth",
        result_path="/results/predictions_Noosaville.csv",
    )

    run_station(
        name="Tewantin",
        img_dir="/data/modified_other",
        file_path="/data/processed_Tewantin1.csv",
        processed_dir="/data/processed_image_other",
        model_path="/results/model_saved/best_model_Tewantin.pth",
        result_path="/results/predictions_Tewantin.csv",
    )
