import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.MultiModalFusion import MultiModalFusion
from my_utils.Trainer import Trainer, TrainerMixup
from my_utils.helper import compute_mape, compute_maermse, save_results_to_csv
from dataprocess.dataprocess import load_data_from_csv, WeatherImageLoader
from my_utils.online_mekf import MEKFOnlineAdapter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class MAPE(nn.Module):
    """MAPE loss."""
    def __init__(self):
        super(MAPE, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100


if __name__ == '__main__':
    torch.manual_seed(42)

    for i in range(0, 4):
        if i == 0:
            img_dir = "/data/modified"
            file_path = '/data/processed_Gympie_north1.csv'
            processed_dir = '/data/processed_image'
            model_dir = '/results/model_saved/best_model_Gympie.pth'
            results_dir = '/results/predictions_Gympie.csv'

        if i == 1:
            img_dir = "/data/modified_other"
            file_path = '/data/processed_Coolum1.csv'
            processed_dir = '/data/processed_image_other'
            model_dir = '/results/model_saved/best_model_Coolum.pth'
            results_dir = '/results/predictions_Coolum.csv'

        if i == 2:
            img_dir = "/data/modified_other"
            file_path = '/data/processed_Noosaville1.csv'
            processed_dir = '/data/processed_image_other'
            model_dir = '/results/model_saved/best_model_Noosaville.pth'
            results_dir = '/results/predictions_Noosaville.csv'

        if i == 3:
            img_dir = "/data/modified_other"
            file_path = '/data/processed_Tewantin1.csv'
            processed_dir = '/data/processed_image_other'
            model_dir = '/results/model_saved/best_model_Tewantin.pth'
            results_dir = '/results/predictions_Tewantin.csv'

        weather_loader = WeatherImageLoader(img_dir, processed_dir)
        train_loader, test_loader, scaler_y = load_data_from_csv(file_path)

        IMAGE_SEQ_LEN = 6

        model = MultiModalFusion(
            weather_loader=weather_loader,
            dim=64,
            depth=5,
            image_seq_len=IMAGE_SEQ_LEN
        ).to(device)

        criterion = MAPE()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        trainer = TrainerMixup(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler_y=scaler_y,
            epochs=500,
            train_loader=train_loader,
            test_loader=test_loader,
            scheduler=scheduler,
            alpha=0.75,
            sigma=0.0005,
            save_dir=model_dir
        )
        trainer.train()

        base_model = MultiModalFusion(
            weather_loader=weather_loader,
            dim=64,
            depth=5,
            image_seq_len=IMAGE_SEQ_LEN
        ).to(device)
        base_model.load_state_dict(torch.load(model_dir, map_location=device))

        online_test_loader = DataLoader(
            test_loader.dataset,
            batch_size=1,
            shuffle=False
        )

        mekf = MEKFOnlineAdapter(
            model=base_model,
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
        print(f'[Online] MAPE: {mape:.3f}%, RMSE: {rmse:.3f}, MAE: {mae:.3f}')

        save_results_to_csv(
            online_predictions_denorm,
            online_actuals_denorm,
            output_file=results_dir.replace(".csv", "_online.csv")
        )


