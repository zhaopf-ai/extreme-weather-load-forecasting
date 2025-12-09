import torch
import torch.nn as nn
import torch.optim as optim
from models.MultiModalFusion import MultiModalFusion
from my_utils.Trainer import TrainerMixup
from my_utils.helper import compute_mape, compute_maermse, save_results_to_csv
from dataprocess.dataprocess import load_data_from_csv, WeatherImageLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class MAPE(nn.Module):
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

        state_dict = torch.load(model_dir, map_location=device)
        model.load_state_dict(state_dict)

        criterion = MAPE()
        dummy_optimizer = optim.Adam(model.parameters(), lr=0.001)
        dummy_scheduler = ReduceLROnPlateau(dummy_optimizer, mode='min', factor=0.5, patience=10, verbose=False)

        tester = TrainerMixup(
            model=model,
            criterion=criterion,
            optimizer=dummy_optimizer,
            scaler_y=scaler_y,
            epochs=1,
            train_loader=train_loader,
            test_loader=test_loader,
            scheduler=dummy_scheduler,
            alpha=0.75,
            sigma=0.0005,
            save_dir=model_dir
        )

        predictions, actuals = tester.test(model)
        mape = compute_mape(predictions, actuals, scaler_y)
        rmse, mae = compute_maermse(predictions, actuals, scaler_y)
        print(f'[Offline Test] Station {i}, MAPE: {mape:.3f}%, RMSE: {rmse:.3f}, MAE: {mae:.3f}')
        predictions_denorm = scaler_y * predictions
        actuals_denorm = scaler_y * actuals

        save_results_to_csv(predictions_denorm, actuals_denorm, output_file=results_dir)
        print(f'Offline predictions saved to: {results_dir}')
