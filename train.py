import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.MultiModalFusion import MultiModalFusion
from dataprocess.dataprocess import load_data_from_csv, WeatherImageLoader
from my_utils.Trainer import Trainer, TrainerMixup
from my_utils.helper import compute_mape, compute_maermse, save_results_to_csv


class MAPE(nn.Module):
    """MAPE loss"""
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100


STATION_CFGS = [
    dict(
        name="Gympie",
        img_dir="/data/modified",
        file_path="/data/processed_Gympie_north1.csv",
        processed_dir="/data/processed_image",
        model_path="/results/model_saved/best_model_Gympie.pth",
        result_path="/results/predictions_Gympie.csv",
    ),
    dict(
        name="Coolum",
        img_dir="/data/modified_other",
        file_path="/data/processed_Coolum1.csv",
        processed_dir="/data/processed_image_other",
        model_path="/results/model_saved/best_model_Coolum.pth",
        result_path="/results/predictions_Coolum.csv",
    ),
    dict(
        name="Noosaville",
        img_dir="/data/modified_other",
        file_path="/data/processed_Noosaville1.csv",
        processed_dir="/data/processed_image_other",
        model_path="/results/model_saved/best_model_Noosaville.pth",
        result_path="/results/predictions_Noosaville.csv",
    ),
    dict(
        name="Tewantin",
        img_dir="/data/modified_other",
        file_path="/data/processed_Tewantin1.csv",
        processed_dir="/data/processed_image_other",
        model_path="/results/model_saved/best_model_Tewantin.pth",
        result_path="/results/predictions_Tewantin.csv",
    ),
]


def eval_offline(model, test_loader, scaler_y, device, save_csv_path=None, station_name=""):
    """Offline evaluation with optional CSV export."""
    model.eval()
    preds, gts = [], []

    with torch.no_grad():
        for weather, past, time, targets, year, month, day, hour in test_loader:
            weather = weather.to(device)
            past = past.to(device)
            time = time.to(device)
            year = year.to(device)
            month = month.to(device)
            day = day.to(device)
            hour = hour.to(device)
            targets = targets.to(device)

            y_hat = model(weather, past, time, year, month, day, hour)
            preds.append(y_hat.cpu().numpy())
            gts.append(targets.cpu().numpy())

    preds = np.concatenate(preds, axis=0).reshape(-1, 1)
    gts = np.concatenate(gts, axis=0).reshape(-1, 1)

    preds_denorm = scaler_y * preds
    gts_denorm = scaler_y * gts

    mape = compute_mape(preds, gts, scaler_y)
    rmse, mae = compute_maermse(preds, gts, scaler_y)
    print(f"[{station_name} Offline] MAPE: {mape:.3f}%, RMSE: {rmse:.3f}, MAE: {mae:.3f}")

    if save_csv_path:
        out_csv = save_csv_path.replace(".csv", "_offline.csv")
        save_results_to_csv(preds_denorm, gts_denorm, output_file=out_csv)
        print(f"[{station_name}] Offline predictions saved to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Offline training for MultiModalFusion")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--image_seq_len", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--sigma", type=float, default=5e-4)
    parser.add_argument("--offline_eval", action="store_true")
    parser.add_argument("--stations", type=str, default="Coolum,Noosaville,Tewantin")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.stations == "all":
        selected = STATION_CFGS
    else:
        wanted = set([s.strip() for s in args.stations.split(",")])
        selected = [cfg for cfg in STATION_CFGS if cfg["name"] in wanted]
        if not selected:
            raise ValueError(f"Invalid station names: {wanted}")

    for cfg in selected:
        name = cfg["name"]
        print(f"\n========== Offline training for {name} ==========")

        weather_loader = WeatherImageLoader(cfg["img_dir"], cfg["processed_dir"])
        train_loader, test_loader, scaler_y = load_data_from_csv(cfg["file_path"])

        model = MultiModalFusion(
            weather_loader=weather_loader,
            dim=args.dim,
            depth=args.depth,
            image_seq_len=args.image_seq_len
        ).to(device)

        criterion = MAPE()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        if args.no_mixup:
            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scaler_y=scaler_y,
                epochs=args.epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                scheduler=scheduler,
                save_dir=cfg["model_path"],
            )
        else:
            trainer = TrainerMixup(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scaler_y=scaler_y,
                epochs=args.epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                scheduler=scheduler,
                alpha=args.alpha,
                sigma=args.sigma,
                save_dir=cfg["model_path"],
            )

        os.makedirs(os.path.dirname(cfg["model_path"]), exist_ok=True)
        trainer.train()

        if args.offline_eval:
            best = MultiModalFusion(
                weather_loader=weather_loader,
                dim=args.dim,
                depth=args.depth,
                image_seq_len=args.image_seq_len
            ).to(device)
            state = torch.load(cfg["model_path"], map_location=device)
            best.load_state_dict(state)

            eval_loader = DataLoader(test_loader.dataset, batch_size=1, shuffle=False)
            eval_offline(best, eval_loader, scaler_y, device, save_csv_path=cfg["result_path"], station_name=name)


if __name__ == "__main__":
    main()
