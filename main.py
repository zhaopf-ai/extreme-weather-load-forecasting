from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataprocess.dataprocess import WeatherImageLoader, load_data
from models.MultiModalFusion import MultiModalFusion
from my_utils.Trainer import TrainerMixup, Trainer
from my_utils.helper import compute_maermse, compute_mape, save_results_to_csv
from my_utils.online_mekf import MEKFOnlineAdapter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # ---- dataset / paths ----
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["Gympie"],
        choices=["Gympie"],
        help="Which datasets to run ",
    )
    p.add_argument("--data_root", type=str, default="D:\python project\image extreme weather\data")
    p.add_argument("--exp_root", type=str, default="D:\python project\image_extreme_weather_v2/results")

    # ---- data build ----
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--his_len", type=int, default=24)
    p.add_argument("--pre_len", type=int, default=9)
    p.add_argument("--add_weather_noise", action="store_true", default=True)
    p.add_argument("--no_weather_noise", dest="add_weather_noise", action="store_false")
    p.add_argument("--noise_seed", type=int, default=42)

    p.add_argument("--train_end_date", type=str, default="2021-11-30")
    p.add_argument("--val_start_date", type=str, default="2021-12-01")
    p.add_argument("--val_end_date", type=str, default="2021-12-31")
    p.add_argument("--test_start_date", type=str, default="2022-01-01")
    p.add_argument("--test_end_date", type=str, default="2022-02-10")

    # ---- model ----
    p.add_argument("--image_seq_len", type=int, default=6)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=5)

    p.add_argument("--image_backbone", default="mixer", choices=["mixer", "cnn3d", "vit"])
    p.add_argument("--vit_depth", type=int, default=5)
    p.add_argument("--vit_heads", type=int, default=4)

    # ---- training ----
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--scheduler_patience", type=int, default=10)

    p.add_argument("--mixup_alpha", type=float, default=0.75)
    p.add_argument("--mixup_sigma", type=float, default=5e-4)

    # ---- online MEKF ----
    p.add_argument("--mekf_R", type=float, default=0.08)
    p.add_argument("--mekf_Q0", type=float, default=0.0)
    p.add_argument("--mekf_mu_v", type=float, default=0.70)
    p.add_argument("--mekf_mu_p", type=float, default=0.80)
    p.add_argument("--mekf_lamb", type=float, default=1.0)
    p.add_argument("--mekf_delta", type=float, default=0.0)

    # ---- misc ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_flops", action="store_false")
    p.add_argument("--max_print_flops_table", action="store_false")

    return p


# -------------------------
# Dataset config
# -------------------------
@dataclass(frozen=True)
class DatasetConfig:
    name: str
    img_subdir: str
    data_file: str
    processed_subdir: str


DATASET_TABLE: Dict[str, DatasetConfig] = {
    "Gympie": DatasetConfig(
        name="Gympie",
        img_subdir="modified",
        data_file="Gym.npz",
        processed_subdir="processed_image",
    ),
    "Coolum": DatasetConfig(
        name="Coolum",
        img_subdir="modified_other",
        data_file="Coo.npz",
        processed_subdir="processed_image_other",
    ),
    "Noosaville": DatasetConfig(
        name="Noosaville",
        img_subdir="modified_other",
        data_file="Noo.npz",
        processed_subdir="processed_image_other",
    ),
    "Tewantin": DatasetConfig(
        name="Tewantin",
        img_subdir="modified_other",
        data_file="Tew.npz",
        processed_subdir="processed_image_other",
    ),
}


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_flops_params(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_print_table: bool = False,
    modal_value: int = 0,
) -> str:
    model.eval()

    weather, past, time_feat, targets, year, month, day, hour = next(iter(loader))
    bs = weather.shape[1]

    ind = torch.arange(bs, device=device, dtype=torch.long)
    lam = torch.ones((bs, 1), device=device, dtype=torch.float32)

    inputs = (
        weather.to(device),
        past.to(device),
        time_feat.to(device),
        year.to(device),
        month.to(device),
        day.to(device),
        hour.to(device),
        ind,
        lam,
        int(modal_value),
    )

    with torch.no_grad():
        flops = FlopCountAnalysis(model, inputs)
        total_flops = flops.total()

    flops_per_sample = total_flops / bs

    msg1 = f"FLOPs per forward (batch={bs}): {total_flops/1e9:.4f} GFLOPs"
    msg2 = f"FLOPs per sample: {flops_per_sample/1e9:.4f} GFLOPs"
    flops_log_text = msg1 + "\n" + msg2

    print("\n========== Model Complexity (fvcore) ==========")
    print(parameter_count_table(model))
    print(msg1)
    print(msg2)
    if max_print_table:
        print(flop_count_table(flops))
    print("=============================================\n")

    return flops_log_text



class MAPE(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100


def run_one_dataset(cfg: DatasetConfig, args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    exp_root = Path(args.exp_root)
    exp_root.mkdir(parents=True, exist_ok=True)

    img_dir = data_root / cfg.img_subdir
    data_path = data_root / cfg.data_file
    processed_dir = data_root / cfg.processed_subdir

    run_tag = f"{cfg.name}_{args.image_backbone}"

    model_path = exp_root / "model_saved" / f"best_model_{run_tag}.pth"
    pred_path = exp_root / f"predictions_{run_tag}.csv"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n================ Dataset: {cfg.name} ================")
    print(f"img_dir      : {img_dir}")
    print(f"data_path    : {data_path}")
    print(f"processed_dir: {processed_dir}")
    print(f"model_path   : {model_path}")
    print(f"pred_path    : {pred_path}")
    print("====================================================\n")

    weather_loader = WeatherImageLoader(str(img_dir), str(processed_dir))

    train_loader, val_loader, test_loader, scaler_y, time_feat_dim = load_data(
        str(data_path),
        batch_size=args.batch_size,
        his_length=args.his_len,
        pre_length=args.pre_len,
        add_weather_noise=args.add_weather_noise,
        noise_seed=args.noise_seed,
        train_end_date=args.train_end_date,
        val_start_date=args.val_start_date,
        val_end_date=args.val_end_date,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
    )

    model = MultiModalFusion(
        weather_loader=weather_loader,
        dim=args.dim,
        depth=args.depth,
        image_seq_len=args.image_seq_len,
        past_len=args.his_len,
        pred_len=args.pre_len,
        time_feat_dim=time_feat_dim,
        image_backbone=args.image_backbone,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
    ).to(device)

    total_params, trainable_params = count_params(model)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    flops_log_text = ""

    if args.print_flops:
        flops_loader = DataLoader(train_loader.dataset, batch_size=1, shuffle=False)
        flops_log_text = print_flops_params(
            model,
            flops_loader,
            device,
            max_print_table=bool(args.max_print_flops_table),
            modal_value=0,
        )


    criterion = MAPE()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        verbose=True,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler_y=scaler_y,
        epochs=args.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        scheduler=scheduler,
        save_dir=str(model_path),
        run_tag=run_tag,
        log_dir=str(exp_root / "logs"),     
        extra_log_header=flops_log_text,    
    )
    trainer.train()


    base_model = MultiModalFusion(
        weather_loader=weather_loader,
        dim=args.dim,
        depth=args.depth,
        image_seq_len=args.image_seq_len,
        past_len=args.his_len,
        pred_len=args.pre_len,
        time_feat_dim=time_feat_dim,
        weather_feat_dim=4,
        image_backbone=args.image_backbone,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
    ).to(device)
    base_model.load_state_dict(torch.load(model_path, map_location=device))

    online_test_loader = DataLoader(test_loader.dataset, batch_size=1, shuffle=False)

    mekf = MEKFOnlineAdapter(
        model=base_model,
        R=args.mekf_R,
        Q0=args.mekf_Q0,
        mu_v=args.mekf_mu_v,
        mu_p=args.mekf_mu_p,
        lamb=args.mekf_lamb,
        delta=args.mekf_delta,
    )

    online_predictions: List[np.ndarray] = []
    online_actuals: List[np.ndarray] = []
    online_times: List[float] = []

    for weather, past, time_feat, targets, year, month, day, hour in online_test_loader:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        y_hat, y_true = mekf.step(
            weather,
            past,
            time_feat,
            year,
            month,
            day,
            hour,
            targets,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        online_times.append(time.perf_counter() - t0)

        online_predictions.append(y_hat.numpy())
        online_actuals.append(y_true.numpy())

    online_times_arr = np.asarray(online_times, dtype=np.float64)
    print(f"[Online] avg time per sample: {online_times_arr.mean()*1000:.2f} ms ({online_times_arr.mean():.6f} s)")

    time_file = pred_path.with_name(pred_path.stem + "_online_time.csv")
    np.savetxt(time_file, online_times_arr, delimiter=",", header="time_s", comments="")
    print(f"[Online] per-sample time saved to: {time_file}")

    online_predictions_arr = np.stack(online_predictions, axis=0)
    online_actuals_arr = np.stack(online_actuals, axis=0)

    online_predictions_denorm = scaler_y * online_predictions_arr
    online_actuals_denorm = scaler_y * online_actuals_arr

    mape = compute_mape(online_predictions_arr, online_actuals_arr, scaler_y)
    rmse, mae = compute_maermse(online_predictions_arr, online_actuals_arr, scaler_y)
    print(f"[Online] MAPE: {mape:.3f}%, RMSE: {rmse:.3f}, MAE: {mae:.3f}")

    out_file = pred_path.with_name(pred_path.stem + "_online.csv")
    save_results_to_csv(
        online_predictions_denorm,
        online_actuals_denorm,
        output_file=str(out_file),
    )


def main() -> None:
    args = build_arg_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    for name in args.datasets:
        cfg = DATASET_TABLE[name]
        run_one_dataset(cfg, args)


if __name__ == "__main__":
    main()
