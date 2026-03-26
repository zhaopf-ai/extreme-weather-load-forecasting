import math
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.Mixer import SpatioTemporalMixerBlock, ResNet3DEncoder, ViT
from models.GMMF import GatedMultimodalLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiModalFusion(nn.Module):
    def __init__(
        self,
        weather_loader,
        dim,
        depth,
        image_seq_len=6,
        image_backbone="mixer",
        vit_depth=4,
        vit_heads=4,
        past_len=17,
        daily_len=7,
        weekly_len=4,
        pred_len=1,
        time_feat_dim=24,
        weather_feat_dim=4,
    ):
        super().__init__()

        self.weather_loader = weather_loader
        self.dim = dim
        self.depth = depth
        self.image_seq_len = image_seq_len
        self.past_len = past_len
        self.daily_len = daily_len
        self.weekly_len = weekly_len
        self.pred_len = pred_len

        self.patch_size = 16
        self.num_patch = (224 // self.patch_size) ** 2
        self.token_dim = 256
        self.time_dim = 16
        self.channel_dim = 2048

        weather_in_dim = weather_feat_dim * pred_len
        time_in_dim = time_feat_dim * pred_len
        daily_in_dim = daily_len * pred_len

        self.image_backbone = image_backbone.lower()
        self.image_fc = nn.Linear(dim, dim)

        if self.image_backbone == "mixer":
            self.to_patch_embedding = nn.Sequential(
                nn.Conv2d(3, dim, self.patch_size, self.patch_size),
                Rearrange("b c h w -> b (h w) c"),
            )

            self.mixer_blocks = nn.ModuleList([
                SpatioTemporalMixerBlock(
                    dim,
                    self.num_patch,
                    image_seq_len,
                    self.token_dim,
                    self.time_dim,
                    self.channel_dim,
                )
                for _ in range(depth)
            ])

            self.mixer_outs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])

        elif self.image_backbone == "cnn3d":
            self.cnn3d = ResNet3DEncoder(
                depth=18,
                in_ch=3,
                base_channels=75,
                temporal_downsample=False,
                out_dim=dim,
            )

        elif self.image_backbone == "vit":
            self.vit_frame = ViT(
                img_size=224,
                patch_size=self.patch_size,
                in_ch=3,
                dim=dim,
                depth=int(vit_depth),
                heads=int(vit_heads),
                mlp_ratio=4.0,
                dropout=0.25,
                use_cls_token=True,
            )

        else:
            raise ValueError(f"Unsupported image_backbone: {image_backbone}")

        self.numerical_weather_fc = nn.Sequential(
            nn.Linear(weather_in_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.tcn_past = nn.Sequential(
            nn.Conv1d(1, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.ReLU(),
        )

        self.tcn_daily = nn.Sequential(
            nn.Conv1d(daily_len, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.ReLU(),
        )

        self.tcn_weekly = nn.Sequential(
            nn.Conv1d(weekly_len, dim, 2, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 2, padding=1),
            nn.ReLU(),
        )

        self.past_load_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.daily_load_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.d_mlp = nn.Sequential(
            nn.Linear(daily_in_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.weekly_load_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.hist_fusion_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.time_fc = nn.Sequential(
            nn.Linear(time_in_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.gmf_weather = nn.ModuleList([
            GatedMultimodalLayer(dim, dim, dim) for _ in range(depth)
        ])
        self.gmf_load = nn.ModuleList([
            GatedMultimodalLayer(dim, dim, dim) for _ in range(depth)
        ])
        self.gmf_final = nn.ModuleList([
            GatedMultimodalLayer(dim, dim, dim) for _ in range(depth)
        ])

        self.output_fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, pred_len),
        )

    def forward(self, weather, past, daily, weekly, time, year, month, day, hour, ind, lam, Modal=None):
        batch_size = year.size(0)
        timestamps = [
            datetime(
                int(year[i].item()),
                int(month[i].item()),
                int(day[i].item()),
                int(hour[i].item()),
            )
            for i in range(batch_size)
        ]

        seq = []
        for ts in timestamps:
            imgs = [
                self.weather_loader.load_images_for_hour(
                    ts - timedelta(hours=self.image_seq_len - 1 - k)
                )
                for k in range(self.image_seq_len)
            ]
            seq.append(torch.stack(imgs, dim=0))

        weather_feature = torch.stack(seq).to(device)

        if Modal == 1:
            weather_feature_j = weather_feature[ind]
            lam = lam.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            weather_feature = lam * weather_feature + (1 - lam) * weather_feature_j

        if self.image_backbone == "mixer":
            B, L, C, H, W = weather_feature.shape
            x = weather_feature.view(B * L, C, H, W)
            x = self.to_patch_embedding(x).view(B, L, self.num_patch, self.dim)

            for i in range(self.depth):
                x = self.mixer_blocks[i](x)
            image_out = self.image_fc(x.mean(dim=(1, 2)))

        elif self.image_backbone == "cnn3d":
            x = weather_feature.permute(0, 2, 1, 3, 4)
            image_out = self.image_fc(self.cnn3d(x))

        elif self.image_backbone == "vit":
            B, L, C, H, W = weather_feature.shape
            x = weather_feature.view(B * L, C, H, W)
            x = self.vit_frame(x).view(B, L, self.dim)
            image_out = self.image_fc(x.mean(dim=1))

        weather = weather.reshape(weather.size(0), -1)
        numerical_weather = self.numerical_weather_fc(weather)

        past_feat = self.past_load_mlp(self.tcn_past(past.unsqueeze(1)).mean(dim=2))

        daily_tcn_in = daily.transpose(1, 2).contiguous()
        weekly_tcn_in = weekly.transpose(1, 2).contiguous()

        daily_feat = self.daily_load_mlp(self.tcn_daily(daily_tcn_in).mean(dim=2))
        weekly_feat = self.weekly_load_mlp(self.tcn_weekly(weekly_tcn_in).mean(dim=2))

        daily = daily.reshape(daily.size(0), -1)
        L_d = self.d_mlp(daily)

        hist_feat = self.hist_fusion_mlp(torch.cat([past_feat, daily_feat, weekly_feat], dim=1))

        time = time.reshape(time.size(0), -1)
        time = self.time_fc(time)

        k_t = image_out
        for g in self.gmf_weather:
            k_t = g(k_t, numerical_weather)

        L_t = hist_feat
        for g in self.gmf_load:
            L_t = g(L_t, time)

        fusion = L_t
        for g in self.gmf_final:
            fusion = g(fusion, k_t)

        fusion = torch.cat([fusion, L_d], dim=1)

        return self.output_fc(fusion)
