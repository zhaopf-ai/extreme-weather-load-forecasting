import math
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.Mixer import SpatioTemporalMixerBlock, ResNet3DEncoder, ViT
from models.GMMF import GatedMultimodalLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiModalFusion(nn.Module):
    """Multimodal forecasting model with image, weather, load, and time inputs."""

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
        self.pred_len = pred_len

        self.patch_size = 16
        self.num_patch = (224 // self.patch_size) ** 2
        self.token_dim = 256
        self.time_dim = 16
        self.channel_dim = 2048

        weather_in_dim = weather_feat_dim * pred_len
        time_in_dim = time_feat_dim * pred_len

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
            self.cnn3d = ResNet3DEncoder(depth=18, in_ch=3, base_channels=75, temporal_downsample=False, out_dim=dim)


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

        self.tcn_load = nn.Sequential(
            nn.Conv1d(1, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.ReLU(),
        )

        self.past_load_mlp = nn.Sequential(
            nn.Linear(dim, dim),
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
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, pred_len),
        )

    def forward(self, weather, past, time, year, month, day, hour, ind, lam, Modal=None):
        timestamps = [
            datetime(
                int(year[i]), int(month[i]), int(day[i]), int(hour[i])
            )
            for i in range(len(year))
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

        numerical_weather = self.numerical_weather_fc(weather)

        tcn_out = self.tcn_load(past.unsqueeze(1)).mean(dim=2)
        past_load = self.past_load_mlp(tcn_out)

        time = self.time_fc(time)

        k_t = image_out
        for g in self.gmf_weather:
            k_t = g(k_t, numerical_weather)

        L_t = past_load
        for g in self.gmf_load:
            L_t = g(L_t, time)

        fusion = L_t
        for g in self.gmf_final:
            fusion = g(fusion, k_t)

        return self.output_fc(fusion)
