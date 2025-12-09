import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from datetime import datetime, timedelta
from models.Mixer import SpatioTemporalMixerBlock
from models.GMMF import GatedMultimodalLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiModalFusion(nn.Module):
    """Multimodal fusion model using spatio-temporal mixers and gated fusion blocks."""

    def __init__(self, weather_loader, dim, depth, image_seq_len=6):
        super().__init__()
        self.weather_loader = weather_loader
        self.dim = dim
        self.depth = depth
        self.image_seq_len = image_seq_len

        self.patch_size = 16
        self.num_patch = (224 // self.patch_size) ** 2
        self.token_dim = 256
        self.time_dim = 64
        self.channel_dim = 2048

        # Image → patch embeddings
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(3, self.dim, self.patch_size, self.patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        # Spatio-temporal mixer blocks
        self.mixer_blocks = nn.ModuleList([
            SpatioTemporalMixerBlock(
                self.dim,
                self.num_patch,
                self.image_seq_len,
                self.token_dim,
                self.time_dim,
                self.channel_dim
            ) for _ in range(self.depth)
        ])

        self.mixer_outs = nn.ModuleList([
            nn.Linear(self.dim, self.dim) for _ in range(self.depth)
        ])

        self.image_fc = nn.Linear(self.dim, self.dim)

        # Numerical weather features
        self.numerical_weather_fc = nn.Sequential(
            nn.Linear(4, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        # Past load → TCN encoder
        self.tcn_load = nn.Sequential(
            nn.Conv1d(1, self.dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.dim, self.dim, 3, padding=1),
            nn.ReLU(),
        )

        # Past load MLP
        self.past_load_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        # Time features
        self.time_fc = nn.Sequential(
            nn.Linear(24, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        # Gated multimodal fusion blocks
        self.gmf_weather = nn.ModuleList([
            GatedMultimodalLayer(self.dim, self.dim, self.dim) for _ in range(self.depth)
        ])
        self.gmf_load = nn.ModuleList([
            GatedMultimodalLayer(self.dim, self.dim, self.dim) for _ in range(self.depth)
        ])
        self.gmf_final = nn.ModuleList([
            GatedMultimodalLayer(self.dim, self.dim, self.dim) for _ in range(self.depth)
        ])

        # Final prediction head
        self.output_fc = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, 1),
        )

    def forward(self, weather, past, time, year, month, day, hour, ind, lam, Modal=None):
        # Build timestamps per batch item
        timestamps = [
            datetime(
                int(year[i].item()),
                int(month[i].item()),
                int(day[i].item()),
                int(hour[i].item()),
            )
            for i in range(len(year))
        ]

        # Load image sequences
        seq_list = []
        for ts in timestamps:
            imgs = []
            for k in range(self.image_seq_len):
                ts_k = ts - timedelta(hours=self.image_seq_len - 1 - k)
                imgs.append(self.weather_loader.load_images_for_hour(ts_k))
            seq_list.append(torch.stack(imgs, dim=0))

        weather_feature = torch.stack(seq_list).to(device)

        # Mix-up for images (optional)
        if Modal == 1:
            weather_feature_j = weather_feature[ind]
            lam_expanded = lam.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            weather_feature = lam_expanded * weather_feature + (1 - lam_expanded) * weather_feature_j

        # Patch embeddings
        B, L, C, H, W = weather_feature.shape
        weather_feature = weather_feature.view(B * L, C, H, W)
        patch_embed = self.to_patch_embedding(weather_feature)
        weather_feature = patch_embed.view(B, L, self.num_patch, self.dim)

        # Spatio-temporal mixing over image patches
        image_outs = []
        for i in range(self.depth):
            weather_feature = self.mixer_blocks[i](weather_feature)
            out = weather_feature.mean(dim=2).mean(dim=1)  # global avg pooling
            image_outs.append(self.mixer_outs[i](out))

        image_out_final = self.image_fc(image_outs[-1])

        # Numerical weather embedding
        numerical_weather = self.numerical_weather_fc(weather)

        # Past load → TCN → MLP
        tcn_out = self.tcn_load(past.unsqueeze(1)).mean(dim=2)
        past_load = self.past_load_mlp(tcn_out)

        # Time embedding
        time = self.time_fc(time)

        # Weather stream: GMMU(im_t, n_t)
        k_t = image_out_final
        for i in range(self.depth):
            k_t = self.gmf_weather[i](k_t, numerical_weather)

        # Load stream: GMMU(h_t, c_t)
        L_t = past_load
        for i in range(self.depth):
            L_t = self.gmf_load[i](L_t, time)

        # Final multimodal fusion: GMMU(L_t, k_t)
        fusion = L_t
        for i in range(self.depth):
            fusion = self.gmf_final[i](fusion, k_t)

        return self.output_fc(fusion)
