import math
from datetime import datetime, timedelta
from typing import List, Optional

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ProbSparseSelfAttention(nn.Module):
    """
    Proportional sparse sampling:
    top-u queries, u = factor * L
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        factor: float = 0.6,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        x = x.view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        sparsity = scores.max(dim=-1).values - scores.mean(dim=-1)

        u = max(1, min(l, int(self.factor * l)))
        top_idx = torch.topk(sparsity, k=u, dim=-1).indices

        q_sel = torch.gather(
            q, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head)
        )
        scores_sel = torch.matmul(q_sel, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = self.dropout(torch.softmax(scores_sel, dim=-1))
        out_sel = torch.matmul(attn, v)

        out = torch.zeros_like(q)
        out.scatter_(
            2, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head), out_sel
        )
        out = out.transpose(1, 2).contiguous().view(b, l, self.d_model)
        return self.o_proj(out)


class InformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        factor: float,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.attn = ProbSparseSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            factor=factor,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.ELU()
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pool_stride: int) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))

        y = self.conv(x.transpose(1, 2))
        y = self.act(y)
        y = F.max_pool1d(y, kernel_size=3, stride=pool_stride, padding=1)

        x = self.norm2(y.transpose(1, 2))
        x = self.norm3(x + self.ffn(x))
        return x


class InformerEncoder(nn.Module):

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        factor: float = 0.6,
        d_ff: int = 2048,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = DataEmbedding(c_in, d_model, dropout=dropout)
        self.pos = PositionalEncoding(d_model, max_len=seq_len + 8)
        self.layers = nn.ModuleList(
            [
                InformerEncoderLayer(d_model, n_heads, factor, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(self.embed(x))
        for i, layer in enumerate(self.layers):
            stride = 1 if i == 0 else 2
            x = layer(x, pool_stride=stride)
        return x


class ViTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False
        )[0]
        x = x + self.ffn(self.norm2(x))
        return x


class OpticalFlowViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        assert image_size % patch_size == 0

        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        patch_dim = 2 * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.blocks = nn.ModuleList(
            [
                ViTBlock(d_model, n_heads, d_ff, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _compute_flow(self, image_seq: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = image_seq.shape
        image_np = image_seq.detach().cpu().float().numpy()
        flow_list = []

        for bi in range(b):
            cur = []
            for i in range(t - 1):
                img1 = image_np[bi, i].transpose(1, 2, 0)
                img2 = image_np[bi, i + 1].transpose(1, 2, 0)

                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

                img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(
                    "uint8"
                )
                img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(
                    "uint8"
                )

                flow = cv2.calcOpticalFlowFarneback(
                    img1,
                    img2,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                cur.append(torch.from_numpy(flow.transpose(2, 0, 1)).float())
            flow_list.append(torch.stack(cur, dim=0))

        return torch.stack(flow_list, dim=0).to(image_seq.device)

    def _patchify(self, flow_seq: torch.Tensor) -> torch.Tensor:
        """
        flow_seq: [B, L, 2, H, W]
        return:   [B, L, N, patch_dim]
        """
        b, l, c, h, w = flow_seq.shape
        p = self.patch_size
        x = flow_seq.reshape(b * l, c, h, w)
        patches = F.unfold(x, kernel_size=p, stride=p)
        patches = patches.transpose(1, 2)
        patches = patches.view(b, l, self.num_patches, -1)
        return patches

    def forward(self, image_seq: torch.Tensor) -> torch.Tensor:
        """
        image_seq: [B, L+1, 3, H, W]
        return: [B, L, N, D]
        """
        flow_seq = self._compute_flow(image_seq)
        patches = self._patchify(flow_seq)
        b, l, n, _ = patches.shape

        x = self.patch_embed(patches) + self.pos_embed[:, :n].unsqueeze(1)
        x = x.view(b * l, n, -1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x.view(b, l, n, -1)
        return x


class WeatherMLPEncoder(nn.Module):
    def __init__(
        self,
        weather_feat_dim: int,
        d_model: int,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(weather_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, weather_seq: torch.Tensor) -> torch.Tensor:
        return self.net(weather_seq)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, d_model)
        self.k_proj = nn.Linear(kv_dim, d_model)
        self.v_proj = nn.Linear(kv_dim, d_model)

        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_ = self.q_proj(q)
        k_ = self.k_proj(kv)
        v_ = self.v_proj(kv)
        out = self.attn(q_, k_, v_, need_weights=False)[0]
        out = self.norm1(q_ + out)
        out = self.norm2(out + self.ffn(out))
        return out


class SimpleDecoder(nn.Module):
    """
    Single-layer decoder to reflect decoder_layers = 1
    """

    def __init__(
        self,
        pred_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        num_queries: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, d_model))

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model * num_queries, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        b = memory.size(0)
        q = self.query_embed.expand(b, -1, -1)
        dec = self.decoder(tgt=q, memory=memory)
        dec = dec.reshape(b, -1)
        return self.out_proj(dec)


class WeatherImageSequenceBuilder:
    def __init__(self, weather_loader, image_seq_len: int):
        self.weather_loader = weather_loader
        self.image_seq_len = image_seq_len

    def build(
        self,
        year: torch.Tensor,
        month: torch.Tensor,
        day: torch.Tensor,
        hour: torch.Tensor,
    ) -> torch.Tensor:
        timestamps = [
            datetime(int(year[i]), int(month[i]), int(day[i]), int(hour[i]))
            for i in range(len(year))
        ]

        batch_seq: List[torch.Tensor] = []
        for ts in timestamps:
            frames = []
            for k in range(self.image_seq_len + 1):
                cur_ts = ts - timedelta(hours=self.image_seq_len - k)
                img = self.weather_loader.load_images_for_hour(cur_ts)
                frames.append(img)
            batch_seq.append(torch.stack(frames, dim=0))

        return torch.stack(batch_seq, dim=0).to(device)


class InViMT(nn.Module):

    def __init__(
        self,
        weather_loader,
        his_len: int,
        pred_len: int,
        image_seq_len: int,
        weather_feat_dim: int = 4,
        d_model: int = 512,
        d_ff: int = 2048,
        n_heads: int = 8,
        informer_layers: int = 2,
        vit_layers: int = 4,
        decoder_layers: int = 1,
        sparse_factor: float = 0.6,
        patch_size: int = 16,
        image_size: int = 224,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.his_len = his_len
        self.pred_len = pred_len
        self.image_seq_len = image_seq_len
        self.weather_feat_dim = weather_feat_dim
        self.d_model = d_model

        self.image_builder = WeatherImageSequenceBuilder(
            weather_loader=weather_loader,
            image_seq_len=image_seq_len,
        )

        self.load_encoder = InformerEncoder(
            c_in=1,
            seq_len=his_len,
            d_model=d_model,
            n_heads=n_heads,
            factor=sparse_factor,
            d_ff=d_ff,
            n_layers=informer_layers,
            dropout=dropout,
        )

        self.image_encoder = OpticalFlowViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            depth=vit_layers,
            dropout=dropout,
        )

        self.weather_encoder = WeatherMLPEncoder(
            weather_feat_dim=weather_feat_dim,
            d_model=d_model,
            hidden_dim=d_model,
            dropout=dropout,
        )

        self.ca_img_to_load = CrossAttentionBlock(
            d_model, d_model, d_model, n_heads, dropout
        )
        self.ca_wea_to_load = CrossAttentionBlock(
            d_model, d_model, d_model, n_heads, dropout
        )
        self.ca_load_to_img = CrossAttentionBlock(
            d_model, d_model, d_model, n_heads, dropout
        )
        self.ca_load_to_wea = CrossAttentionBlock(
            d_model, d_model, d_model, n_heads, dropout
        )

        self.memory_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if decoder_layers != 1:
            raise ValueError("This configured version fixes decoder_layers=1.")

        self.decoder = SimpleDecoder(
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_queries=1,
            dropout=dropout,
        )

    def _reshape_weather(self, weather: torch.Tensor) -> torch.Tensor:
        b = weather.size(0)
        return weather.view(b, self.pred_len, self.weather_feat_dim)

    def forward(
        self,
        weather: torch.Tensor,
        past: torch.Tensor,
        daily: Optional[torch.Tensor],
        weekly: Optional[torch.Tensor],
        time: Optional[torch.Tensor],
        year: torch.Tensor,
        month: torch.Tensor,
        day: torch.Tensor,
        hour: torch.Tensor,
        ind: Optional[torch.Tensor] = None,
        lam: Optional[torch.Tensor] = None,
        Modal: Optional[int] = None,
    ) -> torch.Tensor:
        weather_seq = self._reshape_weather(weather)
        wea_feat = self.weather_encoder(weather_seq)

        past_seq = past.unsqueeze(-1)
        load_feat = self.load_encoder(past_seq)

        image_seq = self.image_builder.build(year, month, day, hour)
        img_feat = self.image_encoder(image_seq)
        b, li, n, d = img_feat.shape
        img_feat = img_feat.view(b, li * n, d)

        load_from_img = self.ca_img_to_load(load_feat, img_feat)
        load_from_wea = self.ca_wea_to_load(load_feat, wea_feat)
        img_from_load = self.ca_load_to_img(img_feat, load_feat)
        wea_from_load = self.ca_load_to_wea(wea_feat, load_feat)

        memory = torch.cat(
            [load_from_img, load_from_wea, img_from_load, wea_from_load], dim=1
        )
        memory = self.memory_proj(memory)

        out = self.decoder(memory)
        return out