import math
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = state
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, spatial_size: Tuple[int, int], device_: torch.device):
        h, w = spatial_size
        h0 = torch.zeros(batch_size, self.hidden_dim, h, w, device=device_)
        c0 = torch.zeros(batch_size, self.hidden_dim, h, w, device=device_)
        return h0, c0


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        cells = []
        for i in range(num_layers):
            cur_in_dim = input_dim if i == 0 else hidden_dim
            cells.append(
                ConvLSTMCell(
                    input_dim=cur_in_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    bias=bias,
                )
            )
        self.cells = nn.ModuleList(cells)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.dim() != 5:
            raise ValueError(f"ConvLSTM expects 5D input [B,T,C,H,W], got {x.shape}")

        b, seq_len, _, h, w = x.shape
        cur_input = x

        for layer_idx in range(self.num_layers):
            h_cur, c_cur = self.cells[layer_idx].init_hidden(
                batch_size=b,
                spatial_size=(h, w),
                device_=x.device,
            )

            outputs = []
            for t in range(seq_len):
                h_cur, c_cur = self.cells[layer_idx](cur_input[:, t], (h_cur, c_cur))
                outputs.append(h_cur)

            cur_input = torch.stack(outputs, dim=1)

        return cur_input, (h_cur, c_cur)


class RICNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        roi_size: int = 16,
        pooled_size: int = 4,
        out_dim: int = 64,
        conv_channels: int = 64,
        roi_center: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.roi_size = roi_size
        self.pooled_size = pooled_size
        self.roi_center = roi_center

        self.avg_pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(conv_channels * pooled_size * pooled_size, out_dim)

    def _crop_roi(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        If roi_center is None, crop from center.
        """
        b, c, h, w = x.shape
        rs = min(self.roi_size, h, w)

        if self.roi_center is None:
            cy, cx = h // 2, w // 2
        else:
            cy, cx = self.roi_center

        y1 = max(0, cy - rs // 2)
        x1 = max(0, cx - rs // 2)
        y2 = min(h, y1 + rs)
        x2 = min(w, x1 + rs)

        if (y2 - y1) < rs:
            y1 = max(0, y2 - rs)
        if (x2 - x1) < rs:
            x1 = max(0, x2 - rs)

        roi = x[:, :, y1:y2, x1:x2]
        return roi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        roi = self._crop_roi(x)
        roi = self.avg_pool(roi)
        feat = self.conv(roi)
        feat = feat.flatten(start_dim=1)
        feat = self.fc(feat)
        return feat


class ImageEncoderConvLSTM_RICNN(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_channels: int = 3,
        convlstm_hidden: int = 32,
        convlstm_layers: int = 1,
        convlstm_kernel: int = 3,
        roi_size: int = 16,
        pooled_size: int = 4,
        ricnn_channels: int = 64,
        roi_center: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.convlstm = ConvLSTM(
            input_dim=in_channels,
            hidden_dim=convlstm_hidden,
            kernel_size=convlstm_kernel,
            num_layers=convlstm_layers,
        )
        self.ricnn = RICNN(
            in_channels=convlstm_hidden,
            roi_size=roi_size,
            pooled_size=pooled_size,
            out_dim=out_dim,
            conv_channels=ricnn_channels,
            roi_center=roi_center,
        )

    def forward(self, image_seq: torch.Tensor) -> torch.Tensor:
        """
        image_seq: [B, T, C, H, W]
        """
        _, (h_last, _) = self.convlstm(image_seq)
        z_img = self.ricnn(h_last)
        return z_img


class LoadGRUEncoder(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 1, out_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L] or [B, L, 1]
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, h = self.gru(x)
        feat = h[-1]
        feat = self.fc(feat)
        return feat


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, out_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabularFusion(nn.Module):
    def __init__(self, hist_dim: int, aux_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hist_dim + aux_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, z_hist: torch.Tensor, z_aux: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_hist, z_aux], dim=-1))


class PredictionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, pred_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class DCCALoss(nn.Module):
    def __init__(self, outdim_size: int, use_all_singular_values: bool = True, r1: float = 1e-3, r2: float = 1e-3, eps: float = 1e-9):
        super().__init__()
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.r1 = r1
        self.r2 = r2
        self.eps = eps

    def forward(self, H1: torch.Tensor, H2: torch.Tensor) -> torch.Tensor:
        if H1.dim() != 2 or H2.dim() != 2:
            raise ValueError(f"DCCALoss expects 2D inputs [B,D], got {H1.shape}, {H2.shape}")
        if H1.size(0) != H2.size(0):
            raise ValueError("Batch sizes of two views must match for DCCA.")

        m = H1.size(0)
        o1 = H1.size(1)
        o2 = H2.size(1)

        if m < 2:
            return H1.new_tensor(0.0)

        H1bar = H1 - H1.mean(dim=0, keepdim=True)
        H2bar = H2 - H2.mean(dim=0, keepdim=True)

        sigma12 = (H1bar.t() @ H2bar) / (m - 1)
        sigma11 = (H1bar.t() @ H1bar) / (m - 1) + self.r1 * torch.eye(o1, device=H1.device, dtype=H1.dtype)
        sigma22 = (H2bar.t() @ H2bar) / (m - 1) + self.r2 * torch.eye(o2, device=H2.device, dtype=H2.dtype)

        # eigen decomposition
        D1, V1 = torch.linalg.eigh(sigma11)
        D2, V2 = torch.linalg.eigh(sigma22)

        pos_idx1 = D1 > self.eps
        pos_idx2 = D2 > self.eps

        D1 = D1[pos_idx1]
        V1 = V1[:, pos_idx1]
        D2 = D2[pos_idx2]
        V2 = V2[:, pos_idx2]

        sigma11_inv_sqrt = V1 @ torch.diag(D1.pow(-0.5)) @ V1.t()
        sigma22_inv_sqrt = V2 @ torch.diag(D2.pow(-0.5)) @ V2.t()

        Tval = sigma11_inv_sqrt @ sigma12 @ sigma22_inv_sqrt

        if self.use_all_singular_values:
            corr = torch.linalg.svdvals(Tval).sum()
        else:
            TT = Tval.t() @ Tval
            eigvals = torch.linalg.eigvalsh(TT)
            eigvals = torch.clamp(eigvals, min=self.eps)
            topk = min(self.outdim_size, eigvals.numel())
            corr = torch.sqrt(eigvals.topk(topk).values).sum()

        return -corr


class DCCALateFusionModel(nn.Module):
    def __init__(
        self,
        weather_loader,
        dim: int = 64,
        image_seq_len: int = 6,
        past_len: int = 6,
        pred_len: int = 1,
        time_feat_dim: int = 24,
        weather_feat_dim: int = 4,
        daily_len: int = 6,
        weekly_len: int = 4,
        convlstm_hidden: int = 32,
        convlstm_layers: int = 1,
        convlstm_kernel: int = 3,
        roi_size: int = 16,
        pooled_size: int = 4,
        ricnn_channels: int = 64,
        roi_center: Optional[Tuple[int, int]] = None,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        aux_hidden: int = 64,
        fusion_hidden: int = 64,
        pred_hidden: int = 64,
        dcca_dim: int = 32,
        lambda_dcca: float = 1e-4,
        dcca_use_all_singular_values: bool = True,
        dropout: float = 0.0,
        dcca_r1: float = 1e-3,
        dcca_r2: float = 1e-3,
    ):
        super().__init__()
        self.weather_loader = weather_loader
        self.dim = dim
        self.image_seq_len = image_seq_len
        self.past_len = past_len
        self.pred_len = pred_len
        self.time_feat_dim = time_feat_dim
        self.weather_feat_dim = weather_feat_dim
        self.daily_len = daily_len
        self.weekly_len = weekly_len
        self.lambda_dcca = lambda_dcca

        aux_in_dim = daily_len + weekly_len + weather_feat_dim * pred_len + time_feat_dim * pred_len

        self.image_encoder = ImageEncoderConvLSTM_RICNN(
            out_dim=dim,
            in_channels=3,
            convlstm_hidden=convlstm_hidden,
            convlstm_layers=convlstm_layers,
            convlstm_kernel=convlstm_kernel,
            roi_size=roi_size,
            pooled_size=pooled_size,
            ricnn_channels=ricnn_channels,
            roi_center=roi_center,
        )

        self.hist_encoder = LoadGRUEncoder(
            input_dim=1,
            hidden_dim=gru_hidden,
            num_layers=gru_layers,
            out_dim=dim,
        )

        self.aux_encoder = MLPEncoder(
            input_dim=aux_in_dim,
            hidden_dim=aux_hidden,
            out_dim=dim,
            dropout=dropout,
        )

        self.tabular_fusion = TabularFusion(hist_dim=dim, aux_dim=dim, out_dim=dim)

        self.image_proj = nn.Sequential(
            nn.Linear(dim, dcca_dim),
            nn.ReLU(),
            nn.Linear(dcca_dim, dcca_dim),
        )
        self.tabular_proj = nn.Sequential(
            nn.Linear(dim, dcca_dim),
            nn.ReLU(),
            nn.Linear(dcca_dim, dcca_dim),
        )

        self.pred_head = PredictionHead(
            in_dim=dim * 2,
            hidden_dim=pred_hidden,
            pred_len=pred_len,
        )

        self.dcca_loss_fn = DCCALoss(
            outdim_size=dcca_dim,
            use_all_singular_values=dcca_use_all_singular_values,
            r1=dcca_r1,
            r2=dcca_r2,
        )

        self.last_extra: Dict[str, Any] = {}

    def _build_image_sequence(
        self,
        year: torch.Tensor,
        month: torch.Tensor,
        day: torch.Tensor,
        hour: torch.Tensor,
    ) -> torch.Tensor:
        timestamps = [
            datetime(
                int(year[i].item()),
                int(month[i].item()),
                int(day[i].item()),
                int(hour[i].item()),
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

        image_seq = torch.stack(seq, dim=0).to(device)
        return image_seq

    def _mix_image_if_needed(
        self,
        image_seq: torch.Tensor,
        ind: torch.Tensor,
        lam: torch.Tensor,
        Modal: Optional[int],
    ) -> torch.Tensor:
        if Modal == 1:
            image_seq_j = image_seq[ind]
            while lam.dim() < image_seq.dim():
                lam = lam.unsqueeze(-1)
            image_seq = lam * image_seq + (1.0 - lam) * image_seq_j
        return image_seq

    def forward(
        self,
        weather: torch.Tensor,
        past: torch.Tensor,
        daily: torch.Tensor,
        weekly: torch.Tensor,
        time: torch.Tensor,
        year: torch.Tensor,
        month: torch.Tensor,
        day: torch.Tensor,
        hour: torch.Tensor,
        ind: Optional[torch.Tensor] = None,
        lam: Optional[torch.Tensor] = None,
        Modal: Optional[int] = None,
        return_extra: bool = False,
    ):
        if ind is None:
            ind = torch.arange(weather.size(0), device=weather.device)
        if lam is None:
            lam = torch.ones(weather.size(0), 1, 1, 1, device=weather.device)

        image_seq = self._build_image_sequence(year, month, day, hour)
        image_seq = self._mix_image_if_needed(image_seq, ind, lam, Modal)

        z_img = self.image_encoder(image_seq)
        z_hist = self.hist_encoder(past)

        aux_input = torch.cat([daily, weekly, weather, time], dim=-1)
        z_aux = self.aux_encoder(aux_input)
        z_ts = self.tabular_fusion(z_hist, z_aux)

        z_img_dcca = self.image_proj(z_img)
        z_ts_dcca = self.tabular_proj(z_ts)

        dcca_loss = self.dcca_loss_fn(z_img_dcca, z_ts_dcca)

        z_fused = torch.cat([z_img, z_ts], dim=-1)
        pred = self.pred_head(z_fused)

        extra = {
            "z_img": z_img,
            "z_hist": z_hist,
            "z_aux": z_aux,
            "z_ts": z_ts,
            "z_img_dcca": z_img_dcca,
            "z_ts_dcca": z_ts_dcca,
            "dcca_loss": dcca_loss,
            "total_aux_loss": self.lambda_dcca * dcca_loss,
        }
        self.last_extra = extra

        if return_extra:
            return pred, extra
        return pred

    def get_last_dcca_loss(self) -> torch.Tensor:
        if "dcca_loss" not in self.last_extra:
            return torch.tensor(0.0, device=device)
        return self.last_extra["dcca_loss"]

    def get_last_total_aux_loss(self) -> torch.Tensor:
        if "total_aux_loss" not in self.last_extra:
            return torch.tensor(0.0, device=device)
        return self.last_extra["total_aux_loss"]