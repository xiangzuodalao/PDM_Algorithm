from __future__ import annotations

import math

import torch
import torch.nn as nn


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(
            kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2
        )

    def forward(self, x: torch.Tensor):
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelation(nn.Module):
    def __init__(self, factor: int = 1):
        super().__init__()
        self.factor = factor

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B, H, L, Dh = q.shape
        qf = torch.fft.rfft(q, dim=2)
        kf = torch.fft.rfft(k, dim=2)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=L, dim=2)
        top_k = max(1, int(self.factor * math.log(L)))
        weights, delays = torch.topk(corr.mean(dim=-1), top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)

        out = torch.zeros_like(v)
        for i in range(top_k):
            d = delays[..., i].unsqueeze(-1).unsqueeze(-1)
            v_shift = torch.gather(
                v, dim=2, index=(torch.arange(L, device=v.device).view(1, 1, -1, 1) - d) % L
            )
            out = out + weights[..., i].unsqueeze(-1).unsqueeze(-1) * v_shift
        return out


def _split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    B, L, D = x.shape
    d_head = D // n_heads
    return x.view(B, L, n_heads, d_head).permute(0, 2, 1, 3)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, H, L, Dh = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, factor: int = 1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ac = AutoCorrelation(factor=factor)

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor) -> torch.Tensor:
        q = _split_heads(self.q_proj(x_q), self.n_heads)
        k = _split_heads(self.k_proj(x_k), self.n_heads)
        v = _split_heads(self.v_proj(x_v), self.n_heads)
        out = self.ac(q, k, v)
        out = _merge_heads(out)
        return self.dropout(self.out_proj(out))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, moving_avg: int):
        super().__init__()
        self.attn = AutoCorrelationLayer(d_model, n_heads, dropout=dropout)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = x + self.dropout(self.attn(x, x, x))
        seasonal, trend = self.decomp1(x)
        y = self.conv1(seasonal.transpose(1, 2))
        y = self.activation(y)
        y = self.dropout(self.conv2(y).transpose(1, 2))
        seasonal2, trend2 = self.decomp2(seasonal + y)
        return seasonal2, trend + trend2


class Encoder(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, e_layers: int, moving_avg: int
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout, moving_avg) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        trend_sum = 0
        for layer in self.layers:
            x, trend = layer(x)
            trend_sum = trend_sum + trend
        return self.norm(x), trend_sum


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, moving_avg: int):
        super().__init__()
        self.self_attn = AutoCorrelationLayer(d_model, n_heads, dropout=dropout)
        self.cross_attn = AutoCorrelationLayer(d_model, n_heads, dropout=dropout)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, trend: torch.Tensor):
        x = x + self.dropout(self.self_attn(x, x, x))
        seasonal, trend1 = self.decomp1(x)
        x = seasonal + self.dropout(self.cross_attn(seasonal, enc_out, enc_out))
        seasonal2, trend2 = self.decomp2(x)
        y = self.conv1(seasonal2.transpose(1, 2))
        y = self.activation(y)
        y = self.dropout(self.conv2(y).transpose(1, 2))
        seasonal3, trend3 = self.decomp3(seasonal2 + y)
        trend = trend + trend1 + trend2 + trend3
        return seasonal3, trend


class Decoder(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, d_layers: int, moving_avg: int
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout, moving_avg) for _ in range(d_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, trend: torch.Tensor):
        for layer in self.layers:
            x, trend = layer(x, enc_out, trend)
        x = self.norm(x)
        seasonal_out = self.proj(x).squeeze(-1)
        trend_out = trend.squeeze(-1) if trend.ndim == 3 and trend.size(-1) == 1 else trend
        return seasonal_out, trend_out


class Autoformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        moving_avg: int = 25,
        e_layers: int = 2,
        d_layers: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.value_emb = nn.Linear(1, d_model)
        self.encoder = Encoder(d_model, n_heads, d_ff, dropout, e_layers, moving_avg)
        self.decoder = Decoder(d_model, n_heads, d_ff, dropout, d_layers, moving_avg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_in = self.value_emb(x)
        enc_out, trend = self.encoder(enc_in)
        dec_in = x[:, -self.label_len :, :]
        zeros = torch.zeros(x.size(0), self.pred_len, x.size(2), device=x.device, dtype=x.dtype)
        dec_in = torch.cat([dec_in, zeros], dim=1)
        dec_in = self.value_emb(dec_in)
        seasonal_out, trend_out = self.decoder(dec_in, enc_out, trend)
        return seasonal_out[:, -self.pred_len :] + trend_out[:, -self.pred_len :]
