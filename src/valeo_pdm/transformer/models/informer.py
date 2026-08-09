import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer = x.size(1)
        return x + self.pe[:layer].unsqueeze(0)


class TokenEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.act(y)


class DataEmbedding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.token = TokenEmbedding(d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token(x)
        x = self.pos_enc(x)
        return self.dropout(x)


class TriangularCausalMask:
    def __init__(self, size: int, device: torch.device):
        self.mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def __call__(self) -> torch.Tensor:
        return self.mask.unsqueeze(0)


def _split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    B, L, D = x.shape
    d_head = D // n_heads
    return x.view(B, L, n_heads, d_head).permute(0, 2, 1, 3)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, H, L, Dh = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)


class ProbAttention(nn.Module):
    def __init__(self, d_head: int, n_heads: int, dropout: float = 0.1, factor: int = 5):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.factor = factor

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, Lq, Dh = q.shape
        Lk = k.size(2)
        scale = 1.0 / math.sqrt(self.d_head)

        scores_full = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if attn_mask is not None:
            scores_full = scores_full.masked_fill(attn_mask, float("-inf"))

        sparsity = scores_full.abs().max(dim=-1).values
        u = max(1, int(self.factor * math.log(max(Lk, 2))))
        top_idx = sparsity.topk(k=u, dim=-1).indices

        out = torch.zeros(B, H, Lq, Dh, device=q.device, dtype=q.dtype)
        for b in range(B):
            for h in range(H):
                sel = top_idx[b, h]
                scores_sel = scores_full[b, h, sel]
                attn_sel = torch.softmax(scores_sel, dim=-1)
                attn_sel = self.dropout(attn_sel)
                v_h = v[b, h]
                out[b, h, sel] = torch.matmul(attn_sel, v_h)

                context = v_h.mean(dim=0, keepdim=True)
                all_idx = torch.arange(Lq, device=q.device)
                mask_non = torch.ones(Lq, dtype=torch.bool, device=q.device)
                mask_non[sel] = False
                non_idx = all_idx[mask_non]
                if non_idx.numel() > 0:
                    out[b, h, non_idx] = context.expand(non_idx.numel(), Dh)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, attn_type: str = "prob"):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_type = attn_type

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.prob_attn = (
            ProbAttention(self.d_head, n_heads, dropout) if attn_type == "prob" else None
        )

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = _split_heads(self.q_proj(x_q), self.n_heads)
        k = _split_heads(self.k_proj(x_k), self.n_heads)
        v = _split_heads(self.v_proj(x_v), self.n_heads)

        if self.attn_type == "prob":
            out = self.prob_attn(q, k, v, attn_mask)
        else:
            scale = 1.0 / math.sqrt(self.d_head)
            scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.einsum("bhqk,bhkd->bhqd", attn, v)

        out = _merge_heads(out)
        out = self.out_proj(out)
        return self.dropout(out)


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, attn_type: str = "prob"
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, attn_type=attn_type)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(self.attn(x, x, x))
        x = self.norm1(x + y)
        y = self.conv1(x.transpose(1, 2))
        y = self.activation(y)
        y = self.dropout(self.conv2(y).transpose(1, 2))
        x = self.norm2(x + y)
        return x


class ConvLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.norm(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        e_layers: int,
        attn_type: str = "prob",
        distil: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, n_heads, d_ff, dropout, attn_type=attn_type)
                for _ in range(e_layers)
            ]
        )
        self.distil = distil
        self.conv_layers = (
            nn.ModuleList([ConvLayer(d_model) for _ in range(e_layers - 1)])
            if distil and e_layers > 1
            else None
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.conv_layers is not None and i < len(self.conv_layers):
                x = self.conv_layers[i](x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, attn_type="full")
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, attn_type="full")
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        device = x.device
        L = x.size(1)
        mask = TriangularCausalMask(L, device)()
        y = self.dropout(self.self_attn(x, x, x, attn_mask=mask))
        x = self.norm1(x + y)
        y = self.dropout(self.cross_attn(x, enc_out, enc_out))
        x = self.norm2(x + y)
        y = self.conv1(x.transpose(1, 2))
        y = self.activation(y)
        y = self.dropout(self.conv2(y).transpose(1, 2))
        x = self.norm3(x + y)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, d_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(d_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)


class Informer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        e_layers: int = 2,
        d_layers: int = 2,
        attn_type: str = "prob",
        distil: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.enc_embedding = DataEmbedding(d_model, dropout)
        self.dec_embedding = DataEmbedding(d_model, dropout)
        self.encoder = Encoder(
            d_model, n_heads, d_ff, dropout, e_layers, attn_type=attn_type, distil=distil
        )
        self.decoder = Decoder(d_model, n_heads, d_ff, dropout, d_layers)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_in = self.enc_embedding(x)
        enc_out = self.encoder(enc_in)

        dec_in = x[:, -self.label_len :, :]
        zeros = torch.zeros(x.size(0), self.pred_len, x.size(2), device=x.device, dtype=x.dtype)
        dec_in = torch.cat([dec_in, zeros], dim=1)
        dec_in = self.dec_embedding(dec_in)

        dec_out = self.decoder(dec_in, enc_out)
        out = self.proj(dec_out[:, -self.pred_len :, :]).squeeze(-1)
        return out
