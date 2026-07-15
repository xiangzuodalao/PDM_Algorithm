from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from valeo_pdm.db.data_reader import load_postgres_timeseries
from valeo_pdm.transformer.data import load_series, make_windows, normalize_train_mean_std, split_train_val_test
from valeo_pdm.transformer.train_informer import normalize_columns, parse_time, resample_group


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleGRUForecast(nn.Module):
    """A lightweight GRU forecaster used to simulate an independent model pipeline."""

    def __init__(self, seq_len: int, pred_len: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True
        )
        self.proj = nn.Linear(hidden_size, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        y = self.proj(last)
        return y


def build_dataloaders(
    values: np.ndarray, seq_len: int, label_len: int, pred_len: int, batch_size: int, stride: int = 12
) -> Tuple[DataLoader, DataLoader, DataLoader, float, float]:
    X, Y = make_windows(values, seq_len, label_len, pred_len, step=stride)
    if len(X) == 0:
        raise ValueError("样本数量为0，请检查序列长度与窗口参数。")
    train_idx, val_idx, test_idx = split_train_val_test(len(X))
    X_norm, mean, std = normalize_train_mean_std(X[train_idx], X)
    Y_norm = (Y - mean) / std

    def to_loader(idxs):
        x_t = torch.from_numpy(X_norm[idxs])
        y_t = torch.from_numpy(Y_norm[idxs])
        ds = TensorDataset(x_t, y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    return to_loader(train_idx), to_loader(val_idx), to_loader(test_idx), mean, std


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
    show_progress: bool = False,
    epoch: int = 0,
    total_epochs: int = 0,
) -> float:
    model.train()
    crit = nn.MSELoss()
    total_loss = 0.0
    n = 0
    iterator = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False) if show_progress else loader
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad()
        y_hat = model(x)
        loss = crit(y_hat, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optim.step()
        total_loss += float(loss.item()) * len(x)
        n += len(x)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    crit = nn.MSELoss()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = crit(y_hat, y)
        total_loss += float(loss.item()) * len(x)
        n += len(x)
    return total_loss / max(n, 1)


def do_training(
    *,
    source: str,
    data: str,
    equipment_code: str,
    meas_code: str,
    days_back: int,
    freq: str,
    config: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    save: str,
    d_model: int,
    n_heads: int,  # unused but kept for interface compatibility
    d_ff: int,  # unused but kept for interface compatibility
    dropout: float,
    e_layers: int,
    d_layers: int,
    attn_type: str,  # unused but kept for interface compatibility
    distil_flag: str,  # unused but kept for interface compatibility
    early_stop: bool,
    patience: int,
    lr_sched: bool,
    lr_factor: float,
    lr_patience: int,
    weight_decay: float,
    grad_clip: float,
    stride: int = 12,
):
    os.makedirs(save, exist_ok=True)
    set_seed(42)

    if source == "csv":
        values = load_series(data)
    else:
        df = load_postgres_timeseries(config, equipment_code, meas_code, days_back=days_back)
        df = normalize_columns(df)
        df = parse_time(df)
        gdf = df[df["meas_code"] == meas_code] if "meas_code" in df.columns else df
        series_df = resample_group(gdf, freq=freq)
        values = pd.to_numeric(series_df["value"], errors="coerce").dropna().to_numpy(dtype=np.float32)

    train_loader, val_loader, test_loader, mean, std = build_dataloaders(
        values, seq_len, label_len, pred_len, batch_size, stride=stride
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleGRUForecast(seq_len=seq_len, pred_len=pred_len, hidden_size=d_model, num_layers=e_layers, dropout=dropout).to(
        device
    )

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=lr_factor, patience=lr_patience)
        if lr_sched
        else None
    )
    best_val = float("inf")
    no_improve = 0
    best_path = os.path.join(save, "testmodel_best.pt")
    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optim, device, grad_clip=grad_clip, show_progress=True, epoch=epoch, total_epochs=epochs
        )
        val_loss = evaluate(model, val_loader, device)
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        print(f"[testmodel] Epoch {epoch}: train MSE={train_loss:.6f}, val MSE={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "args": {
                        "seq_len": seq_len,
                        "pred_len": pred_len,
                        "d_model": d_model,
                        "num_layers": e_layers,
                        "dropout": dropout,
                    },
                },
                best_path,
            )
            print(f"[testmodel] Saved best checkpoint to {best_path}")
            no_improve = 0
        else:
            no_improve += 1
        if scheduler is not None:
            scheduler.step(val_loss)
        if early_stop and no_improve >= patience:
            print(f"[testmodel] Early stopping after {patience} epochs without improvement.")
            break

    test_loss = evaluate(model, test_loader, device)
    print(f"[testmodel] Test MSE={test_loss:.6f}")

    plots: dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt

        # loss curve
        loss_path = os.path.join(save, "testmodel_loss_curve.png")
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="train_mse")
        plt.plot(val_losses, label="val_mse")
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.title("TestModel Training / Validation Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_path, dpi=150)
        plt.close()
        plots["loss_curve"] = loss_path

        # forecast plot (first sample)
        forecast_path = os.path.join(save, "testmodel_forecast.png")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        x0, y0 = next(iter(test_loader))
        x0 = x0.to(device)
        y0 = y0.to(device)
        with torch.no_grad():
            y_hat = model(x0)
        mean = float(ckpt["mean"])
        std = float(ckpt["std"])
        y_true = (y0[0].detach().cpu().numpy() * std + mean).astype(np.float64)
        y_pred = (y_hat[0].detach().cpu().numpy() * std + mean).astype(np.float64)
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="true")
        plt.plot(y_pred, label="pred")
        plt.xlabel("step")
        plt.ylabel("value")
        plt.title("TestModel Forecast (test sample)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(forecast_path, dpi=150)
        plt.close()
        plots["forecast"] = forecast_path
    except ImportError:
        print("matplotlib 未安装，跳过训练曲线/预测曲线图片生成；可安装 valeo-pdm[viz] 启用。")
    except Exception as e:
        print(f"[testmodel] 生成图片失败（已跳过，不影响训练结果）：{e}")

    return {"best_path": best_path, "best_val": best_val, "test_loss": test_loss, "plots": plots}
