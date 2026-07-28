"""MCP 工具入参模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DataSource = Literal["db", "postgres", "sqlserver", "csv"]
ExecutionMode = Literal["local_only", "platform"]


class TrainParams(BaseModel):
    """允许覆盖的训练参数；未提供的值由服务端有效配置补齐。"""

    model_config = ConfigDict(extra="forbid")

    seq_len: int | None = Field(default=None, ge=1, le=10000)
    label_len: int | None = Field(default=None, ge=1, le=10000)
    pred_len: int | None = Field(default=None, ge=1, le=10000)
    stride: int | None = Field(default=None, ge=1, le=10000)
    batch_size: int | None = Field(default=None, ge=1, le=256)
    epochs: int | None = Field(default=None, ge=1, le=200)
    lr: float | None = Field(default=None, gt=0, le=1)
    d_model: int | None = Field(default=None, ge=8, le=1024)
    n_heads: int | None = Field(default=None, ge=1, le=32)
    d_ff: int | None = Field(default=None, ge=8, le=4096)
    dropout: float | None = Field(default=None, ge=0, lt=1)
    e_layers: int | None = Field(default=None, ge=1, le=12)
    d_layers: int | None = Field(default=None, ge=1, le=12)
    attn_type: Literal["prob", "full"] | None = None
    distil: bool | None = None
    patience: int | None = Field(default=None, ge=1, le=200)
    lr_factor: float | None = Field(default=None, gt=0, lt=1)
    lr_patience: int | None = Field(default=None, ge=1, le=200)
    weight_decay: float | None = Field(default=None, ge=0, le=1)
    grad_clip: float | None = Field(default=None, gt=0, le=1000)
    moving_avg: int | None = Field(default=None, ge=1, le=10000)

    @model_validator(mode="after")
    def validate_combinations(self) -> TrainParams:
        if (
            self.label_len is not None
            and self.seq_len is not None
            and self.label_len > self.seq_len
        ):
            raise ValueError("label_len 不能大于 seq_len")
        if (
            self.d_model is not None
            and self.n_heads is not None
            and self.d_model % self.n_heads != 0
        ):
            raise ValueError("d_model 必须能被 n_heads 整除")
        if self.moving_avg is not None and self.moving_avg % 2 == 0:
            raise ValueError("moving_avg 必须为奇数")
        return self
