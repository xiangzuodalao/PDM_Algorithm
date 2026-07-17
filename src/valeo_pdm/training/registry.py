from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any


Trainer = Callable[..., dict[str, Any]]
TRAINER_IMPORTS: dict[str, tuple[str, str]] = {
    "informer": ("valeo_pdm.training.entrypoints", "train_informer"),
    "testmodel": ("valeo_pdm.transformer.train_informer", "do_training"),
    "autoformer": ("valeo_pdm.training.entrypoints", "train_autoformer"),
    "xlstm": ("valeo_pdm.transformer.train_xlstm", "do_training"),
    "timellm": ("valeo_pdm.time_llm.train_timellm", "do_training"),
}


def get_trainer(model_type: str) -> Trainer:
    mt = str(model_type).strip().lower()
    target = TRAINER_IMPORTS.get(mt)
    if target is None:
        raise ValueError(f"不支持的模型类型: {model_type}")
    module_name, attribute = target
    module = import_module(module_name)
    return getattr(module, attribute)
