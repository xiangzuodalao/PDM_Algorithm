from __future__ import annotations

from collections.abc import Callable
from typing import Any


Trainer = Callable[..., dict[str, Any]]


def get_trainer(model_type: str) -> Trainer:
    mt = str(model_type).strip().lower()
    if mt in {"informer", "testmodel"}:
        from valeo_pdm.transformer.train_informer import do_training as trainer

        return trainer
    if mt == "autoformer":
        from valeo_pdm.transformer.train_autoformer import do_training as trainer

        return trainer
    
    if mt == "xlstm":
        from valeo_pdm.transformer.train_xlstm import do_training as trainer

        return trainer
    
    if mt == "timellm":#新注册模型timellm
        from valeo_pdm.time_llm.train_timellm import do_training as trainer
        return trainer
    raise ValueError(f"不支持的模型类型: {model_type}")
