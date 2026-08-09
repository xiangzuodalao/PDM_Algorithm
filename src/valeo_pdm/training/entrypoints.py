from __future__ import annotations

from typing import Any


def train_informer(**kwargs: Any) -> dict[str, Any]:
    """延迟导入 Informer，保持 Trainer 注册目标可独立导入探测。"""

    from valeo_pdm.transformer.train_informer import do_training

    return do_training(**kwargs)


def train_autoformer(**kwargs: Any) -> dict[str, Any]:
    """延迟导入 Autoformer，保持 Trainer 注册目标可独立导入探测。"""

    from valeo_pdm.transformer.train_autoformer import do_training

    return do_training(**kwargs)
