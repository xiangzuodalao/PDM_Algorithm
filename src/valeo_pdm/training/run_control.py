from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from filelock import FileLock, Timeout

from valeo_pdm.paths import checkpoints_dir
from valeo_pdm.training.plan import TrainingPlan, TrainingPlanError
from valeo_pdm.transformer.artifacts import training_output_dir


MANIFEST_FILENAME = "training_manifest.json"
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "interrupted"})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _process_identity(pid: int) -> str | None:
    """返回可区分容器重启后 PID 复用的 boot-id 与进程启动时钟。"""

    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = stat.rsplit(")", 1)[1].split()
        start_time = fields[19]
    except (OSError, IndexError, UnicodeError):
        return None
    return f"{boot_id}:{start_time}"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_safe_run_dir(plan: TrainingPlan) -> Path:
    root = checkpoints_dir().resolve()
    base_candidate = training_output_dir(
        plan.equipment_code, plan.meas_code, plan.model_type
    )
    base = base_candidate.resolve()
    if not _is_relative_to(base, root):
        raise TrainingPlanError(
            "训练产物目录超出 checkpoints 根目录",
            status_code=400,
            code="UNSAFE_OUTPUT_PATH",
        )
    candidate = base / plan.model_info_id
    resolved = candidate.resolve(strict=False)
    if not _is_relative_to(resolved, root) or resolved.parent != base:
        raise TrainingPlanError(
            "训练产物目录无效", status_code=400, code="UNSAFE_OUTPUT_PATH"
        )
    return candidate


def ensure_run_target_available(plan: TrainingPlan) -> Path:
    run_dir = resolve_safe_run_dir(plan)
    if os.path.lexists(run_dir):
        raise TrainingPlanError(
            "ModelInfoID 对应的训练目录已存在，请使用新的 ModelInfoID",
            status_code=409,
            code="MODEL_INFO_ID_CONFLICT",
        )
    return run_dir


def _lock_path(plan: TrainingPlan) -> Path:
    root = checkpoints_dir().resolve()
    identity = "\0".join((plan.equipment_code, plan.meas_code, plan.model_info_id))
    return root / ".locks" / f"{sha256(identity.encode()).hexdigest()}.lock"


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    payload = dict(manifest)
    payload["updated_at"] = _utc_now()
    run_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=run_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, run_dir / MANIFEST_FILENAME)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def read_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / MANIFEST_FILENAME
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as exc:
        raise TrainingPlanError(
            "训练状态文件不可读", status_code=500, code="INVALID_MANIFEST"
        ) from exc
    return data if isinstance(data, dict) else None


def initial_manifest(plan: TrainingPlan) -> dict[str, Any]:
    now = _utc_now()
    return {
        "version": 1,
        "model_info_id": plan.model_info_id,
        "plan_hash": plan.plan_hash,
        "status": "reserved",
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "best_val": None,
        "test_loss": None,
        "error_code": None,
        "pid": os.getpid(),
        "process_identity": _process_identity(os.getpid()),
    }


def update_manifest(run_dir: Path, **changes: Any) -> dict[str, Any]:
    manifest = read_manifest(run_dir)
    if manifest is None:
        raise TrainingPlanError(
            "训练状态不存在", status_code=404, code="TRAINING_STATUS_NOT_FOUND"
        )
    manifest.update(changes)
    write_manifest(run_dir, manifest)
    return manifest


def mark_interrupted_if_stale(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """将宿主进程已经消失的非终态任务标记为 interrupted。"""

    if manifest.get("status") not in {"reserved", "running"}:
        return manifest
    pid = manifest.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return manifest
    expected_identity = manifest.get("process_identity")
    if isinstance(expected_identity, str):
        current_identity = _process_identity(pid)
        if current_identity is not None:
            if current_identity == expected_identity:
                return manifest
            return update_manifest(
                run_dir,
                status="interrupted",
                finished_at=_utc_now(),
                error_code="TRAINING_PROCESS_EXITED",
            )
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return update_manifest(
            run_dir,
            status="interrupted",
            finished_at=_utc_now(),
            error_code="TRAINING_PROCESS_EXITED",
        )
    except PermissionError:
        return manifest
    return manifest


def finite_metric(value: Any) -> float | None:
    try:
        metric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return metric if math.isfinite(metric) else None


@contextmanager
def training_run_lock(plan: TrainingPlan) -> Iterator[None]:
    """在训练及收尾期间持有同一 ModelInfoID 的跨进程文件锁。"""

    lock_path = _lock_path(plan)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_path)
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise TrainingPlanError(
            "同一 ModelInfoID 的训练任务正在执行",
            status_code=409,
            code="TRAINING_ALREADY_RUNNING",
        ) from exc
    try:
        yield
    finally:
        lock.release()


def create_training_run(plan: TrainingPlan) -> Path:
    """调用方须已持有 training_run_lock；本函数原子占用目标目录。"""

    run_dir = ensure_run_target_available(plan)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    identity_hash = sha256(plan.model_info_id.encode()).hexdigest()
    staging_dir = run_dir.parent / f".reserve-{identity_hash}-{uuid4().hex}"
    try:
        staging_dir.mkdir(exist_ok=False)
        write_manifest(staging_dir, initial_manifest(plan))
        if os.path.lexists(run_dir):
            raise FileExistsError(run_dir)
        staging_dir.rename(run_dir)
    except FileExistsError as exc:
        raise TrainingPlanError(
            "ModelInfoID 对应的训练目录已存在，请使用新的 ModelInfoID",
            status_code=409,
            code="MODEL_INFO_ID_CONFLICT",
        ) from exc
    finally:
        if staging_dir.exists():
            manifest_path = staging_dir / MANIFEST_FILENAME
            if manifest_path.exists():
                manifest_path.unlink()
            staging_dir.rmdir()
    return run_dir


@contextmanager
def reserve_training_run(plan: TrainingPlan) -> Iterator[Path]:
    """无需重新解析计划时使用的便捷入口。"""

    with training_run_lock(plan):
        yield create_training_run(plan)
