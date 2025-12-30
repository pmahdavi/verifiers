"""Checkpoint utilities for evaluation resume functionality."""

import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

from pydantic import BaseModel

from verifiers.types import RolloutInput, SamplingArgs, State

logger = logging.getLogger(__name__)


class CheckpointMeta(BaseModel):
    """Pydantic model for checkpoint metadata."""

    checkpoint_version: str = "1.0"
    run_id: str
    env_id: str
    model: str
    rollouts_per_example: int
    sampling_args_hash: str
    completed_example_ids: list[int]
    total_examples: int
    completed_rollouts: int
    start_time: str
    last_checkpoint_time: str


def hash_sampling_args(sampling_args: SamplingArgs) -> str:
    """Create deterministic hash of sampling args for validation."""
    serialized = json.dumps(sampling_args, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def atomic_write_jsonl(filepath: Path, data: list[dict]) -> None:
    """Atomically write JSONL file using temp-file-then-rename."""
    temp_path = filepath.with_suffix(".jsonl.tmp")
    try:
        with open(temp_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(temp_path), str(filepath))  # Atomic on all platforms
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_json(filepath: Path, data: dict) -> None:
    """Atomically write JSON file using temp-file-then-rename."""
    temp_path = filepath.with_suffix(".json.tmp")
    try:
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(temp_path), str(filepath))  # Atomic on all platforms
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def load_checkpoint(checkpoint_dir: Path) -> tuple[CheckpointMeta | None, list[dict]]:
    """Load checkpoint metadata and completed rollouts.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Tuple of (metadata, rollout_data). If no checkpoint exists, returns (None, []).
    """
    meta_path = checkpoint_dir / "checkpoint_meta.json"
    data_path = checkpoint_dir / "checkpoint.jsonl"

    try:
        with open(meta_path) as f:
            meta_dict = json.load(f)
        meta = CheckpointMeta(**meta_dict)

        rollouts: list[dict] = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    rollouts.append(json.loads(line))

        # Validate consistency
        if len(rollouts) != meta.completed_rollouts:
            logger.warning(
                f"Checkpoint data mismatch: meta says {meta.completed_rollouts} rollouts, "
                f"but found {len(rollouts)}"
            )

        logger.info(
            f"Loaded checkpoint: {len(meta.completed_example_ids)} examples, "
            f"{len(rollouts)} rollouts"
        )
        return meta, rollouts
    except FileNotFoundError:
        return None, []
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_dir}: {e}")
        raise ValueError(f"Corrupt checkpoint at {checkpoint_dir}: {e}") from e


def validate_checkpoint_compatibility(
    meta: CheckpointMeta,
    env_id: str,
    model: str,
    rollouts_per_example: int,
    sampling_args: SamplingArgs,
) -> tuple[bool, str]:
    """Validate that checkpoint is compatible with current run config.

    Args:
        meta: Checkpoint metadata
        env_id: Current environment ID
        model: Current model name
        rollouts_per_example: Current rollouts per example
        sampling_args: Current sampling arguments

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    errors: list[str] = []

    if meta.env_id != env_id:
        errors.append(f"env_id mismatch: checkpoint={meta.env_id}, current={env_id}")

    if meta.model != model:
        errors.append(f"model mismatch: checkpoint={meta.model}, current={model}")

    if meta.rollouts_per_example != rollouts_per_example:
        errors.append(
            f"rollouts_per_example mismatch: checkpoint={meta.rollouts_per_example}, "
            f"current={rollouts_per_example}"
        )

    current_hash = hash_sampling_args(sampling_args)
    if meta.sampling_args_hash != current_hash:
        errors.append("sampling_args changed since checkpoint")

    if errors:
        return False, "; ".join(errors)
    return True, ""


def state_to_checkpoint_dict(state: State) -> dict[str, Any]:
    """Convert a State object to a checkpoint-serializable dict.

    Matches the format used in results.jsonl for consistency.
    """
    timing = state.get("timing", {})
    return {
        "example_id": state.get("example_id", 0),
        "prompt": state.get("prompt"),
        "completion": state.get("completion"),
        "answer": state.get("answer", ""),
        "task": state.get("task", "default"),
        "reward": state.get("reward", 0.0),
        "metrics": state.get("metrics", {}),
        "generation_ms": timing.get("generation_ms", 0),
        "scoring_ms": timing.get("scoring_ms", 0),
        "total_ms": timing.get("total_ms", 0),
    }


def checkpoint_dict_to_state(rd: dict[str, Any]) -> State:
    """Reconstruct a State object from checkpoint data.

    Args:
        rd: Rollout data dict from checkpoint.jsonl

    Returns:
        Reconstructed State object
    """
    state = State(
        input=RolloutInput(
            prompt=rd["prompt"],
            example_id=rd["example_id"],
            task=rd.get("task", "default"),
            answer=rd.get("answer", ""),
        )
    )
    state["completion"] = rd.get("completion")
    state["reward"] = rd.get("reward", 0.0)
    state["metrics"] = rd.get("metrics", {})
    state["timing"] = {
        "generation_ms": rd.get("generation_ms", 0),
        "scoring_ms": rd.get("scoring_ms", 0),
        "total_ms": rd.get("total_ms", 0),
    }
    state["is_completed"] = True
    return state


def save_checkpoint(
    checkpoint_dir: Path,
    all_states: list[State],
    meta: CheckpointMeta,
) -> None:
    """Save checkpoint atomically.

    Args:
        checkpoint_dir: Directory to save checkpoint files
        all_states: List of completed State objects
        meta: Checkpoint metadata (will be updated with current progress)
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Convert states to serializable format
    rollout_data = [state_to_checkpoint_dict(state) for state in all_states]

    # Update metadata with current progress
    completed_ids = list(set(state.get("example_id", 0) for state in all_states))
    meta_dict = meta.model_dump()
    meta_dict["completed_example_ids"] = completed_ids
    meta_dict["completed_rollouts"] = len(all_states)
    meta_dict["last_checkpoint_time"] = datetime.now().isoformat()

    # Atomic writes
    atomic_write_jsonl(checkpoint_dir / "checkpoint.jsonl", rollout_data)
    atomic_write_json(checkpoint_dir / "checkpoint_meta.json", meta_dict)

    logger.debug(
        f"Checkpoint saved: {len(all_states)} rollouts, "
        f"{len(completed_ids)} unique examples"
    )
