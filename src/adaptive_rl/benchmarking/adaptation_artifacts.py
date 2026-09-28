"""JSON and analysis-friendly CSV serialization for Issue #265 runs."""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

CSV_FIELDS = (
    "replicate_index",
    "training_seed",
    "arm",
    "phase",
    "episode_index",
    "episode_seed",
    "algorithm",
    "environment",
    "reward",
    "length",
    "success",
    "collision",
    "terminated",
    "truncated",
    "policy_fingerprint_start",
    "policy_fingerprint_end",
    "update_block",
    "update_seed",
    "update_status",
    "parameter_delta_l2",
)


def _plain(value: Any) -> Any:
    """Convert supported scientific data values into strict JSON primitives."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, (str, int, float, bool)) for key in value):
            raise TypeError("artifact mappings must have primitive keys")
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(nested) for nested in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported artifact value type: {type(value).__name__}")


def _episode_rows(artifact: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for replicate_index, replicate in enumerate(artifact.get("replicates", []), start=1):
        common = {
            "replicate_index": replicate_index,
            "training_seed": replicate.get("training_seed"),
            "algorithm": artifact.get("experiment", {}).get("algorithm"),
            "environment": artifact.get("experiment", {}).get("environment"),
        }
        segments = (
            ("shared_pre_shift_episodes", "pre", "shared"),
            ("shared_shock_episodes", "post", "shared"),
            ("adaptive_episodes", "post", "adaptive"),
            ("fixed_episodes", "post", "fixed"),
        )
        for key, phase, arm in segments:
            for episode in replicate.get(key, []):
                row = {field: None for field in CSV_FIELDS}
                row.update(common)
                row.update(
                    arm=arm,
                    phase=phase,
                    episode_index=episode.get("episode_index"),
                    episode_seed=episode.get("episode_seed"),
                    reward=episode.get("reward"),
                    length=episode.get("length"),
                    success=episode.get("success"),
                    collision=episode.get("collision"),
                    terminated=episode.get("terminated"),
                    truncated=episode.get("truncated"),
                    policy_fingerprint_start=episode.get("policy_fingerprint_start"),
                    policy_fingerprint_end=episode.get("policy_fingerprint_end"),
                    update_block=episode.get("update_block"),
                    update_seed=episode.get("update_seed"),
                    update_status=episode.get("update_status"),
                    parameter_delta_l2=episode.get("parameter_delta_l2"),
                )
                yield row


def write_adaptation_artifacts(
    artifact: Mapping[str, Any],
    output_dir: str | Path,
    *,
    stem: str = "adaptation",
) -> tuple[Path, Path]:
    """Write strict JSON and flattened CSV without overwriting prior results.

    Both payloads are first written to temporary files, then installed with
    exclusive hard links. If either target already exists, no target is
    overwritten and the call raises ``FileExistsError``.
    """
    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a non-empty filename component")
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / f"{stem}.json"
    csv_path = target_dir / f"{stem}.csv"
    plain = _plain(artifact)
    if not isinstance(plain, dict):
        raise TypeError("artifact root must be a mapping")

    temp_paths: list[Path] = []
    try:
        for suffix, writer in (
            (".json", lambda handle: json.dump(plain, handle, indent=2, allow_nan=False)),
            (".csv", lambda handle: _write_csv(handle, plain)),
        ):
            fd, temp_name = tempfile.mkstemp(prefix=f".{stem}-", suffix=suffix, dir=target_dir)
            temp_path = Path(temp_name)
            temp_paths.append(temp_path)
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
        # link() is atomic and fails if the destination exists.
        os.link(temp_paths[0], json_path)
        try:
            os.link(temp_paths[1], csv_path)
        except BaseException:
            json_path.unlink()
            raise
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
    return json_path, csv_path


def _write_csv(handle: Any, artifact: Mapping[str, Any]) -> None:
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
    writer.writeheader()
    writer.writerows(_episode_rows(artifact))


__all__ = ["CSV_FIELDS", "write_adaptation_artifacts"]
