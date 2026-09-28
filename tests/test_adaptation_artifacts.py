"""Structured artifact serialization and no-overwrite guarantees."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from adaptive_rl.benchmarking.adaptation_artifacts import write_adaptation_artifacts


def _artifact():
    return {
        "schema_version": "1.0",
        "protocol_version": "2.0",
        "issue": "265",
        "experiment": {"algorithm": "ppo", "environment": "drone"},
        "replicates": [
            {
                "training_seed": 31001,
                "shared_pre_shift_episodes": [
                    {
                        "episode_index": 1,
                        "episode_seed": 123,
                        "reward": np.float64(2.0),
                        "length": 15,
                        "success": True,
                        "collision": False,
                        "terminated": True,
                        "truncated": False,
                        "transitions": [{"observation": np.asarray([0.1, 0.2])}],
                    }
                ],
                "shared_shock_episodes": [],
                "adaptive_episodes": [],
                "fixed_episodes": [],
            }
        ],
    }


def test_json_and_flattened_csv_are_written_without_opaque_objects(tmp_path) -> None:
    json_path, csv_path = write_adaptation_artifacts(_artifact(), tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["replicates"][0]["shared_pre_shift_episodes"][0]["transitions"][0][
        "observation"
    ] == [0.1, 0.2]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["arm"] == "shared"
    assert rows[0]["phase"] == "pre"
    assert rows[0]["episode_seed"] == "123"


def test_existing_artifacts_are_never_overwritten(tmp_path) -> None:
    json_path, csv_path = write_adaptation_artifacts(_artifact(), tmp_path)
    before_json = json_path.read_bytes()
    before_csv = csv_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_adaptation_artifacts(_artifact(), tmp_path)
    assert json_path.read_bytes() == before_json
    assert csv_path.read_bytes() == before_csv


def test_nonfinite_values_are_rejected_for_strict_json(tmp_path) -> None:
    data = _artifact()
    data["value"] = float("nan")
    with pytest.raises(ValueError, match="JSON compliant"):
        write_adaptation_artifacts(data, tmp_path)
