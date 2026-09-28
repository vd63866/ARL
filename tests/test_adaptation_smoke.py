"""CI-sized end-to-end Issue #265 protocol smoke test."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from adaptive_rl.cli import app
from adaptive_rl.protocol.constants import TRAINING_SEEDS
from adaptive_rl.protocol.seeds import frozen_schedule, schedule_fingerprint


def test_cli_adaptation_smoke_runs_complete_protocol_and_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "adaptation-smoke"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "adaptation",
            "--smoke",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    artifact = json.loads((output_dir / "adaptation.json").read_text(encoding="utf-8"))
    assert artifact["run_type"] == "smoke"
    replicate = artifact["replicates"][0]
    assert replicate["training_seed"] == TRAINING_SEEDS[0]
    assert replicate["status"] == "completed"
    train_config = replicate["training_provenance"]["effective_config"]
    assert train_config["adaptation_benchmark"] is None
    assert train_config["environment"]["parameters"]["num_obstacles"] == 8
    assert train_config["environment"]["parameters"]["wind_speed"] == 0.5
    assert train_config["environment"]["parameters"]["gust_sigma"] == 0.15
    assert replicate["schedule_fingerprint"] == schedule_fingerprint(frozen_schedule())
    assert len(replicate["shared_pre_shift_episodes"]) == 15
    assert len(replicate["shared_shock_episodes"]) == 5
    assert len(replicate["adaptive_episodes"]) == 10
    assert len(replicate["fixed_episodes"]) == 10
    assert [block["block_episode"] for block in replicate["update_blocks"]] == list(range(5, 15))
    assert [block["visible_episode_indices"] for block in replicate["update_blocks"]] == [
        list(range(1, boundary + 1)) for boundary in range(5, 15)
    ]
    assert [e["episode_seed"] for e in replicate["adaptive_episodes"]] == [
        e["episode_seed"] for e in replicate["fixed_episodes"]
    ]
    assert replicate["fixed_final_fingerprint"] == replicate["frozen_fingerprint"]
    assert artifact["paired_analysis"]["drone_disturbed/ppo"]["status"] == "inconclusive"
    assert (output_dir / "adaptation.csv").is_file()
