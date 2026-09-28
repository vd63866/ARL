"""End-to-end smoke test for the full RL pipeline via the CLI entry point.

Exercises the lifecycle that CI relies on, using the lightweight ``configs/ci_smoke.yaml``:

    Stage 1  CLI & Environment    config loads, ``make_env("drone")`` steps
    Stage 2  Training             ``adaptive-rl train`` exits 0
    Stage 3  Artifacts            final ``.zip``, checkpoints and metadata JSON on disk
    Stage 4  Reload & Evaluation  checkpoint reloads; ``adaptive-rl evaluate`` writes a report
    Stage 5  Schema Validation    ``evaluation.json`` has the required, finite fields
    Stage 6  Demo Execution       ``adaptive-rl demo-drone`` runs deterministically

Each stage is its own test and every failure message is prefixed with the stage name, so a
red CI run points straight at the regressed part of the pipeline. Later stages depend on the
module-scoped fixtures of earlier ones; if training fails, stages 3-6 error in setup with the
Stage 2 message.

Run locally with ``pytest -v tests/test_e2e_smoke.py``.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner, Result

from adaptive_rl.cli import app
from adaptive_rl.config import ExperimentConfig, load_config
from adaptive_rl.environments.registry import make_env

SMOKE_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "ci_smoke.yaml"
EVAL_EPISODES = 5
MIN_MODEL_BYTES = 10 * 1024
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

runner = CliRunner()


def _fail(stage: str, message: str, result: Result | None = None) -> None:
    detail = f"\n--- CLI output ---\n{_clean(result.output)}" if result is not None else ""
    if result is not None and result.exception is not None:
        detail += f"\n--- exception ---\n{result.exception!r}"
    pytest.fail(f"[{stage}] {message}{detail}", pytrace=False)


def _clean(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def _reject_non_finite(token: str) -> float:
    raise ValueError(f"non-finite JSON constant {token!r}")


@pytest.fixture(scope="module")
def smoke_config(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ExperimentConfig]:
    """Copy of ci_smoke.yaml whose outputs are redirected into a temporary directory."""
    out_dir = tmp_path_factory.mktemp("e2e_smoke")
    raw = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["output_dir"] = str(out_dir / "artifacts")
    raw["log_dir"] = str(out_dir / "artifacts" / "logs")
    config_path = out_dir / "ci_smoke.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config_path, load_config(config_path)


@pytest.fixture(scope="module")
def trained(smoke_config: tuple[Path, ExperimentConfig]) -> Result:
    """Stage 2: run ``adaptive-rl train`` once for the whole module."""
    config_path, _ = smoke_config
    result = runner.invoke(app, ["train", "--config", str(config_path)])
    if result.exit_code != 0:
        _fail("Stage 2: Training", f"`train` exited with code {result.exit_code}", result)
    return result


@pytest.fixture(scope="module")
def evaluation_report(
    smoke_config: tuple[Path, ExperimentConfig], trained: Result
) -> tuple[Path, Result]:
    """Stage 4: evaluate the saved final model with a fresh Evaluator via the CLI."""
    config_path, cfg = smoke_config
    model_path = cfg.output_dir / "models" / f"{cfg.name}_final.zip"
    report_path = cfg.output_dir / "evaluation.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(config_path),
            "--model",
            str(model_path),
            "--episodes",
            str(EVAL_EPISODES),
            "--output-report",
            str(report_path),
        ],
    )
    if result.exit_code != 0:
        _fail(
            "Stage 4: Reload & Evaluation",
            f"`evaluate` exited with code {result.exit_code}",
            result,
        )
    return report_path, result


def test_stage1_config_and_environment(smoke_config: tuple[Path, ExperimentConfig]) -> None:
    """Stage 1: the smoke config validates and the drone environment steps cleanly."""
    stage = "Stage 1: CLI & Environment"
    config_path, cfg = smoke_config

    result = runner.invoke(app, ["config", "validate", str(config_path)])
    if result.exit_code != 0:
        _fail(stage, "`config validate` rejected configs/ci_smoke.yaml", result)

    assert cfg.training is not None, f"[{stage}] smoke config has no 'training' section"
    budget = cfg.training.total_timesteps
    assert 500 <= budget <= 1000, f"[{stage}] smoke budget must be 500-1000 steps, got {budget}"
    assert 0 < cfg.training.checkpoint_freq < budget, (
        f"[{stage}] checkpoint_freq must yield an intermediate checkpoint before step {budget}"
    )

    env = make_env(cfg.environment.name, **cfg.environment.parameters)
    try:
        obs, _ = env.reset(seed=cfg.seed)
        assert env.observation_space.contains(obs), f"[{stage}] reset() observation out of space"
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)), f"[{stage}] step() produced non-finite observation"
        assert math.isfinite(float(reward)), f"[{stage}] step() produced non-finite reward"
        assert isinstance(terminated, bool) and isinstance(truncated, bool), (
            f"[{stage}] step() must return boolean terminated/truncated flags"
        )
    finally:
        env.close()


def test_stage2_training(trained: Result) -> None:
    """Stage 2: training exits cleanly and reports its summary.

    Artifact paths are checked on disk in Stage 3, not in the output: Rich wraps long paths
    inside the summary panel depending on terminal width.
    """
    if "Training Completed Successfully" not in _clean(trained.output):
        _fail("Stage 2: Training", "`train` did not print its completion summary", trained)


def test_stage3_artifacts(trained: Result, smoke_config: tuple[Path, ExperimentConfig]) -> None:
    """Stage 3: final weights, intermediate checkpoints and metadata are persisted."""
    stage = "Stage 3: Artifact Verification"
    _, cfg = smoke_config
    assert cfg.training is not None

    model_path = cfg.output_dir / "models" / f"{cfg.name}_final.zip"
    assert model_path.is_file(), f"[{stage}] final model missing at {model_path}"
    size = model_path.stat().st_size
    assert size > MIN_MODEL_BYTES, f"[{stage}] final model is only {size} bytes (<10 KB)"

    checkpoint_dir = cfg.output_dir / "checkpoints" / cfg.name
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.zip"))
    intermediate = [
        cp for cp in checkpoints if int(cp.stem.rsplit("_", 1)[-1]) < cfg.training.total_timesteps
    ]
    assert intermediate, f"[{stage}] no intermediate checkpoint in {checkpoint_dir}"
    for cp in checkpoints:
        assert cp.stat().st_size > MIN_MODEL_BYTES, f"[{stage}] checkpoint {cp.name} is <10 KB"

    metadata_path = cfg.output_dir / "metadata" / f"{cfg.name}_training.json"
    assert metadata_path.is_file(), f"[{stage}] training metadata missing at {metadata_path}"
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata.get("experiment_name") == cfg.name, f"[{stage}] metadata name mismatch"
    assert metadata.get("total_timesteps") == cfg.training.total_timesteps, (
        f"[{stage}] metadata total_timesteps mismatch"
    )
    created_at = metadata.get("created_at")
    assert isinstance(created_at, str), f"[{stage}] metadata has no 'created_at' timestamp"
    try:
        timestamp = datetime.fromisoformat(created_at)
    except ValueError:
        pytest.fail(f"[{stage}] metadata 'created_at' is not ISO-8601: {created_at!r}")
    assert timestamp.tzinfo is not None, f"[{stage}] metadata 'created_at' lacks a timezone"


def test_stage4_reload_and_evaluation(
    evaluation_report: tuple[Path, Result], smoke_config: tuple[Path, ExperimentConfig]
) -> None:
    """Stage 4: an intermediate checkpoint reloads and the evaluation report is written."""
    stage = "Stage 4: Reload & Evaluation"
    _, cfg = smoke_config
    report_path, _ = evaluation_report
    assert report_path.is_file(), f"[{stage}] `evaluate` did not write {report_path}"

    from adaptive_rl.algorithms.ppo import PPOAlgorithm

    checkpoint = min(
        (cfg.output_dir / "checkpoints" / cfg.name).glob("checkpoint_step_*.zip"),
        key=lambda cp: int(cp.stem.rsplit("_", 1)[-1]),
    )
    env = make_env(cfg.environment.name, **cfg.environment.parameters)
    try:
        algo = PPOAlgorithm.from_pretrained(checkpoint, env=env)
        obs, _ = env.reset(seed=cfg.seed)
        action, _ = algo.predict(obs, deterministic=True)
        assert env.action_space.contains(np.asarray(action, dtype=np.float32)), (
            f"[{stage}] reloaded checkpoint {checkpoint.name} predicted an invalid action"
        )
    finally:
        env.close()


def test_stage5_evaluation_schema(evaluation_report: tuple[Path, Result]) -> None:
    """Stage 5: evaluation.json is valid JSON with finite, bounded metrics."""
    stage = "Stage 5: Schema Validation"
    report_path, _ = evaluation_report
    try:
        report = json.loads(
            report_path.read_text(encoding="utf-8"), parse_constant=_reject_non_finite
        )
    except ValueError as err:
        pytest.fail(f"[{stage}] {report_path.name} is not valid finite JSON: {err}")

    assert isinstance(report, dict), f"[{stage}] report must be a JSON object"
    missing = {"episodes", "mean_reward", "success_rate", "collision_rate"} - report.keys()
    assert not missing, f"[{stage}] report is missing fields: {sorted(missing)}"

    assert report["episodes"] == EVAL_EPISODES, (
        f"[{stage}] expected episodes == {EVAL_EPISODES}, got {report['episodes']!r}"
    )
    mean_reward = report["mean_reward"]
    assert isinstance(mean_reward, (int, float)) and math.isfinite(mean_reward), (
        f"[{stage}] mean_reward must be a finite number, got {mean_reward!r}"
    )
    for key in ("success_rate", "collision_rate"):
        value = report[key]
        assert isinstance(value, (int, float)) and 0.0 <= value <= 1.0, (
            f"[{stage}] {key} must be a number in [0, 1], got {value!r}"
        )


def test_stage6_demo_is_deterministic(
    trained: Result, smoke_config: tuple[Path, ExperimentConfig]
) -> None:
    """Stage 6: demo-drone runs on the saved model and repeats exactly for a fixed seed."""
    stage = "Stage 6: Demo Execution"
    config_path, cfg = smoke_config
    model_path = cfg.output_dir / "models" / f"{cfg.name}_final.zip"
    args = ["demo-drone", "--model", str(model_path), "--config", str(config_path), "--seed", "42"]

    first = runner.invoke(app, args)
    if first.exit_code != 0:
        _fail(stage, f"`demo-drone` exited with code {first.exit_code}", first)
    if "OUTCOME:" not in _clean(first.output):
        _fail(stage, "`demo-drone` did not report a flight outcome", first)

    second = runner.invoke(app, args)
    if second.exit_code != 0:
        _fail(stage, f"second `demo-drone` run exited with code {second.exit_code}", second)
    assert _clean(first.output) == _clean(second.output), (
        f"[{stage}] `demo-drone` output differs between two runs with the same seed"
    )
