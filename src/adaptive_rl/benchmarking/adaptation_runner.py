"""Protocol-ordered Issue #265 train/freeze/share/fork experiment runner."""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import gymnasium as gym
import numpy as np

from adaptive_rl.algorithms.adaptation import (
    PPOAdaptationAdapter,
    SACAdaptationAdapter,
    run_adaptation_update,
)
from adaptive_rl.benchmarking.adaptation_artifacts import write_adaptation_artifacts
from adaptive_rl.benchmarking.adaptation_runtime import EpisodeRecord, evaluate_episode
from adaptive_rl.benchmarking.adaptation_statistics import analyze_primary_cells
from adaptive_rl.config import ExperimentConfig, compute_config_sha256
from adaptive_rl.environments.registry import make_env
from adaptive_rl.protocol.adaptation import build_update_batch, validate_block_sequence
from adaptive_rl.protocol.constants import K_PRE, N_POST, PRIMARY_CELLS, TRAINING_SEEDS
from adaptive_rl.protocol.fork import fork_adaptive_and_fixed, model_fingerprint
from adaptive_rl.protocol.recovery import compute_recovery
from adaptive_rl.protocol.seeds import frozen_schedule, schedule_fingerprint
from adaptive_rl.protocol.statistics import decide_family
from adaptive_rl.training.trainer import get_trainer

logger = logging.getLogger(__name__)

EnvironmentFactory = Callable[..., gym.Env]
TrainerFactory = Callable[..., Any]


def _repository_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for distribution in ("adaptive-rl", "gymnasium", "stable-baselines3", "torch", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {"repository_commit": commit, "working_tree_dirty": dirty, "runtime_versions": versions}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _card_path() -> Path:
    return Path(__file__).resolve().parents[3] / "docs" / "research" / "TREATMENT_CARD.md"


def _new_env(
    config: ExperimentConfig,
    *,
    shifted: bool,
    environment_factory: EnvironmentFactory,
    max_steps: Optional[int] = None,
) -> gym.Env:
    parameters = dict(config.environment.parameters)
    parameters["max_steps"] = int(max_steps or config.environment.max_steps)
    if shifted:
        if config.adaptation_benchmark is None:
            raise ValueError("configuration does not declare the Issue #265 adaptation cell")
        parameters.update(config.adaptation_benchmark.shift_parameters)
    return environment_factory(config.environment.name, **parameters)


def _adapter_for(algorithm_name: str):
    if algorithm_name == "ppo":
        return PPOAdaptationAdapter()
    if algorithm_name == "sac":
        return SACAdaptationAdapter()
    raise ValueError(f"Issue #265 supports PPO and SAC, got {algorithm_name!r}")


def _train_once(
    config: ExperimentConfig,
    training_seed: int,
    training_dir: Path,
    *,
    trainer_factory: TrainerFactory,
    environment_factory: EnvironmentFactory,
    smoke: bool,
) -> tuple[Any, dict[str, Any]]:
    if config.training is None:
        raise ValueError("Issue #265 requires a training configuration")
    if training_dir.exists():
        raise FileExistsError(f"training output already exists: {training_dir}")
    training_dir.mkdir(parents=True)

    effective = config.model_copy(deep=True)
    effective.seed = int(training_seed)
    effective.name = f"{config.name}_seed_{training_seed}"
    effective.output_dir = training_dir
    effective.log_dir = training_dir / "logs"
    # The training component does not receive the TEST-B scenario payload.
    effective.adaptation_benchmark = None
    algorithm_config = effective.algorithm.model_copy(deep=True)
    algorithm_parameters = dict(algorithm_config.parameters)
    algorithm_parameters["seed"] = int(training_seed)
    if smoke:
        effective.training.total_timesteps = 32
        if algorithm_config.name.lower() == "ppo":
            algorithm_parameters.update({"n_steps": 16, "n_epochs": 1})
            algorithm_config.batch_size = min(8, algorithm_config.batch_size)
        else:
            algorithm_parameters.update(
                {"learning_starts": 1, "gradient_steps": 1, "buffer_size": 64}
            )
            algorithm_config.batch_size = min(8, algorithm_config.batch_size)
    algorithm_config.parameters = algorithm_parameters
    effective.algorithm = algorithm_config
    training_env = _new_env(
        effective,
        shifted=False,
        environment_factory=environment_factory,
        max_steps=8 if smoke else None,
    )
    trainer = None
    started = time.perf_counter()
    try:
        trainer = trainer_factory(config=effective, env=training_env)
        result = trainer.fit()
        algorithm = trainer.algorithm
    except BaseException:
        if trainer is not None:
            trainer.close()
        else:
            training_env.close()
        raise
    else:
        training_seconds = float(time.perf_counter() - started)
        trainer.close()
    return algorithm, {
        "training_seed": training_seed,
        "total_timesteps_requested": effective.training.total_timesteps,
        "total_timesteps_completed": int(getattr(algorithm, "num_timesteps", 0)),
        "training_time_seconds": training_seconds,
        "model_path": str(result.final_model_path),
        "training_metadata_path": str(result.metadata_path) if result.metadata_path else None,
        "episodes_completed": int(result.episodes_completed),
        "mean_reward": float(result.mean_reward),
        "smoke_override": bool(smoke),
        "effective_config": effective.model_dump(mode="json"),
    }


def _run_evaluation_segment(
    *,
    algorithm: Any,
    env: gym.Env,
    training_seed: int,
    algorithm_name: str,
    environment_name: str,
    phase: str,
    indices: Sequence[int],
    seeds: Sequence[int],
    deterministic: bool,
    arm: str,
    initial_update_log: Any = None,
) -> list[EpisodeRecord]:
    if len(indices) != len(seeds):
        raise ValueError("episode indices and seeds must have matching lengths")
    records: list[EpisodeRecord] = []
    update_log = initial_update_log
    for episode_index, episode_seed in zip(indices, seeds):
        records.append(
            evaluate_episode(
                algorithm=algorithm,
                env=env,
                training_seed=training_seed,
                phase=phase,
                episode_index=episode_index,
                episode_seed=episode_seed,
                algorithm_name=algorithm_name,
                environment_name=environment_name,
                deterministic=deterministic,
                arm=arm,
                update_log=update_log,
            )
        )
        update_log = None
    return records


def _recover(pre: Sequence[EpisodeRecord], post: Sequence[EpisodeRecord]) -> dict[str, Any]:
    result = compute_recovery(
        [episode.reward for episode in pre], [episode.reward for episode in post]
    )
    output = asdict(result)
    output["T_H"] = result.truncated_recovery_time
    return output


@dataclass
class ReplicateResult:
    training_seed: int
    status: str
    schedule_fingerprint: Optional[str] = None
    training_provenance: dict[str, Any] = field(default_factory=dict)
    frozen_fingerprint: Optional[str] = None
    fixed_final_fingerprint: Optional[str] = None
    pre_shift_performance: Optional[float] = None
    shock_performance: Optional[float] = None
    shared_pre_shift_episodes: list[EpisodeRecord] = field(default_factory=list)
    shared_shock_episodes: list[EpisodeRecord] = field(default_factory=list)
    adaptive_episodes: list[EpisodeRecord] = field(default_factory=list)
    fixed_episodes: list[EpisodeRecord] = field(default_factory=list)
    update_blocks: list[dict[str, Any]] = field(default_factory=list)
    adaptive_recovery: Optional[dict[str, Any]] = None
    fixed_recovery: Optional[dict[str, Any]] = None
    seeds: dict[str, list[int]] = field(default_factory=dict)
    effective_nominal_parameters: dict[str, Any] = field(default_factory=dict)
    effective_shift_parameters: dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_replicate(
    config: ExperimentConfig,
    training_seed: int,
    training_dir: Path,
    *,
    schedule: dict[int, dict[str, list[int]]],
    smoke: bool,
    trainer_factory: TrainerFactory,
    environment_factory: EnvironmentFactory,
) -> ReplicateResult:
    result = ReplicateResult(training_seed=training_seed, status="failed")
    try:
        result.schedule_fingerprint = schedule_fingerprint(schedule)
        algorithm, training_provenance = _train_once(
            config,
            training_seed,
            training_dir,
            trainer_factory=trainer_factory,
            environment_factory=environment_factory,
            smoke=smoke,
        )
        result.training_provenance = training_provenance
        frozen_fingerprint = model_fingerprint(algorithm)
        result.frozen_fingerprint = frozen_fingerprint
        algorithm.model.policy.set_training_mode(False)

        nominal_params = dict(config.environment.parameters)
        nominal_params["max_steps"] = 8 if smoke else config.environment.max_steps
        shifted_params = dict(nominal_params)
        shifted_params.update(config.adaptation_benchmark.shift_parameters)
        result.effective_nominal_parameters = dict(nominal_params)
        result.effective_shift_parameters = dict(shifted_params)
        result.seeds = {phase: list(values) for phase, values in schedule[training_seed].items()}

        nominal_env = _new_env(
            config,
            shifted=False,
            environment_factory=environment_factory,
            max_steps=8 if smoke else None,
        )
        try:
            result.shared_pre_shift_episodes = _run_evaluation_segment(
                algorithm=algorithm,
                env=nominal_env,
                training_seed=training_seed,
                algorithm_name=config.algorithm.name.lower(),
                environment_name=config.environment.name,
                phase="pre",
                indices=range(1, K_PRE + 1),
                seeds=schedule[training_seed]["pre"],
                deterministic=config.evaluation.deterministic,
                arm="shared",
            )
        finally:
            nominal_env.close()
        if model_fingerprint(algorithm) != frozen_fingerprint:
            raise RuntimeError("shared pre-shift evaluation mutated the frozen policy")
        result.pre_shift_performance = float(
            np.mean([episode.reward for episode in result.shared_pre_shift_episodes])
        )

        # Shift introduction occurs only after training and the one shared pre segment.
        shift_before = model_fingerprint(algorithm)
        shock_env = _new_env(
            config,
            shifted=True,
            environment_factory=environment_factory,
            max_steps=8 if smoke else None,
        )
        try:
            get_effective = getattr(shock_env, "get_effective_parameters", None)
            effective = dict(get_effective()) if callable(get_effective) else shifted_params
            for key, expected in config.adaptation_benchmark.shift_parameters.items():
                if effective.get(key) != expected:
                    raise RuntimeError(
                        f"TEST-B parameter {key!r} did not apply: expected {expected!r}, "
                        f"got {effective.get(key)!r}"
                    )
            result.effective_shift_parameters = effective
            result.shared_shock_episodes = _run_evaluation_segment(
                algorithm=algorithm,
                env=shock_env,
                training_seed=training_seed,
                algorithm_name=config.algorithm.name.lower(),
                environment_name=config.environment.name,
                phase="post",
                indices=range(1, 6),
                seeds=schedule[training_seed]["post"][:5],
                deterministic=config.evaluation.deterministic,
                arm="shared",
            )
        finally:
            shock_env.close()
        if model_fingerprint(algorithm) != shift_before:
            raise RuntimeError("shift introduction or shared shock evaluation mutated the policy")
        result.shock_performance = float(
            np.mean([episode.reward for episode in result.shared_shock_episodes])
        )

        adaptive, fixed, fork_fingerprint = fork_adaptive_and_fixed(algorithm)
        if fork_fingerprint != frozen_fingerprint:
            raise RuntimeError("Adaptive/Fixed forks did not originate at the frozen fingerprint")
        adapter = _adapter_for(config.algorithm.name.lower())
        post_history = [episode.post_shift_data() for episode in result.shared_shock_episodes]
        adaptive_env = _new_env(
            config,
            shifted=True,
            environment_factory=environment_factory,
            max_steps=8 if smoke else None,
        )
        try:
            for episode_index in range(6, N_POST + 1):
                boundary = episode_index - 1
                batch = build_update_batch(training_seed, post_history, block_episode=boundary)
                update_log = run_adaptation_update(adaptive, adapter, batch)
                result.update_blocks.append(update_log.to_dict())
                record = _run_evaluation_segment(
                    algorithm=adaptive,
                    env=adaptive_env,
                    training_seed=training_seed,
                    algorithm_name=config.algorithm.name.lower(),
                    environment_name=config.environment.name,
                    phase="post",
                    indices=[episode_index],
                    seeds=[schedule[training_seed]["post"][episode_index - 1]],
                    deterministic=config.evaluation.deterministic,
                    arm="adaptive",
                    initial_update_log=update_log,
                )[0]
                result.adaptive_episodes.append(record)
                post_history.append(record.post_shift_data())
        finally:
            adaptive_env.close()
        validate_block_sequence([block["block_episode"] for block in result.update_blocks])

        fixed_env = _new_env(
            config,
            shifted=True,
            environment_factory=environment_factory,
            max_steps=8 if smoke else None,
        )
        try:
            result.fixed_episodes = _run_evaluation_segment(
                algorithm=fixed,
                env=fixed_env,
                training_seed=training_seed,
                algorithm_name=config.algorithm.name.lower(),
                environment_name=config.environment.name,
                phase="post",
                indices=range(6, N_POST + 1),
                seeds=schedule[training_seed]["post"][5:],
                deterministic=config.evaluation.deterministic,
                arm="fixed",
            )
        finally:
            fixed_env.close()
        result.fixed_final_fingerprint = fixed.fingerprint
        if result.fixed_final_fingerprint != frozen_fingerprint:
            raise RuntimeError("Fixed policy fingerprint changed during its evaluation arm")

        adaptive_post = result.shared_shock_episodes + result.adaptive_episodes
        fixed_post = result.shared_shock_episodes + result.fixed_episodes
        result.adaptive_recovery = _recover(result.shared_pre_shift_episodes, adaptive_post)
        result.fixed_recovery = _recover(result.shared_pre_shift_episodes, fixed_post)
        result.status = "completed"
    except Exception as exc:
        result.failure_reason = f"{type(exc).__name__}: {exc}"
        logger.exception("Issue #265 replicate failed for training seed %s", training_seed)
    return result


def run_adaptation_benchmark(
    config: ExperimentConfig,
    *,
    output_dir: str | Path | None = None,
    training_seeds: Optional[Sequence[int]] = None,
    smoke: bool = False,
    trainer_factory: TrainerFactory = get_trainer,
    environment_factory: EnvironmentFactory = make_env,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the selected preregistered replicates and write JSON/CSV artifacts.

    A smoke run is visibly marked and uses one preregistered seed, a tiny
    training budget, and short episodes. It validates execution only; its
    outputs are not empirical research data.
    """
    if config.adaptation_benchmark is None:
        raise ValueError("configuration must include the Issue #265 adaptation_benchmark section")
    if config.algorithm.name.strip().lower() not in {"ppo", "sac"}:
        raise ValueError("Issue #265 supports only PPO and SAC")
    if config.training is None:
        raise ValueError("Issue #265 requires a training section")
    schedule = frozen_schedule()
    schedule_fp = schedule_fingerprint(schedule)
    if training_seeds is None:
        selected_seeds = [TRAINING_SEEDS[0]] if smoke else list(TRAINING_SEEDS)
    else:
        selected_seeds = [int(seed) for seed in training_seeds]
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("training_seeds must be non-empty and unique")
    if not set(selected_seeds).issubset(TRAINING_SEEDS):
        raise ValueError("all selected training seeds must come from TRAINING_SEEDS")
    if smoke and len(selected_seeds) != 1:
        raise ValueError("smoke mode runs exactly one preregistered training seed")

    card = _card_path()
    if not card.is_file():
        raise FileNotFoundError(f"Treatment Card is required before experiment execution: {card}")
    card_sha = _sha256_file(card)
    config_file = Path(config_path).resolve() if config_path is not None else None
    config_file_sha = _sha256_file(config_file) if config_file is not None else None

    target_dir = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("adaptation.json", "adaptation.csv"):
        if (target_dir / suffix).exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {target_dir / suffix}")
    training_root = target_dir / "training"
    for seed in selected_seeds:
        if (training_root / f"seed_{seed}").exists():
            raise FileExistsError(f"refusing to overwrite training output for seed {seed}")

    results = [
        _run_replicate(
            config,
            seed,
            training_root / f"seed_{seed}",
            schedule=schedule,
            smoke=smoke,
            trainer_factory=trainer_factory,
            environment_factory=environment_factory,
        )
        for seed in selected_seeds
    ]

    vectors: dict[str, tuple[list[Optional[float]], list[Optional[float]]]] = {}
    for cell in PRIMARY_CELLS:
        vectors[cell] = ([None] * len(TRAINING_SEEDS), [None] * len(TRAINING_SEEDS))
    cell_name = f"drone_disturbed/{config.algorithm.name.strip().lower()}"
    if cell_name not in vectors:
        raise ValueError(f"configured Issue #265 cell {cell_name!r} is not preregistered")
    fixed_vector, adaptive_vector = vectors[cell_name]
    for replicate in results:
        if replicate.status != "completed":
            continue
        position = TRAINING_SEEDS.index(replicate.training_seed)
        fixed_vector[position] = float(replicate.fixed_recovery["truncated_recovery_time"])
        adaptive_vector[position] = float(replicate.adaptive_recovery["truncated_recovery_time"])
    paired = analyze_primary_cells(vectors)
    family_decision = decide_family(
        {cell: analysis.primary_p_value for cell, analysis in paired.items()}
    )

    provenance = _repository_metadata()
    artifact = {
        "schema_version": "1.0",
        "protocol_version": config.adaptation_benchmark.protocol_version,
        "issue": "265",
        "run_type": "smoke" if smoke else "full_or_selected_research_run",
        "treatment_card_sha256": card_sha,
        "schedule_fingerprint": schedule_fp,
        "experiment": {
            "name": config.name,
            "algorithm": config.algorithm.name.strip().lower(),
            "environment": config.environment.name,
            "scenario": config.adaptation_benchmark.scenario,
            "planned_replicates": len(TRAINING_SEEDS),
            "selected_training_seeds": selected_seeds,
            "config_sha256": compute_config_sha256(config),
            "config": config.model_dump(mode="json"),
        },
        "replicates": [replicate.to_dict() for replicate in results],
        "paired_analysis": {cell: result.to_dict() for cell, result in paired.items()},
        "family_decision": family_decision,
        "failure_summary": {
            "failed_replicates": [
                {"training_seed": replicate.training_seed, "reason": replicate.failure_reason}
                for replicate in results
                if replicate.status != "completed"
            ],
            "completed_replicates": sum(rep.status == "completed" for rep in results),
            "valid_pairs": {cell: result.valid_n for cell, result in paired.items()},
        },
        "provenance": {
            **provenance,
            "treatment_card_path": str(card),
            "config_path": str(config_file) if config_file else None,
            "config_file_sha256": config_file_sha,
        },
        "scientific_claim": "Harness execution alone does not establish empirical superiority.",
    }
    json_path, csv_path = write_adaptation_artifacts(artifact, target_dir)
    artifact["artifact_paths"] = {"json": str(json_path), "csv": str(csv_path)}
    return artifact


__all__ = ["ReplicateResult", "run_adaptation_benchmark"]
