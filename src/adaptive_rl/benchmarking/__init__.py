"""Benchmarking and experimental evaluation modules for AdaptiveRL.

``run_learning_curve_benchmark`` is the PPO learning-curve benchmark (issue
#245); it delegates evaluation and cross-seed statistics to
``adaptive_rl.evaluation`` (issue #244). Optional RL libraries are imported
only when a benchmark actually executes.
"""

from importlib import import_module
from typing import Any

from adaptive_rl.benchmarking.adaptation_artifacts import write_adaptation_artifacts
from adaptive_rl.benchmarking.adaptation_statistics import (
    PairedRecoveryAnalysis,
    analyze_paired_recovery,
    analyze_primary_cells,
)
from adaptive_rl.benchmarking.learning_curve import (
    BenchmarkRunError,
    LearningCurveBenchmarkResult,
    LearningCurvePoint,
    plot_learning_curve,
    run_learning_curve_benchmark,
    validate_budgets,
)

_ABLATION_EXPORTS = {
    "REWARD_ABLATION_VARIANTS",
    "ConvergenceEvaluationCallback",
    "RewardAblationVariant",
    "export_ablation_csv",
    "export_ablation_json",
    "get_ablation_variant",
    "run_reward_ablation_experiment",
}


def __getattr__(name: str) -> Any:
    if name in _ABLATION_EXPORTS:
        return getattr(import_module(".ablation", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "REWARD_ABLATION_VARIANTS",
    "ConvergenceEvaluationCallback",
    "RewardAblationVariant",
    "export_ablation_csv",
    "export_ablation_json",
    "get_ablation_variant",
    "run_reward_ablation_experiment",
    "BenchmarkRunError",
    "LearningCurveBenchmarkResult",
    "LearningCurvePoint",
    "PairedRecoveryAnalysis",
    "analyze_paired_recovery",
    "analyze_primary_cells",
    "plot_learning_curve",
    "run_learning_curve_benchmark",
    "validate_budgets",
    "write_adaptation_artifacts",
]
