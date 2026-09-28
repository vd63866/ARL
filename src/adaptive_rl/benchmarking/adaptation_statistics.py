"""Paired recovery-time and six-cell multiplicity analysis for Issue #265."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Optional, Sequence

from adaptive_rl.protocol.constants import HORIZON, MIN_VALID_N, PLANNED_N, PRIMARY_CELLS
from adaptive_rl.protocol.statistics import (
    bootstrap_percentile_ci,
    cohen_dz,
    exact_sign_test,
    exact_wilcoxon_signed_rank,
    holm_adjust,
    impute_differences,
    paired_differences,
    paired_t_interval,
    paired_t_test,
)


@dataclass(frozen=True)
class PairedRecoveryAnalysis:
    planned_n: int
    completed_n: int
    failed_n: int
    valid_n: int
    differences: list[Optional[float]]
    mean_difference: Optional[float]
    standard_deviation: Optional[float]
    primary_p_value: Optional[float]
    holm_adjusted_p_value: Optional[float]
    interval_95: Optional[list[float]]
    cohen_dz: Optional[float]
    exact_sign_p_value: Optional[float]
    exact_wilcoxon_p_value: Optional[float]
    bootstrap_interval_95: Optional[list[float]]
    failure_imputation_bounds: Optional[dict[str, dict[str, Any]]]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_paired_recovery(
    fixed_t_h: Sequence[Optional[float]],
    adaptive_t_h: Sequence[Optional[float]],
    *,
    planned_n: int = PLANNED_N,
    horizon: float = HORIZON,
    min_valid_n: int = MIN_VALID_N,
) -> PairedRecoveryAnalysis:
    """Analyze one cell using the frozen paired tests and sensitivity methods."""
    if len(fixed_t_h) != len(adaptive_t_h):
        raise ValueError("Fixed and Adaptive T_H vectors must have equal length.")
    if len(fixed_t_h) != planned_n:
        raise ValueError(f"Expected {planned_n} planned pairs, got {len(fixed_t_h)}.")

    completed_n = sum(f is not None and a is not None for f, a in zip(fixed_t_h, adaptive_t_h))
    failed_n = planned_n - completed_n
    paired = paired_differences(fixed_t_h, adaptive_t_h)
    differences = [
        None if fixed is None or adaptive is None else float(adaptive) - float(fixed)
        for fixed, adaptive in zip(fixed_t_h, adaptive_t_h)
    ]
    worst = impute_differences(fixed_t_h, adaptive_t_h, horizon, "worst_for_adaptive")
    best = impute_differences(fixed_t_h, adaptive_t_h, horizon, "best_for_adaptive")
    bounds = {
        "worst_for_adaptive": {"mean_difference": sum(worst) / planned_n, "per_replicate": worst},
        "best_for_adaptive": {"mean_difference": sum(best) / planned_n, "per_replicate": best},
    }
    if len(paired) < min_valid_n:
        return PairedRecoveryAnalysis(
            planned_n=planned_n,
            completed_n=completed_n,
            failed_n=failed_n,
            valid_n=len(paired),
            differences=differences,
            mean_difference=None,
            standard_deviation=None,
            primary_p_value=None,
            holm_adjusted_p_value=None,
            interval_95=None,
            cohen_dz=None,
            exact_sign_p_value=None,
            exact_wilcoxon_p_value=None,
            bootstrap_interval_95=None,
            failure_imputation_bounds=bounds,
            status="inconclusive",
        )

    primary = paired_t_test(paired, min_valid_n=min_valid_n)
    return PairedRecoveryAnalysis(
        planned_n=planned_n,
        completed_n=completed_n,
        failed_n=failed_n,
        valid_n=len(paired),
        differences=differences,
        mean_difference=primary.mean,
        standard_deviation=primary.std_dev,
        primary_p_value=primary.p_value,
        holm_adjusted_p_value=None,
        interval_95=list(paired_t_interval(paired, min_valid_n=min_valid_n)),
        cohen_dz=cohen_dz(paired),
        exact_sign_p_value=exact_sign_test(paired, min_valid_n=min_valid_n).p_value,
        exact_wilcoxon_p_value=exact_wilcoxon_signed_rank(paired, min_valid_n=min_valid_n).p_value,
        bootstrap_interval_95=list(bootstrap_percentile_ci(paired, min_valid_n=min_valid_n)),
        failure_imputation_bounds=bounds,
        status="evaluable",
    )


def analyze_primary_cells(
    outcomes: dict[str, tuple[Sequence[Optional[float]], Sequence[Optional[float]]]],
) -> dict[str, PairedRecoveryAnalysis]:
    """Analyze the six frozen cells and apply Holm across the full cell family.

    Missing or invalid cells remain explicitly inconclusive. Their unavailable
    p-values are represented as 1.0 during family correction, and remain null
    in their own output record.
    """
    if set(outcomes) != set(PRIMARY_CELLS):
        raise ValueError("outcomes must contain exactly the preregistered PRIMARY_CELLS")
    results = {cell: analyze_paired_recovery(*outcomes[cell]) for cell in PRIMARY_CELLS}
    adjusted = holm_adjust(
        [
            results[cell].primary_p_value if results[cell].primary_p_value is not None else 1.0
            for cell in PRIMARY_CELLS
        ]
    )
    return {
        cell: replace(results[cell], holm_adjusted_p_value=adjusted[index])
        if results[cell].primary_p_value is not None
        else results[cell]
        for index, cell in enumerate(PRIMARY_CELLS)
    }


__all__ = ["PairedRecoveryAnalysis", "analyze_paired_recovery", "analyze_primary_cells"]
