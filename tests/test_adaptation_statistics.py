"""Tests for paired Issue #265 statistical reporting and Holm correction."""

from __future__ import annotations

import pytest

from adaptive_rl.benchmarking.adaptation_statistics import (
    analyze_paired_recovery,
    analyze_primary_cells,
)
from adaptive_rl.protocol.constants import PRIMARY_CELLS


def test_inconclusive_cell_suppresses_inference_below_minimum_valid_n() -> None:
    result = analyze_paired_recovery(
        fixed_t_h=[15.0] * 7 + [None, None, None],
        adaptive_t_h=[10.0] * 7 + [None, None, None],
    )
    assert result.valid_n == 7
    assert result.failed_n == 3
    assert result.status == "inconclusive"
    assert result.primary_p_value is None
    assert result.holm_adjusted_p_value is None
    assert result.failure_imputation_bounds is not None


def test_evaluable_cell_contains_paired_analysis_and_sensitivity_outputs() -> None:
    result = analyze_paired_recovery(
        fixed_t_h=[15.0] * 10,
        adaptive_t_h=[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 15.0, 15.0],
    )
    assert result.status == "evaluable"
    assert result.valid_n == 10
    assert result.mean_difference is not None and result.mean_difference < 0.0
    assert result.primary_p_value is not None
    assert result.interval_95 is not None
    assert result.exact_sign_p_value is not None
    assert result.exact_wilcoxon_p_value is not None
    assert result.bootstrap_interval_95 is not None


def test_holm_adjustment_covers_all_preregistered_cells_and_invalid_is_null() -> None:
    outcomes = {cell: ([15.0] * 10, [6.0] * 10) for cell in PRIMARY_CELLS}
    outcomes[PRIMARY_CELLS[-1]] = ([15.0] * 7 + [None] * 3, [6.0] * 7 + [None] * 3)
    results = analyze_primary_cells(outcomes)
    assert tuple(results) == PRIMARY_CELLS
    assert results[PRIMARY_CELLS[0]].holm_adjusted_p_value is not None
    assert results[PRIMARY_CELLS[-1]].status == "inconclusive"
    assert results[PRIMARY_CELLS[-1]].primary_p_value is None
    assert results[PRIMARY_CELLS[-1]].holm_adjusted_p_value is None


def test_cell_mapping_must_match_the_frozen_family() -> None:
    with pytest.raises(ValueError, match="PRIMARY_CELLS"):
        analyze_primary_cells({})
