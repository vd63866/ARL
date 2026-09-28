"""Leakage and ordering tests for the Issue #265 update boundary."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_rl.protocol.adaptation import (
    PostShiftEpisode,
    Transition,
    build_update_batch,
    call_update_atomically,
    validate_block_sequence,
)
from adaptive_rl.protocol.seeds import derive_seed


def _episode(training_seed: int, index: int) -> PostShiftEpisode:
    transition = Transition(
        observation=np.asarray([index], dtype=np.float32),
        action=np.asarray([index], dtype=np.float32),
        reward=float(index),
        next_observation=np.asarray([index + 0.5], dtype=np.float32),
        terminated=False,
        truncated=True,
    )
    return PostShiftEpisode(
        index=index,
        seed=derive_seed(training_seed, "post", index),
        transitions=(transition,),
    )


@pytest.mark.parametrize("boundary", range(5, 15))
def test_each_update_sees_exactly_its_completed_post_prefix(boundary: int) -> None:
    training_seed = 31001
    history = tuple(_episode(training_seed, index) for index in range(1, boundary + 1))
    batch = build_update_batch(training_seed, history, block_episode=boundary)

    assert batch.visible_episode_indices == tuple(range(1, boundary + 1))
    assert [int(t.observation[0]) for t in batch.transitions] == list(range(1, boundary + 1))
    assert batch.seed == derive_seed(training_seed, "update", boundary - 5)


def test_update_boundary_rejects_future_missing_or_misordered_data() -> None:
    history = tuple(_episode(31001, index) for index in range(1, 7))
    with pytest.raises(ValueError, match="exactly episodes 1..5"):
        build_update_batch(31001, history, block_episode=5)
    with pytest.raises(ValueError, match="ordered completed episodes"):
        build_update_batch(
            31001, (history[0], history[2], history[1], *history[3:]), block_episode=6
        )
    with pytest.raises(ValueError, match="only after"):
        build_update_batch(31001, history, block_episode=15)


def test_block_schedule_has_no_b15() -> None:
    validate_block_sequence(tuple(range(5, 15)))
    with pytest.raises(ValueError, match="exactly"):
        validate_block_sequence(tuple(range(5, 16)))


def test_transition_arrays_are_detached_and_read_only() -> None:
    observation = np.asarray([1.0], dtype=np.float32)
    transition = Transition(observation, observation, 1.0, observation, False, True)
    observation[0] = 99.0
    assert transition.observation[0] == 1.0
    with pytest.raises(ValueError):
        transition.observation[0] = 2.0


def test_failed_update_rolls_back_adapter_target_state() -> None:
    class Target:
        def __init__(self) -> None:
            self.weight = [1.0]

    class FailingAdapter:
        def update(self, algorithm, batch) -> None:
            del batch
            algorithm.weight[0] = 2.0
            raise RuntimeError("invalid update")

    target = Target()
    with pytest.raises(RuntimeError, match="invalid update"):
        call_update_atomically(target, FailingAdapter(), object())  # type: ignore[arg-type]
    assert target.weight == [1.0]
