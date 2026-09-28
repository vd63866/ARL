"""Deterministic wind shift support in the existing drone environment."""

from __future__ import annotations

import numpy as np

from adaptive_rl.config import load_config
from adaptive_rl.environments.drone import DroneNavigation3DEnv


def test_issue_265_config_freezes_nominal_and_test_b_parameters() -> None:
    config = load_config("configs/drone_distribution_shift.yaml")
    assert config.adaptation_benchmark is not None
    assert config.adaptation_benchmark.scenario == "TEST-B"
    assert config.environment.parameters["num_obstacles"] == 8
    assert config.environment.parameters["wind_speed"] == 0.5
    assert config.adaptation_benchmark.shift_parameters == {
        "num_obstacles": 12,
        "wind_speed": 4.0,
        "gust_sigma": 0.6,
    }


def test_steady_wind_is_an_external_force_and_keeps_action_limit() -> None:
    env = DroneNavigation3DEnv(
        bounds=(30.0, 30.0, 15.0),
        num_obstacles=0,
        max_steps=2,
        wind_speed=4.0,
        gust_sigma=0.0,
        linear_damping=0.05,
    )
    try:
        env.reset(seed=17)
        env.step(np.zeros(3, dtype=np.float32))
        state = env.drone_state
        assert state.velocity[0] > 0.0
        assert np.isclose(np.linalg.norm(state.acceleration), 0.22, atol=1e-7)
        assert env.action_space.contains(np.zeros(3, dtype=np.float32))
    finally:
        env.close()


def test_ou_gust_stream_repeats_for_same_episode_seed() -> None:
    environments = [
        DroneNavigation3DEnv(
            bounds=(30.0, 30.0, 15.0),
            num_obstacles=0,
            max_steps=3,
            wind_speed=4.0,
            gust_sigma=0.6,
            gust_theta=0.15,
        )
        for _ in range(2)
    ]
    try:
        for env in environments:
            env.reset(seed=791)
        for _ in range(3):
            outputs = [env.step(np.zeros(3, dtype=np.float32)) for env in environments]
            np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
            np.testing.assert_array_equal(
                outputs[0][4]["wind_velocity"], outputs[1][4]["wind_velocity"]
            )
    finally:
        for env in environments:
            env.close()
