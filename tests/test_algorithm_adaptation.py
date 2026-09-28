"""Native PPO/SAC adaptation uses only recorded post-shift data."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from adaptive_rl.algorithms.adaptation import (
    PPOAdaptationAdapter,
    SACAdaptationAdapter,
    run_adaptation_update,
)
from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.algorithms.sac import SACAlgorithm
from adaptive_rl.benchmarking.adaptation_runtime import evaluate_episode
from adaptive_rl.protocol.adaptation import PostShiftEpisode, Transition, build_update_batch
from adaptive_rl.protocol.fork import model_fingerprint
from adaptive_rl.protocol.seeds import derive_seed


class _OneStepEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_seed = seed
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(np.asarray(action, dtype=np.float32))
        return (
            np.zeros(3, dtype=np.float32),
            1.0,
            False,
            True,
            {"success": True, "collision": False},
        )


def _batch(algorithm, training_seed: int = 31001):
    model = algorithm.model
    assert model is not None
    episodes = []
    for episode_index in range(1, 6):
        transitions = []
        for step in range(3):
            observation = np.full(model.observation_space.shape, step * 0.1, dtype=np.float32)
            next_observation = observation + 0.05
            if isinstance(algorithm, PPOAlgorithm):
                obs_tensor, _ = model.policy.obs_to_tensor(observation)
                with torch.no_grad():
                    action_tensor, value_tensor, log_prob_tensor = model.policy(obs_tensor)
                    next_obs_tensor, _ = model.policy.obs_to_tensor(next_observation)
                    next_value_tensor = model.policy.predict_values(next_obs_tensor)
                action = action_tensor.cpu().numpy().reshape(-1)
                behavior_value = float(value_tensor.reshape(-1)[0].cpu())
                behavior_log_prob = float(log_prob_tensor.reshape(-1)[0].cpu())
                behavior_next_value = float(next_value_tensor.reshape(-1)[0].cpu())
            else:
                action = np.zeros(model.action_space.shape, dtype=np.float32)
                behavior_value = None
                behavior_log_prob = None
                behavior_next_value = None
            transitions.append(
                Transition(
                    observation=observation,
                    action=action,
                    reward=float(episode_index + step),
                    next_observation=next_observation,
                    terminated=False,
                    truncated=step == 2,
                    behavior_log_prob=behavior_log_prob,
                    behavior_value=behavior_value,
                    behavior_next_value=behavior_next_value,
                )
            )
        episodes.append(
            PostShiftEpisode(
                index=episode_index,
                seed=derive_seed(training_seed, "post", episode_index),
                transitions=tuple(transitions),
            )
        )
    return build_update_batch(training_seed, episodes, block_episode=5)


def test_ppo_native_update_uses_recorded_rollout_without_environment_steps() -> None:
    env = gym.make("Pendulum-v1")
    try:
        algorithm = PPOAlgorithm(
            env=env, n_steps=8, batch_size=4, n_epochs=1, seed=31001, device="cpu"
        )
        batch = _batch(algorithm)
        model = algorithm.model
        assert model is not None
        before = model_fingerprint(algorithm)
        log = run_adaptation_update(algorithm, PPOAdaptationAdapter(), batch)
        assert log.block_episode == 5
        assert log.transition_count == 15
        assert log.visible_episode_indices == (1, 2, 3, 4, 5)
        assert log.update_seed == batch.seed
        assert log.parameter_delta_l2 > 0.0
        assert log.fingerprint_before == before
        assert log.fingerprint_after != before
        assert model.num_timesteps == 0
    finally:
        env.close()


def test_sac_native_update_uses_fresh_post_only_replay_buffer() -> None:
    env = gym.make("Pendulum-v1")
    try:
        algorithm = SACAlgorithm(
            env=env,
            batch_size=4,
            learning_starts=0,
            gradient_steps=1,
            buffer_size=100,
            seed=31001,
            device="cpu",
        )
        batch = _batch(algorithm)
        model = algorithm.model
        assert model is not None
        original_buffer = model.replay_buffer
        before = model_fingerprint(algorithm)
        log = run_adaptation_update(algorithm, SACAdaptationAdapter(), batch)
        assert log.block_episode == 5
        assert log.transition_count == 15
        assert log.visible_episode_indices == (1, 2, 3, 4, 5)
        assert log.update_seed == batch.seed
        assert log.parameter_delta_l2 > 0.0
        assert log.fingerprint_before == before
        assert log.fingerprint_after != before
        assert model.replay_buffer is original_buffer
        assert original_buffer is not None and original_buffer.size() == 0
        assert model.num_timesteps == 0
    finally:
        env.close()


def test_episode_collector_records_transition_and_keeps_policy_constant() -> None:
    model_env = gym.make("Pendulum-v1")
    try:
        algorithm = PPOAlgorithm(
            env=model_env, n_steps=8, batch_size=4, n_epochs=1, seed=31001, device="cpu"
        )
        env = _OneStepEnv()
        before = model_fingerprint(algorithm)
        record = evaluate_episode(
            algorithm=algorithm,
            env=env,
            training_seed=31001,
            phase="post",
            episode_index=1,
            episode_seed=derive_seed(31001, "post", 1),
            algorithm_name="ppo",
            environment_name="fake_drone",
            deterministic=True,
            arm="shared",
        )
        assert env.reset_seed == record.episode_seed
        assert record.length == 1
        assert record.success is True
        assert record.collision is False
        assert len(record.transitions) == 1
        assert record.transitions[0].behavior_log_prob is not None
        assert record.transitions[0].behavior_value is not None
        assert record.transitions[0].behavior_next_value is not None
        assert record.policy_fingerprint_start == record.policy_fingerprint_end == before
        assert model_fingerprint(algorithm) == before
    finally:
        model_env.close()
