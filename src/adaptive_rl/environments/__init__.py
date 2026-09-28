"""Environment interfaces, metadata, and registration for AdaptiveRL."""

from __future__ import annotations

from adaptive_rl.environments.base import AdaptiveRLEnv
from adaptive_rl.environments.drone import (
    DroneKinematics3D,
    DroneNavigation3DEnv,
    DroneState3D,
    ObstacleSphere3D,
)
from adaptive_rl.environments.metadata import EnvironmentMetadata
from adaptive_rl.environments.registry import (
    EnvironmentRegistry,
    RegistryError,
    create_environment,
    get,
    get_metadata,
    list_all_metadata,
    list_environments,
    make_env,
    register,
    registry,
)


def register_default_environments() -> None:
    """Register built-in drone environments into the global registry."""
    if "drone" not in list_environments():
        register(
            "drone",
            lambda **kwargs: DroneNavigation3DEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="drone",
                description="Simulated kinematic 3D drone navigation with continuous translation and 3D LiDAR.",
                observation_type="box",
                action_type="continuous",
                version="0.1.0",
                max_episode_steps=200,
                reward_range=(-100.0, 100.0),
                tags=["continuous", "drone", "3d", "kinematics", "lidar", "navigation"],
            ),
        )

    if "drone_3d" not in list_environments():
        register(
            "drone_3d",
            lambda **kwargs: DroneNavigation3DEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="drone_3d",
                description="Simulated kinematic 3D drone navigation with continuous translation and 3D LiDAR.",
                observation_type="box",
                action_type="continuous",
                version="0.1.0",
                max_episode_steps=200,
                reward_range=(-100.0, 100.0),
                tags=["continuous", "drone", "3d", "kinematics", "lidar", "navigation"],
            ),
        )

    if "drone_disturbed" not in list_environments():
        register(
            "drone_disturbed",
            lambda **kwargs: DroneNavigation3DEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="drone_disturbed",
                description="Drone navigation with deterministic steady-wind and OU gust parameters.",
                observation_type="box",
                action_type="continuous",
                version="0.2.0",
                max_episode_steps=200,
                reward_range=(-100.0, 100.0),
                tags=["continuous", "drone", "distribution-shift", "wind", "gust"],
            ),
        )


# Automatically register default environments
register_default_environments()

__all__ = [
    "AdaptiveRLEnv",
    "DroneKinematics3D",
    "DroneNavigation3DEnv",
    "DroneState3D",
    "EnvironmentMetadata",
    "EnvironmentRegistry",
    "ObstacleSphere3D",
    "RegistryError",
    "create_environment",
    "get",
    "get_metadata",
    "list_all_metadata",
    "list_environments",
    "make_env",
    "register",
    "register_default_environments",
    "registry",
]
