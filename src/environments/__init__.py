"""Robotics environments and wrappers for imitation learning."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseRoboticsEnv(ABC):
    """Abstract base class for robotics environments."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the robotics environment.
        
        Args:
            config: Environment configuration
        """
        self.config = config
        self.observation_space: Optional[spaces.Space] = None
        self.action_space: Optional[spaces.Space] = None
        
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        pass
        
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        pass
        
    @abstractmethod
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        pass
        
    @abstractmethod
    def close(self) -> None:
        """Close the environment."""
        pass


class FetchReachWrapper(BaseRoboticsEnv):
    """Wrapper for FetchReach environment with safety features."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize FetchReach wrapper.
        
        Args:
            config: Environment configuration
        """
        super().__init__(config)
        
        # Create underlying environment
        self.env = gym.make("FetchReach-v1", render_mode="rgb_array")
        
        # Set observation and action spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Safety limits
        self.max_velocity = config.get("max_velocity", 1.0)
        self.max_force = config.get("max_force", 100.0)
        self.emergency_stop = False
        
        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = config.get("max_episode_steps", 50)
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        self.episode_steps = 0
        self.emergency_stop = False
        
        obs, info = self.env.reset(seed=seed)
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step with safety checks."""
        # Safety checks
        if self.emergency_stop:
            logger.warning("Emergency stop activated")
            return self._get_zero_obs(), 0.0, True, False, {"emergency_stop": True}
        
        # Velocity limiting
        action = self._limit_velocity(action)
        
        # Force limiting
        action = self._limit_force(action)
        
        # Take step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update episode tracking
        self.episode_steps += 1
        
        # Check episode termination
        if self.episode_steps >= self.max_episode_steps:
            truncated = True
            
        # Add safety info
        info.update({
            "episode_steps": self.episode_steps,
            "action_magnitude": np.linalg.norm(action),
            "velocity_magnitude": np.linalg.norm(action),  # Simplified
        })
        
        return obs, reward, terminated, truncated, info
        
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        return self.env.render()
        
    def close(self) -> None:
        """Close the environment."""
        self.env.close()
        
    def emergency_stop_func(self) -> None:
        """Emergency stop function."""
        self.emergency_stop = True
        logger.warning("Emergency stop activated")
        
    def _limit_velocity(self, action: np.ndarray) -> np.ndarray:
        """Limit action velocity."""
        action_norm = np.linalg.norm(action)
        if action_norm > self.max_velocity:
            action = action / action_norm * self.max_velocity
        return action
        
    def _limit_force(self, action: np.ndarray) -> np.ndarray:
        """Limit action force."""
        action_norm = np.linalg.norm(action)
        if action_norm > self.max_force:
            action = action / action_norm * self.max_force
        return action
        
    def _get_zero_obs(self) -> np.ndarray:
        """Get zero observation."""
        return np.zeros(self.observation_space.shape)


class TwoLinkArmEnv(BaseRoboticsEnv):
    """Simple 2-link arm environment for testing."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize 2-link arm environment."""
        super().__init__(config)
        
        # Arm parameters
        self.l1 = config.get("l1", 1.0)  # Link 1 length
        self.l2 = config.get("l2", 1.0)  # Link 2 length
        self.m1 = config.get("m1", 1.0)  # Link 1 mass
        self.m2 = config.get("m2", 1.0)  # Link 2 mass
        
        # Target position
        self.target = np.array(config.get("target", [1.5, 0.0]))
        
        # State: [q1, q2, q1_dot, q2_dot]
        self.state = np.zeros(4)
        
        # Action: [tau1, tau2] (joint torques)
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(2,), dtype=np.float32
        )
        
        # Observation: [q1, q2, q1_dot, q2_dot, target_x, target_y]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        
        # Safety limits
        self.max_velocity = config.get("max_velocity", 5.0)
        self.max_torque = config.get("max_torque", 10.0)
        
        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = config.get("max_episode_steps", 100)
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
            
        # Random initial joint angles
        self.state[:2] = np.random.uniform(-np.pi, np.pi, 2)
        self.state[2:] = np.zeros(2)  # Zero initial velocities
        
        self.episode_steps = 0
        
        obs = self._get_observation()
        info = {"episode_steps": self.episode_steps}
        
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Limit action
        action = np.clip(action, -self.max_torque, self.max_torque)
        
        # Simple dynamics (Euler integration)
        dt = 0.01
        
        # Joint accelerations (simplified dynamics)
        q1, q2, q1_dot, q2_dot = self.state
        
        # Inertia matrix (simplified)
        M11 = self.m1 * self.l1**2 + self.m2 * (self.l1**2 + self.l2**2 + 2*self.l1*self.l2*np.cos(q2))
        M12 = self.m2 * (self.l2**2 + self.l1*self.l2*np.cos(q2))
        M21 = M12
        M22 = self.m2 * self.l2**2
        
        M = np.array([[M11, M12], [M21, M22]])
        
        # Coriolis and centrifugal forces (simplified)
        c1 = -self.m2 * self.l1 * self.l2 * np.sin(q2) * (2*q1_dot*q2_dot + q2_dot**2)
        c2 = self.m2 * self.l1 * self.l2 * np.sin(q2) * q1_dot**2
        
        C = np.array([c1, c2])
        
        # Gravity (simplified)
        g = 9.81
        g1 = (self.m1 + self.m2) * self.l1 * g * np.cos(q1) + self.m2 * self.l2 * g * np.cos(q1 + q2)
        g2 = self.m2 * self.l2 * g * np.cos(q1 + q2)
        
        G = np.array([g1, g2])
        
        # Joint accelerations
        q_ddot = np.linalg.solve(M, action - C - G)
        
        # Update state
        self.state[2:] += q_ddot * dt
        self.state[:2] += self.state[2:] * dt
        
        # Limit velocities
        self.state[2:] = np.clip(self.state[2:], -self.max_velocity, self.max_velocity)
        
        # Compute reward
        end_effector_pos = self._get_end_effector_position()
        distance_to_target = np.linalg.norm(end_effector_pos - self.target)
        
        reward = -distance_to_target  # Negative distance as reward
        
        # Check termination
        terminated = distance_to_target < 0.1  # Success threshold
        truncated = self.episode_steps >= self.max_episode_steps
        
        self.episode_steps += 1
        
        obs = self._get_observation()
        info = {
            "end_effector_pos": end_effector_pos,
            "distance_to_target": distance_to_target,
            "episode_steps": self.episode_steps,
        }
        
        return obs, reward, terminated, truncated, info
        
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment (placeholder)."""
        if mode == "rgb_array":
            # Return a simple visualization
            return np.zeros((100, 100, 3), dtype=np.uint8)
        return None
        
    def close(self) -> None:
        """Close the environment."""
        pass
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.concatenate([self.state, self.target])
        
    def _get_end_effector_position(self) -> np.ndarray:
        """Get end effector position."""
        q1, q2 = self.state[:2]
        
        x = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
        y = self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)
        
        return np.array([x, y])


class DifferentialDriveEnv(BaseRoboticsEnv):
    """Differential drive robot environment."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize differential drive environment."""
        super().__init__(config)
        
        # Robot parameters
        self.wheel_base = config.get("wheel_base", 0.3)  # Distance between wheels
        self.wheel_radius = config.get("wheel_radius", 0.05)
        
        # State: [x, y, theta, v_left, v_right]
        self.state = np.zeros(5)
        
        # Action: [v_left, v_right] (wheel velocities)
        self.action_space = spaces.Box(
            low=-2.0, high=2.0, shape=(2,), dtype=np.float32
        )
        
        # Observation: [x, y, theta, v_left, v_right, target_x, target_y]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        # Target position
        self.target = np.array(config.get("target", [2.0, 0.0]))
        
        # Safety limits
        self.max_velocity = config.get("max_velocity", 2.0)
        
        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = config.get("max_episode_steps", 200)
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
            
        # Random initial position
        self.state[0] = np.random.uniform(-1.0, 1.0)  # x
        self.state[1] = np.random.uniform(-1.0, 1.0)  # y
        self.state[2] = np.random.uniform(-np.pi, np.pi)  # theta
        self.state[3:] = np.zeros(2)  # Zero initial wheel velocities
        
        self.episode_steps = 0
        
        obs = self._get_observation()
        info = {"episode_steps": self.episode_steps}
        
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Limit action
        action = np.clip(action, -self.max_velocity, self.max_velocity)
        
        # Update wheel velocities
        self.state[3:] = action
        
        # Compute robot velocity and angular velocity
        v_left, v_right = action
        v_linear = (v_left + v_right) * self.wheel_radius / 2
        v_angular = (v_right - v_left) * self.wheel_radius / self.wheel_base
        
        # Update position and orientation
        dt = 0.01
        x, y, theta = self.state[:3]
        
        x += v_linear * np.cos(theta) * dt
        y += v_linear * np.sin(theta) * dt
        theta += v_angular * dt
        
        # Normalize angle
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        
        self.state[:3] = [x, y, theta]
        
        # Compute reward
        distance_to_target = np.linalg.norm([x, y] - self.target)
        reward = -distance_to_target
        
        # Check termination
        terminated = distance_to_target < 0.2  # Success threshold
        truncated = self.episode_steps >= self.max_episode_steps
        
        self.episode_steps += 1
        
        obs = self._get_observation()
        info = {
            "robot_pos": np.array([x, y]),
            "distance_to_target": distance_to_target,
            "episode_steps": self.episode_steps,
        }
        
        return obs, reward, terminated, truncated, info
        
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment (placeholder)."""
        if mode == "rgb_array":
            return np.zeros((100, 100, 3), dtype=np.uint8)
        return None
        
    def close(self) -> None:
        """Close the environment."""
        pass
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.concatenate([self.state, self.target])


def create_robotics_env(env_name: str, config: Optional[Dict[str, Any]] = None) -> BaseRoboticsEnv:
    """Factory function to create robotics environments.
    
    Args:
        env_name: Name of the environment
        config: Environment configuration
        
    Returns:
        Robotics environment instance
    """
    if config is None:
        config = {}
    
    if env_name == "FetchReach-v1":
        return FetchReachWrapper(config)
    elif env_name == "TwoLinkArm-v0":
        return TwoLinkArmEnv(config)
    elif env_name == "DifferentialDrive-v0":
        return DifferentialDriveEnv(config)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def get_available_environments() -> List[str]:
    """Get list of available environments."""
    return ["FetchReach-v1", "TwoLinkArm-v0", "DifferentialDrive-v0"]
