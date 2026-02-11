"""Data collection and preprocessing for imitation learning."""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from abc import ABC, abstractmethod
import h5py
import pandas as pd

logger = logging.getLogger(__name__)


class BaseDemonstrationCollector(ABC):
    """Abstract base class for demonstration collectors."""
    
    def __init__(self, env, config: Dict[str, Any]) -> None:
        """Initialize demonstration collector.
        
        Args:
            env: Environment to collect demonstrations from
            config: Collection configuration
        """
        self.env = env
        self.config = config
        
    @abstractmethod
    def collect_demonstrations(self, num_episodes: int) -> List[Dict[str, np.ndarray]]:
        """Collect demonstration episodes."""
        pass
        
    @abstractmethod
    def save_demonstrations(self, demonstrations: List[Dict[str, np.ndarray]], path: Union[str, Path]) -> None:
        """Save demonstrations to file."""
        pass
        
    @abstractmethod
    def load_demonstrations(self, path: Union[str, Path]) -> List[Dict[str, np.ndarray]]:
        """Load demonstrations from file."""
        pass


class ExpertDemonstrationCollector(BaseDemonstrationCollector):
    """Collect demonstrations from an expert policy."""
    
    def __init__(self, env, config: Dict[str, Any]) -> None:
        """Initialize expert demonstration collector."""
        super().__init__(env, config)
        
        # Expert policy configuration
        self.expert_type = config.get("expert_type", "random")  # random, scripted, or loaded
        self.expert_model_path = config.get("expert_model_path", None)
        
        # Collection parameters
        self.max_episode_steps = config.get("max_episode_steps", 50)
        self.success_threshold = config.get("success_threshold", 0.8)
        
        # Load expert model if specified
        self.expert_model = None
        if self.expert_model_path and Path(self.expert_model_path).exists():
            self._load_expert_model()
            
    def collect_demonstrations(self, num_episodes: int) -> List[Dict[str, np.ndarray]]:
        """Collect demonstration episodes."""
        logger.info(f"Collecting {num_episodes} expert demonstrations")
        
        demonstrations = []
        successful_episodes = 0
        
        for episode in range(num_episodes):
            logger.info(f"Collecting episode {episode + 1}/{num_episodes}")
            
            # Reset environment
            state, info = self.env.reset()
            
            # Initialize episode data
            states = []
            actions = []
            rewards = []
            dones = []
            
            done = False
            step = 0
            
            while not done and step < self.max_episode_steps:
                # Get expert action
                action = self._get_expert_action(state)
                
                # Take step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                # Store data
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(terminated or truncated)
                
                # Update state
                state = next_state
                done = terminated or truncated
                step += 1
            
            # Check if episode was successful
            episode_reward = sum(rewards)
            is_successful = episode_reward >= self.success_threshold
            
            if is_successful:
                successful_episodes += 1
                
                # Store demonstration
                demonstration = {
                    "states": np.array(states),
                    "actions": np.array(actions),
                    "rewards": np.array(rewards),
                    "dones": np.array(dones),
                    "episode_reward": episode_reward,
                    "episode_length": len(states),
                    "success": True,
                }
                demonstrations.append(demonstration)
                
                logger.info(f"Episode {episode + 1}: Success (reward: {episode_reward:.2f})")
            else:
                logger.info(f"Episode {episode + 1}: Failed (reward: {episode_reward:.2f})")
        
        logger.info(f"Collected {len(demonstrations)} successful demonstrations out of {num_episodes} episodes")
        logger.info(f"Success rate: {len(demonstrations)/num_episodes:.2%}")
        
        return demonstrations
    
    def _get_expert_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from expert policy."""
        if self.expert_type == "random":
            return self.env.action_space.sample()
        elif self.expert_type == "scripted":
            return self._get_scripted_action(state)
        elif self.expert_type == "loaded" and self.expert_model is not None:
            return self._get_model_action(state)
        else:
            raise ValueError(f"Unknown expert type: {self.expert_type}")
    
    def _get_scripted_action(self, state: np.ndarray) -> np.ndarray:
        """Get scripted expert action (environment-specific)."""
        # This is a placeholder - implement environment-specific expert behavior
        # For example, for reaching tasks, move towards target
        if hasattr(self.env, 'target'):
            # Simple reaching behavior
            target = getattr(self.env, 'target', np.zeros(2))
            if len(state) >= 2:
                # Assume first two elements are position
                direction = target - state[:2]
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                action = direction * 0.5  # Scale action
                return np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        # Fallback to random action
        return self.env.action_space.sample()
    
    def _get_model_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from loaded expert model."""
        if self.expert_model is None:
            return self.env.action_space.sample()
        
        # This would depend on the model type
        # For now, return random action
        return self.env.action_space.sample()
    
    def _load_expert_model(self) -> None:
        """Load expert model from file."""
        # Placeholder for model loading
        logger.info(f"Loading expert model from {self.expert_model_path}")
        # Implementation would depend on model format
    
    def save_demonstrations(self, demonstrations: List[Dict[str, np.ndarray]], path: Union[str, Path]) -> None:
        """Save demonstrations to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.pkl':
            with open(path, 'wb') as f:
                pickle.dump(demonstrations, f)
        elif path.suffix == '.json':
            # Convert numpy arrays to lists for JSON serialization
            json_demos = []
            for demo in demonstrations:
                json_demo = {}
                for key, value in demo.items():
                    if isinstance(value, np.ndarray):
                        json_demo[key] = value.tolist()
                    else:
                        json_demo[key] = value
                json_demos.append(json_demo)
            
            with open(path, 'w') as f:
                json.dump(json_demos, f, indent=2)
        elif path.suffix == '.h5':
            with h5py.File(path, 'w') as f:
                for i, demo in enumerate(demonstrations):
                    group = f.create_group(f'episode_{i}')
                    for key, value in demo.items():
                        if isinstance(value, np.ndarray):
                            group.create_dataset(key, data=value)
                        else:
                            group.attrs[key] = value
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Saved {len(demonstrations)} demonstrations to {path}")
    
    def load_demonstrations(self, path: Union[str, Path]) -> List[Dict[str, np.ndarray]]:
        """Load demonstrations from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Demonstration file not found: {path}")
        
        if path.suffix == '.pkl':
            with open(path, 'rb') as f:
                demonstrations = pickle.load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                json_demos = json.load(f)
            
            # Convert lists back to numpy arrays
            demonstrations = []
            for json_demo in json_demos:
                demo = {}
                for key, value in json_demo.items():
                    if isinstance(value, list) and key in ['states', 'actions', 'rewards', 'dones']:
                        demo[key] = np.array(value)
                    else:
                        demo[key] = value
                demonstrations.append(demo)
        elif path.suffix == '.h5':
            demonstrations = []
            with h5py.File(path, 'r') as f:
                for episode_name in f.keys():
                    group = f[episode_name]
                    demo = {}
                    for key in group.keys():
                        demo[key] = group[key][:]
                    for key in group.attrs:
                        demo[key] = group.attrs[key]
                    demonstrations.append(demo)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Loaded {len(demonstrations)} demonstrations from {path}")
        return demonstrations


class DemonstrationPreprocessor:
    """Preprocess demonstration data for training."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.normalize_states = config.get("normalize_states", True)
        self.normalize_actions = config.get("normalize_actions", True)
        self.filter_noise = config.get("filter_noise", False)
        self.noise_threshold = config.get("noise_threshold", 0.1)
        
        # Normalization parameters (computed during preprocessing)
        self.state_mean: Optional[np.ndarray] = None
        self.state_std: Optional[np.ndarray] = None
        self.action_mean: Optional[np.ndarray] = None
        self.action_std: Optional[np.ndarray] = None
    
    def preprocess(self, demonstrations: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """Preprocess demonstration data.
        
        Args:
            demonstrations: Raw demonstration data
            
        Returns:
            Preprocessed demonstration data
        """
        logger.info("Preprocessing demonstration data")
        
        # Compute normalization parameters
        if self.normalize_states or self.normalize_actions:
            self._compute_normalization_params(demonstrations)
        
        # Preprocess each demonstration
        processed_demos = []
        for demo in demonstrations:
            processed_demo = self._preprocess_demo(demo)
            processed_demos.append(processed_demo)
        
        logger.info(f"Preprocessed {len(processed_demos)} demonstrations")
        return processed_demos
    
    def _compute_normalization_params(self, demonstrations: List[Dict[str, np.ndarray]]) -> None:
        """Compute normalization parameters from all demonstrations."""
        all_states = np.concatenate([demo["states"] for demo in demonstrations])
        all_actions = np.concatenate([demo["actions"] for demo in demonstrations])
        
        if self.normalize_states:
            self.state_mean = np.mean(all_states, axis=0)
            self.state_std = np.std(all_states, axis=0)
            self.state_std = np.where(self.state_std == 0, 1.0, self.state_std)
        
        if self.normalize_actions:
            self.action_mean = np.mean(all_actions, axis=0)
            self.action_std = np.std(all_actions, axis=0)
            self.action_std = np.where(self.action_std == 0, 1.0, self.action_std)
    
    def _preprocess_demo(self, demo: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Preprocess a single demonstration."""
        processed_demo = demo.copy()
        
        # Normalize states
        if self.normalize_states and self.state_mean is not None:
            processed_demo["states"] = (demo["states"] - self.state_mean) / self.state_std
        
        # Normalize actions
        if self.normalize_actions and self.action_mean is not None:
            processed_demo["actions"] = (demo["actions"] - self.action_mean) / self.action_std
        
        # Filter noise
        if self.filter_noise:
            processed_demo["actions"] = self._filter_action_noise(processed_demo["actions"])
        
        return processed_demo
    
    def _filter_action_noise(self, actions: np.ndarray) -> np.ndarray:
        """Filter noise from actions using simple smoothing."""
        if len(actions) < 3:
            return actions
        
        # Simple moving average filter
        filtered_actions = actions.copy()
        for i in range(1, len(actions) - 1):
            filtered_actions[i] = 0.5 * actions[i] + 0.25 * (actions[i-1] + actions[i+1])
        
        return filtered_actions
    
    def denormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """Denormalize action."""
        if self.normalize_actions and self.action_mean is not None:
            return normalized_action * self.action_std + self.action_mean
        return normalized_action
    
    def denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """Denormalize state."""
        if self.normalize_states and self.state_mean is not None:
            return normalized_state * self.state_std + self.state_mean
        return normalized_state


class DemonstrationValidator:
    """Validate demonstration data quality."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize validator."""
        self.config = config
        self.min_episode_length = config.get("min_episode_length", 5)
        self.max_episode_length = config.get("max_episode_length", 1000)
        self.min_success_rate = config.get("min_success_rate", 0.5)
        
    def validate(self, demonstrations: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Validate demonstration data.
        
        Args:
            demonstrations: Demonstration data to validate
            
        Returns:
            Validation results
        """
        logger.info("Validating demonstration data")
        
        validation_results = {
            "total_episodes": len(demonstrations),
            "valid_episodes": 0,
            "invalid_episodes": 0,
            "success_rate": 0.0,
            "avg_episode_length": 0.0,
            "issues": [],
        }
        
        if not demonstrations:
            validation_results["issues"].append("No demonstrations provided")
            return validation_results
        
        # Check each demonstration
        valid_demos = []
        episode_lengths = []
        successful_episodes = 0
        
        for i, demo in enumerate(demonstrations):
            issues = []
            
            # Check required keys
            required_keys = ["states", "actions"]
            for key in required_keys:
                if key not in demo:
                    issues.append(f"Missing required key: {key}")
            
            if issues:
                validation_results["invalid_episodes"] += 1
                validation_results["issues"].extend([f"Episode {i}: {issue}" for issue in issues])
                continue
            
            # Check data consistency
            states = demo["states"]
            actions = demo["actions"]
            
            if len(states) != len(actions):
                issues.append("States and actions length mismatch")
            
            if len(states) < self.min_episode_length:
                issues.append(f"Episode too short: {len(states)} < {self.min_episode_length}")
            
            if len(states) > self.max_episode_length:
                issues.append(f"Episode too long: {len(states)} > {self.max_episode_length}")
            
            # Check data types and shapes
            if not isinstance(states, np.ndarray):
                issues.append("States must be numpy array")
            
            if not isinstance(actions, np.ndarray):
                issues.append("Actions must be numpy array")
            
            if issues:
                validation_results["invalid_episodes"] += 1
                validation_results["issues"].extend([f"Episode {i}: {issue}" for issue in issues])
            else:
                valid_demos.append(demo)
                validation_results["valid_episodes"] += 1
                episode_lengths.append(len(states))
                
                # Check success
                if demo.get("success", False):
                    successful_episodes += 1
        
        # Compute summary statistics
        if episode_lengths:
            validation_results["avg_episode_length"] = np.mean(episode_lengths)
        
        if len(demonstrations) > 0:
            validation_results["success_rate"] = successful_episodes / len(demonstrations)
        
        # Check overall quality
        if validation_results["success_rate"] < self.min_success_rate:
            validation_results["issues"].append(
                f"Low success rate: {validation_results['success_rate']:.2%} < {self.min_success_rate:.2%}"
            )
        
        logger.info(f"Validation complete: {validation_results['valid_episodes']}/{validation_results['total_episodes']} episodes valid")
        
        return validation_results
