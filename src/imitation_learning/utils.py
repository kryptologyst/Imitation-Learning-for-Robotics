"""Utility functions for imitation learning."""

import random
import numpy as np
import torch
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_device(device: Optional[str] = None) -> str:
    """Get the best available device.
    
    Args:
        device: Preferred device ('cuda', 'mps', 'cpu')
        
    Returns:
        Available device string
    """
    if device is not None:
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif device == "cpu":
            return "cpu"
    
    # Auto-detect best device
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        """Check if training should stop early.
        
        Args:
            val_score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


def normalize_data(data: np.ndarray, mean: Optional[np.ndarray] = None, 
                  std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize data using z-score normalization.
    
    Args:
        data: Input data to normalize
        mean: Pre-computed mean (if None, computed from data)
        std: Pre-computed std (if None, computed from data)
        
    Returns:
        Normalized data, mean, std
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero
    
    normalized_data = (data - mean) / std
    return normalized_data, mean, std


def denormalize_data(normalized_data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Denormalize data.
    
    Args:
        normalized_data: Normalized data
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized data
    """
    return normalized_data * std + mean


def compute_trajectory_error(
    predicted_trajectory: np.ndarray, 
    expert_trajectory: np.ndarray
) -> Dict[str, float]:
    """Compute trajectory error metrics.
    
    Args:
        predicted_trajectory: Predicted trajectory
        expert_trajectory: Expert trajectory
        
    Returns:
        Dictionary of error metrics
    """
    # Ensure same length
    min_len = min(len(predicted_trajectory), len(expert_trajectory))
    pred = predicted_trajectory[:min_len]
    expert = expert_trajectory[:min_len]
    
    # L2 error
    l2_error = np.mean(np.linalg.norm(pred - expert, axis=1))
    
    # Final position error
    final_error = np.linalg.norm(pred[-1] - expert[-1])
    
    # Maximum error
    max_error = np.max(np.linalg.norm(pred - expert, axis=1))
    
    return {
        "l2_error": l2_error,
        "final_error": final_error,
        "max_error": max_error,
    }


def compute_success_rate(
    rewards: np.ndarray, 
    success_threshold: float = 0.8
) -> float:
    """Compute success rate from rewards.
    
    Args:
        rewards: Array of episode rewards
        success_threshold: Threshold for considering an episode successful
        
    Returns:
        Success rate (0-1)
    """
    successful_episodes = np.sum(rewards >= success_threshold)
    return successful_episodes / len(rewards)


def compute_sample_efficiency(
    training_history: list, 
    target_performance: float = 0.8,
    metric_name: str = "val_loss"
) -> int:
    """Compute sample efficiency (epochs to reach target performance).
    
    Args:
        training_history: List of training metrics per epoch
        target_performance: Target performance threshold
        metric_name: Metric to track
        
    Returns:
        Number of epochs to reach target (or -1 if never reached)
    """
    for i, metrics in enumerate(training_history):
        if metric_name in metrics and metrics[metric_name] <= target_performance:
            return i + 1
    return -1


def create_demonstration_summary(demonstrations: list) -> Dict[str, Any]:
    """Create summary statistics for demonstrations.
    
    Args:
        demonstrations: List of demonstration episodes
        
    Returns:
        Summary statistics dictionary
    """
    if not demonstrations:
        return {}
    
    total_steps = sum(len(demo["states"]) for demo in demonstrations)
    episode_lengths = [len(demo["states"]) for demo in demonstrations]
    
    # State statistics
    all_states = np.concatenate([demo["states"] for demo in demonstrations])
    state_mean = np.mean(all_states, axis=0)
    state_std = np.std(all_states, axis=0)
    
    # Action statistics
    all_actions = np.concatenate([demo["actions"] for demo in demonstrations])
    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0)
    
    return {
        "num_episodes": len(demonstrations),
        "total_steps": total_steps,
        "avg_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.std(episode_lengths),
        "min_episode_length": np.min(episode_lengths),
        "max_episode_length": np.max(episode_lengths),
        "state_dim": all_states.shape[1],
        "action_dim": all_actions.shape[1],
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }
