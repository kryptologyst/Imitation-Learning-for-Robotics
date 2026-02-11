"""Core imitation learning algorithms for robotics."""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import pickle
import json

from .utils import set_seed, get_device, EarlyStopping


logger = logging.getLogger(__name__)


class BaseImitationLearningAgent(ABC):
    """Abstract base class for imitation learning agents."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the imitation learning agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
            device: Device to run on ('cuda', 'mps', 'cpu')
            seed: Random seed for reproducibility
        """
        if seed is not None:
            set_seed(seed)
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = get_device(device)
        self.seed = seed
        
        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = self._build_optimizer()
        self.criterion = self._build_criterion()
        
        # Training state
        self.training_history: List[Dict[str, float]] = []
        self.best_model_state: Optional[Dict[str, Any]] = None
        
    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Build the neural network model."""
        pass
        
    @abstractmethod
    def _build_optimizer(self) -> optim.Optimizer:
        """Build the optimizer."""
        pass
        
    @abstractmethod
    def _build_criterion(self) -> nn.Module:
        """Build the loss function."""
        pass
        
    @abstractmethod
    def train_step(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one training step."""
        pass
        
    def train(
        self,
        demonstrations: List[Dict[str, np.ndarray]],
        validation_split: float = 0.2,
        num_epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 10,
        save_best: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the imitation learning agent.
        
        Args:
            demonstrations: List of demonstration episodes
            validation_split: Fraction of data to use for validation
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save the best model
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training with {len(demonstrations)} demonstrations")
        
        # Prepare data
        states, actions = self._prepare_data(demonstrations)
        
        # Split into train/validation
        train_states, val_states, train_actions, val_actions = self._split_data(
            states, actions, validation_split
        )
        
        # Create data loaders
        train_loader = self._create_data_loader(
            train_states, train_actions, batch_size, shuffle=True
        )
        val_loader = self._create_data_loader(
            val_states, val_actions, batch_size, shuffle=False
        )
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader)
            
            # Log metrics
            metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            self.training_history.append(metrics)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: {metrics}")
            
            # Early stopping
            if early_stopping(val_metrics["loss"]):
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            # Save best model
            if save_best and val_metrics["loss"] < min(
                h["val_loss"] for h in self.training_history[:-1]
            ):
                self.best_model_state = self.model.state_dict().copy()
        
        # Load best model
        if save_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        return self._summarize_training_history()
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action for given state.
        
        Args:
            state: Input state
            
        Returns:
            Predicted action
        """
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            action_tensor = self.model(state_tensor)
            return action_tensor.cpu().numpy().squeeze()
    
    def save_model(self, path: Union[str, Path]) -> None:
        """Save the trained model.
        
        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "training_history": self.training_history,
            "seed": self.seed,
        }
        
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]) -> None:
        """Load a trained model.
        
        Args:
            path: Path to load the model from
        """
        path = Path(path)
        save_dict = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(save_dict["model_state_dict"])
        self.config = save_dict["config"]
        self.training_history = save_dict["training_history"]
        self.seed = save_dict["seed"]
        
        logger.info(f"Model loaded from {path}")
    
    def _prepare_data(
        self, demonstrations: List[Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare demonstration data for training."""
        states_list = []
        actions_list = []
        
        for demo in demonstrations:
            states_list.append(demo["states"])
            actions_list.append(demo["actions"])
        
        states = np.concatenate(states_list, axis=0)
        actions = np.concatenate(actions_list, axis=0)
        
        return states, actions
    
    def _split_data(
        self, 
        states: np.ndarray, 
        actions: np.ndarray, 
        validation_split: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and validation sets."""
        n_samples = len(states)
        n_val = int(n_samples * validation_split)
        
        # Random permutation
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        return (
            states[train_indices],
            states[val_indices],
            actions[train_indices],
            actions[val_indices],
        )
    
    def _create_data_loader(
        self, 
        states: np.ndarray, 
        actions: np.ndarray, 
        batch_size: int, 
        shuffle: bool
    ) -> torch.utils.data.DataLoader:
        """Create PyTorch data loader."""
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
    
    def _train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for states, actions in train_loader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            
            metrics = self.train_step(states, actions)
            total_loss += metrics["loss"]
            num_batches += 1
        
        return {"loss": total_loss / num_batches}
    
    def _validate_epoch(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                predictions = self.model(states)
                loss = self.criterion(predictions, actions)
                total_loss += loss.item()
                num_batches += 1
        
        return {"loss": total_loss / num_batches}
    
    def _summarize_training_history(self) -> Dict[str, List[float]]:
        """Summarize training history."""
        if not self.training_history:
            return {}
        
        summary = {}
        for key in self.training_history[0].keys():
            summary[key] = [epoch[key] for epoch in self.training_history]
        
        return summary


class BehavioralCloningAgent(BaseImitationLearningAgent):
    """Behavioral Cloning agent for imitation learning."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Behavioral Cloning agent."""
        super().__init__(state_dim, action_dim, config, device, seed)
    
    def _build_model(self) -> nn.Module:
        """Build the neural network model."""
        hidden_sizes = self.config.get("model", {}).get("hidden_sizes", [256, 256])
        activation = self.config.get("model", {}).get("activation", "relu")
        dropout = self.config.get("model", {}).get("dropout", 0.1)
        
        layers = []
        input_size = self.state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, self.action_dim))
        
        return nn.Sequential(*layers)
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build the optimizer."""
        lr = self.config.get("training", {}).get("learning_rate", 0.001)
        weight_decay = self.config.get("training", {}).get("weight_decay", 0.0)
        
        return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _build_criterion(self) -> nn.Module:
        """Build the loss function."""
        loss_type = self.config.get("training", {}).get("loss", "mse")
        
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "l1":
            return nn.L1Loss()
        elif loss_type == "smooth_l1":
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def train_step(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one training step."""
        self.optimizer.zero_grad()
        
        predictions = self.model(states)
        loss = self.criterion(predictions, actions)
        
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}


class DAggerAgent(BaseImitationLearningAgent):
    """DAgger (Dataset Aggregation) agent for imitation learning."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize DAgger agent."""
        super().__init__(state_dim, action_dim, config, device, seed)
        self.beta = config.get("dagger", {}).get("beta", 0.5)  # Mixing parameter
        self.iterations = config.get("dagger", {}).get("iterations", 5)
    
    def _build_model(self) -> nn.Module:
        """Build the neural network model."""
        hidden_sizes = self.config.get("model", {}).get("hidden_sizes", [256, 256])
        activation = self.config.get("model", {}).get("activation", "relu")
        dropout = self.config.get("model", {}).get("dropout", 0.1)
        
        layers = []
        input_size = self.state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, self.action_dim))
        
        return nn.Sequential(*layers)
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build the optimizer."""
        lr = self.config.get("training", {}).get("learning_rate", 0.001)
        weight_decay = self.config.get("training", {}).get("weight_decay", 0.0)
        
        return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _build_criterion(self) -> nn.Module:
        """Build the loss function."""
        loss_type = self.config.get("training", {}).get("loss", "mse")
        
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "l1":
            return nn.L1Loss()
        elif loss_type == "smooth_l1":
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def train_step(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one training step."""
        self.optimizer.zero_grad()
        
        predictions = self.model(states)
        loss = self.criterion(predictions, actions)
        
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
    def dagger_train(
        self,
        expert_demonstrations: List[Dict[str, np.ndarray]],
        env,
        num_iterations: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Train using DAgger algorithm.
        
        Args:
            expert_demonstrations: Initial expert demonstrations
            env: Environment for rollouts
            num_iterations: Number of DAgger iterations
            
        Returns:
            Training history
        """
        if num_iterations is None:
            num_iterations = self.iterations
        
        logger.info(f"Starting DAgger training with {num_iterations} iterations")
        
        # Start with expert demonstrations
        current_demonstrations = expert_demonstrations.copy()
        
        for iteration in range(num_iterations):
            logger.info(f"DAgger iteration {iteration + 1}/{num_iterations}")
            
            # Train on current dataset
            history = self.train(
                current_demonstrations,
                validation_split=0.2,
                num_epochs=50,
                save_best=True,
            )
            
            # Collect new rollouts
            new_demonstrations = self._collect_rollouts(
                env, num_episodes=10, beta=self.beta
            )
            
            # Mix datasets
            current_demonstrations.extend(new_demonstrations)
            
            logger.info(f"Dataset size: {len(current_demonstrations)} episodes")
        
        return history
    
    def _collect_rollouts(
        self, env, num_episodes: int, beta: float
    ) -> List[Dict[str, np.ndarray]]:
        """Collect rollouts using current policy mixed with expert."""
        demonstrations = []
        
        for episode in range(num_episodes):
            states = []
            actions = []
            
            state, _ = env.reset()
            done = False
            
            while not done:
                states.append(state)
                
                # Mix expert and learned policy
                if np.random.random() < beta:
                    # Use expert action (simplified - in practice, query expert)
                    action = env.action_space.sample()  # Placeholder
                else:
                    # Use learned policy
                    action = self.predict(state)
                
                actions.append(action)
                state, _, done, truncated, _ = env.step(action)
                done = done or truncated
            
            demonstrations.append({
                "states": np.array(states),
                "actions": np.array(actions),
            })
        
        return demonstrations


def create_agent(
    algorithm: str,
    state_dim: int,
    action_dim: int,
    config: Dict[str, Any],
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> BaseImitationLearningAgent:
    """Factory function to create imitation learning agents.
    
    Args:
        algorithm: Algorithm name ('behavioral_cloning', 'dagger', 'gail')
        state_dim: State dimension
        action_dim: Action dimension
        config: Configuration dictionary
        device: Device to run on
        seed: Random seed
        
    Returns:
        Imitation learning agent
    """
    if algorithm == "behavioral_cloning":
        return BehavioralCloningAgent(state_dim, action_dim, config, device, seed)
    elif algorithm == "dagger":
        return DAggerAgent(state_dim, action_dim, config, device, seed)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
