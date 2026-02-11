"""Tests for imitation learning module."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from imitation_learning import BehavioralCloningAgent, DAggerAgent, create_agent
from imitation_learning.utils import set_seed, get_device, EarlyStopping


class TestBehavioralCloningAgent:
    """Test Behavioral Cloning agent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.state_dim = 10
        self.action_dim = 4
        self.config = {
            "model": {"hidden_sizes": [64, 64], "activation": "relu", "dropout": 0.1},
            "training": {"learning_rate": 0.001, "weight_decay": 0.0, "loss": "mse"}
        }
        self.device = "cpu"
        self.seed = 42
        
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = BehavioralCloningAgent(
            self.state_dim, self.action_dim, self.config, self.device, self.seed
        )
        
        assert agent.state_dim == self.state_dim
        assert agent.action_dim == self.action_dim
        assert agent.config == self.config
        assert agent.device == self.device
        assert agent.seed == self.seed
        
    def test_model_creation(self):
        """Test model creation."""
        agent = BehavioralCloningAgent(
            self.state_dim, self.action_dim, self.config, self.device, self.seed
        )
        
        # Test model structure
        assert isinstance(agent.model, torch.nn.Module)
        
        # Test forward pass
        test_input = torch.randn(1, self.state_dim)
        output = agent.model(test_input)
        assert output.shape == (1, self.action_dim)
        
    def test_training_step(self):
        """Test training step."""
        agent = BehavioralCloningAgent(
            self.state_dim, self.action_dim, self.config, self.device, self.seed
        )
        
        # Create test data
        states = torch.randn(32, self.state_dim)
        actions = torch.randn(32, self.action_dim)
        
        # Test training step
        metrics = agent.train_step(states, actions)
        
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] >= 0
        
    def test_prediction(self):
        """Test action prediction."""
        agent = BehavioralCloningAgent(
            self.state_dim, self.action_dim, self.config, self.device, self.seed
        )
        
        # Test prediction
        state = np.random.randn(self.state_dim)
        action = agent.predict(state)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (self.action_dim,)
        
    def test_save_load_model(self, tmp_path):
        """Test model saving and loading."""
        agent = BehavioralCloningAgent(
            self.state_dim, self.action_dim, self.config, self.device, self.seed
        )
        
        # Save model
        model_path = tmp_path / "test_model.pth"
        agent.save_model(model_path)
        
        assert model_path.exists()
        
        # Create new agent and load model
        new_agent = BehavioralCloningAgent(
            self.state_dim, self.action_dim, self.config, self.device, self.seed
        )
        new_agent.load_model(model_path)
        
        # Test that models are the same
        state = np.random.randn(self.state_dim)
        action1 = agent.predict(state)
        action2 = new_agent.predict(state)
        
        np.testing.assert_array_almost_equal(action1, action2, decimal=5)


class TestDAggerAgent:
    """Test DAgger agent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.state_dim = 10
        self.action_dim = 4
        self.config = {
            "model": {"hidden_sizes": [64, 64], "activation": "relu", "dropout": 0.1},
            "training": {"learning_rate": 0.001, "weight_decay": 0.0, "loss": "mse"},
            "dagger": {"beta": 0.5, "iterations": 3}
        }
        self.device = "cpu"
        self.seed = 42
        
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = DAggerAgent(
            self.state_dim, self.action_dim, self.config, self.device, self.seed
        )
        
        assert agent.state_dim == self.state_dim
        assert agent.action_dim == self.action_dim
        assert agent.beta == 0.5
        assert agent.iterations == 3


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seeds are set
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Generate random numbers
        np_val1 = np.random.random()
        torch_val1 = torch.rand(1).item()
        
        # Reset seed and generate again
        set_seed(42)
        np_val2 = np.random.random()
        torch_val2 = torch.rand(1).item()
        
        # Should be the same
        assert np_val1 == np_val2
        assert torch_val1 == torch_val2
        
    def test_get_device(self):
        """Test device selection."""
        # Test auto device selection
        device = get_device("auto")
        assert device in ["cuda", "mps", "cpu"]
        
        # Test specific device
        device = get_device("cpu")
        assert device == "cpu"
        
    def test_early_stopping(self):
        """Test early stopping."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        
        # Test improvement
        assert not early_stopping(0.5)
        assert not early_stopping(0.4)
        assert not early_stopping(0.3)
        
        # Test no improvement
        assert not early_stopping(0.31)  # Within min_delta
        assert not early_stopping(0.32)
        assert not early_stopping(0.33)
        assert early_stopping(0.34)  # Should trigger early stopping


class TestAgentFactory:
    """Test agent factory function."""
    
    def test_create_behavioral_cloning_agent(self):
        """Test creating behavioral cloning agent."""
        config = {
            "model": {"hidden_sizes": [64, 64]},
            "training": {"learning_rate": 0.001}
        }
        
        agent = create_agent("behavioral_cloning", 10, 4, config)
        
        assert isinstance(agent, BehavioralCloningAgent)
        assert agent.state_dim == 10
        assert agent.action_dim == 4
        
    def test_create_dagger_agent(self):
        """Test creating DAgger agent."""
        config = {
            "model": {"hidden_sizes": [64, 64]},
            "training": {"learning_rate": 0.001},
            "dagger": {"beta": 0.5, "iterations": 3}
        }
        
        agent = create_agent("dagger", 10, 4, config)
        
        assert isinstance(agent, DAggerAgent)
        assert agent.state_dim == 10
        assert agent.action_dim == 4
        
    def test_create_unknown_agent(self):
        """Test creating unknown agent."""
        config = {"model": {"hidden_sizes": [64, 64]}}
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            create_agent("unknown_algorithm", 10, 4, config)


if __name__ == "__main__":
    pytest.main([__file__])
