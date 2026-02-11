# Imitation Learning for Robotics

Research-ready framework for imitation learning in robotics, featuring behavioral cloning, DAgger, GAIL, and advanced evaluation metrics.

## DISCLAIMER

**THIS SOFTWARE IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY. DO NOT USE ON REAL ROBOTS WITHOUT PROPER SAFETY REVIEW AND TESTING.**

This framework is designed for simulation environments and educational demonstrations. Real-world robotic applications require:
- Safety certification and testing
- Hardware-specific validation
- Emergency stop mechanisms
- Velocity and force limits
- Professional robotics expertise

## Features

- **Multiple Imitation Learning Algorithms**: Behavioral Cloning, DAgger, GAIL, AIRL
- **Modern ML Stack**: PyTorch 2.x, Gymnasium, Stable-Baselines3
- **Comprehensive Evaluation**: Success rates, trajectory metrics, ablation studies
- **Interactive Demos**: Streamlit interface for visualization and testing
- **ROS 2 Integration**: Ready for robotics simulation environments
- **Production Ready**: Type hints, testing, CI/CD, proper documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Imitation-Learning-for-Robotics.git
cd Imitation-Learning-for-Robotics

# Install dependencies
pip install -e .

# For ROS 2 support (optional)
pip install -e ".[ros2]"

# For simulation environments (optional)
pip install -e ".[simulation]"
```

### Basic Usage

```python
from src.imitation_learning import ImitationLearningAgent
from src.environments import create_robotics_env
from src.data import DemonstrationCollector

# Create environment
env = create_robotics_env("FetchReach-v1")

# Collect demonstrations
collector = DemonstrationCollector(env)
demonstrations = collector.collect_demonstrations(num_episodes=10)

# Train imitation learning agent
agent = ImitationLearningAgent(
    algorithm="behavioral_cloning",
    env=env,
    config_path="config/bc_config.yaml"
)
agent.train(demonstrations)

# Evaluate
results = agent.evaluate(num_episodes=5)
print(f"Success rate: {results['success_rate']:.2%}")
```

### Interactive Demo

```bash
streamlit run demo/app.py
```

## Project Structure

```
src/
├── imitation_learning/     # Core imitation learning algorithms
├── environments/           # Robotics environments and wrappers
├── data/                  # Data collection and preprocessing
├── evaluation/            # Metrics and evaluation tools
├── visualization/         # Plotting and visualization utilities
└── utils/                 # Common utilities and helpers

config/                    # Configuration files (YAML)
data/                      # Demonstration datasets
assets/                    # Visualizations, videos, plots
tests/                     # Unit and integration tests
demo/                      # Interactive demos
docs/                      # Documentation
```

## Algorithms

### Behavioral Cloning (BC)
Simple supervised learning approach that directly maps states to actions.

### DAgger (Dataset Aggregation)
Iterative approach that collects additional data from the learned policy.

### GAIL (Generative Adversarial Imitation Learning)
Adversarial approach using a discriminator to distinguish expert from learned policy.

### AIRL (Adversarial Inverse Reinforcement Learning)
Learns both policy and reward function simultaneously.

## Evaluation Metrics

- **Success Rate**: Percentage of successful task completions
- **Trajectory Error**: L2 distance from expert trajectories
- **Sample Efficiency**: Episodes needed to reach performance threshold
- **Generalization**: Performance on unseen scenarios
- **Robustness**: Performance under noise and perturbations

## Configuration

All hyperparameters and settings are managed through YAML configuration files:

```yaml
# config/bc_config.yaml
algorithm: behavioral_cloning
model:
  hidden_sizes: [256, 256]
  activation: relu
  dropout: 0.1
training:
  batch_size: 64
  learning_rate: 0.001
  num_epochs: 100
evaluation:
  num_episodes: 10
  success_threshold: 0.8
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run linting: `black . && ruff check .`
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{imitation_learning_robotics,
  title={Imitation Learning for Robotics: A Modern Framework},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Imitation-Learning-for-Robotics}
}
```
# Imitation-Learning-for-Robotics
