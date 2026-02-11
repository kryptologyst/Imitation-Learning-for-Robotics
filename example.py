#!/usr/bin/env python3
"""Example script demonstrating imitation learning framework."""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from imitation_learning import create_agent
from environments import create_robotics_env
from data import ExpertDemonstrationCollector, DemonstrationPreprocessor
from evaluation import ImitationLearningEvaluator
from visualization import ImitationLearningVisualizer
from imitation_learning.utils import set_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run example imitation learning pipeline."""
    
    print("🤖 Imitation Learning for Robotics - Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Configuration
    config = {
        "algorithm": "behavioral_cloning",
        "environment": {
            "name": "TwoLinkArm-v0",
            "max_episode_steps": 50,
            "target": [1.5, 0.0]
        },
        "model": {
            "hidden_sizes": [128, 128],
            "activation": "relu",
            "dropout": 0.1
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 50,
            "validation_split": 0.2,
            "early_stopping_patience": 10
        },
        "data_collection": {
            "expert_type": "random",
            "num_demonstrations": 20,
            "success_threshold": 0.5
        },
        "evaluation": {
            "num_episodes": 5,
            "success_threshold": 0.5
        },
        "device": "cpu",
        "seed": 42
    }
    
    print(f"Configuration: {config['algorithm']} on {config['environment']['name']}")
    
    # Step 1: Create environment
    print("\n1. Creating environment...")
    env = create_robotics_env(config["environment"]["name"], config["environment"])
    print(f"   Environment: {env.__class__.__name__}")
    print(f"   State space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
    
    # Step 2: Collect demonstrations
    print("\n2. Collecting expert demonstrations...")
    collector_config = config["data_collection"]
    collector = ExpertDemonstrationCollector(env, collector_config)
    
    num_demos = collector_config["num_demonstrations"]
    demonstrations = collector.collect_demonstrations(num_demos)
    print(f"   Collected {len(demonstrations)} demonstrations")
    
    # Step 3: Preprocess data
    print("\n3. Preprocessing demonstrations...")
    preprocessor = DemonstrationPreprocessor({})
    processed_demos = preprocessor.preprocess(demonstrations)
    print(f"   Preprocessed {len(processed_demos)} demonstrations")
    
    # Step 4: Create and train agent
    print("\n4. Creating and training agent...")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = create_agent(
        algorithm=config["algorithm"],
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        device=config["device"],
        seed=config["seed"]
    )
    
    training_config = config["training"]
    print(f"   Training {config['algorithm']} agent...")
    
    history = agent.train(
        processed_demos,
        validation_split=training_config["validation_split"],
        num_epochs=training_config["num_epochs"],
        batch_size=training_config["batch_size"],
        early_stopping_patience=training_config["early_stopping_patience"],
        save_best=True
    )
    
    print(f"   Training complete! Final loss: {history.get('loss', [0])[-1]:.4f}")
    
    # Step 5: Evaluate agent
    print("\n5. Evaluating agent...")
    evaluator_config = config["evaluation"]
    evaluator = ImitationLearningEvaluator(evaluator_config)
    
    metrics = evaluator.evaluate(agent, env)
    print(f"   Success rate: {metrics.get('success_rate', 0):.2%}")
    print(f"   Average reward: {metrics.get('avg_reward', 0):.2f}")
    print(f"   Average episode length: {metrics.get('avg_episode_length', 0):.1f}")
    
    # Step 6: Create visualizations
    print("\n6. Creating visualizations...")
    visualizer = ImitationLearningVisualizer({})
    
    # Create output directory
    output_dir = Path("assets/example_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training history
    if hasattr(agent, 'training_history') and agent.training_history:
        visualizer.plot_training_history(
            agent.training_history,
            save_path=output_dir / "training_history.png"
        )
        print(f"   Training history plot saved to {output_dir / 'training_history.png'}")
    
    # Plot evaluation metrics
    visualizer.plot_evaluation_metrics(
        metrics,
        save_path=output_dir / "evaluation_metrics.png"
    )
    print(f"   Evaluation metrics plot saved to {output_dir / 'evaluation_metrics.png'}")
    
    # Step 7: Test agent on new episodes
    print("\n7. Testing agent on new episodes...")
    for episode in range(3):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 50:
            action = agent.predict(state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1
        
        print(f"   Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}")
    
    print("\n✅ Example complete!")
    print(f"📊 Results saved to {output_dir}")
    print("\n⚠️  Remember: This is for simulation only. Do not use on real robots!")
    
    # Save model
    model_path = Path("models/example_model.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save_model(model_path)
    print(f"💾 Model saved to {model_path}")


if __name__ == "__main__":
    main()
