#!/usr/bin/env python3
"""Main training script for imitation learning."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from imitation_learning import create_agent
from environments import create_robotics_env
from data import ExpertDemonstrationCollector, DemonstrationPreprocessor, DemonstrationValidator
from evaluation import ImitationLearningEvaluator, Leaderboard
from visualization import ImitationLearningVisualizer
from imitation_learning.utils import set_seed, get_device


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_level = config.get("logging", {}).get("level", "INFO")
    log_file = config.get("logging", {}).get("log_file", "logs/imitation_learning.log")
    
    # Create logs directory
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collect_demonstrations(config: Dict[str, Any]) -> list:
    """Collect expert demonstrations."""
    logger = logging.getLogger(__name__)
    
    # Create environment
    env_config = config.get("environment", {})
    env = create_robotics_env(env_config["name"], env_config)
    
    # Create demonstration collector
    collector_config = config.get("data_collection", {})
    collector = ExpertDemonstrationCollector(env, collector_config)
    
    # Collect demonstrations
    num_demos = collector_config.get("num_demonstrations", 100)
    demonstrations = collector.collect_demonstrations(num_demos)
    
    # Validate demonstrations
    validator_config = config.get("data_validation", {})
    validator = DemonstrationValidator(validator_config)
    validation_results = validator.validate(demonstrations)
    
    logger.info(f"Demonstration validation: {validation_results}")
    
    # Save demonstrations
    demo_path = Path("data/demonstrations.pkl")
    collector.save_demonstrations(demonstrations, demo_path)
    
    return demonstrations


def train_agent(config: Dict[str, Any], demonstrations: list) -> Any:
    """Train imitation learning agent."""
    logger = logging.getLogger(__name__)
    
    # Create environment
    env_config = config.get("environment", {})
    env = create_robotics_env(env_config["name"], env_config)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create agent
    algorithm = config.get("algorithm", "behavioral_cloning")
    agent = create_agent(
        algorithm=algorithm,
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        device=get_device(config.get("device", "auto")),
        seed=config.get("seed", 42)
    )
    
    # Preprocess demonstrations
    preprocessor_config = config.get("preprocessing", {})
    preprocessor = DemonstrationPreprocessor(preprocessor_config)
    processed_demos = preprocessor.preprocess(demonstrations)
    
    # Train agent
    training_config = config.get("training", {})
    logger.info(f"Training {algorithm} agent...")
    
    if algorithm == "dagger":
        # Special training for DAgger
        history = agent.dagger_train(
            processed_demos,
            env,
            num_iterations=config.get("dagger", {}).get("iterations", 5)
        )
    else:
        # Standard training
        history = agent.train(
            processed_demos,
            validation_split=training_config.get("validation_split", 0.2),
            num_epochs=training_config.get("num_epochs", 100),
            batch_size=training_config.get("batch_size", 64),
            early_stopping_patience=training_config.get("early_stopping_patience", 10),
            save_best=True
        )
    
    # Save trained model
    model_path = Path(f"models/{algorithm}_model.pth")
    agent.save_model(model_path)
    
    logger.info(f"Training complete. Model saved to {model_path}")
    
    return agent


def evaluate_agent(config: Dict[str, Any], agent: Any) -> Dict[str, float]:
    """Evaluate trained agent."""
    logger = logging.getLogger(__name__)
    
    # Create environment
    env_config = config.get("environment", {})
    env = create_robotics_env(env_config["name"], env_config)
    
    # Create evaluator
    evaluator_config = config.get("evaluation", {})
    evaluator = ImitationLearningEvaluator(evaluator_config)
    
    # Evaluate agent
    logger.info("Evaluating agent...")
    metrics = evaluator.evaluate(agent, env)
    
    # Add to leaderboard
    leaderboard_config = {"results_file": "results/leaderboard.json"}
    leaderboard = Leaderboard(leaderboard_config)
    
    algorithm = config.get("algorithm", "behavioral_cloning")
    leaderboard.add_result(algorithm, config, metrics)
    
    logger.info(f"Evaluation complete: {metrics}")
    
    return metrics


def create_visualizations(config: Dict[str, Any], agent: Any, metrics: Dict[str, float]) -> None:
    """Create visualizations."""
    logger = logging.getLogger(__name__)
    
    # Create visualizer
    viz_config = config.get("visualization", {})
    visualizer = ImitationLearningVisualizer(viz_config)
    
    # Create output directory
    output_dir = Path("assets/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training history
    if hasattr(agent, 'training_history') and agent.training_history:
        visualizer.plot_training_history(
            agent.training_history,
            save_path=output_dir / "training_history.png"
        )
    
    # Plot evaluation metrics
    visualizer.plot_evaluation_metrics(
        metrics,
        save_path=output_dir / "evaluation_metrics.png"
        )
    
    logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train imitation learning agent")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--skip-demos", action="store_true",
                       help="Skip demonstration collection (use existing)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training (use existing model)")
    parser.add_argument("--skip-eval", action="store_true",
                       help="Skip evaluation")
    parser.add_argument("--skip-viz", action="store_true",
                       help="Skip visualization")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    if config.get("seed"):
        set_seed(config["seed"])
    
    logger.info("Starting imitation learning training pipeline")
    logger.info(f"Configuration: {config}")
    
    try:
        # Collect demonstrations
        if not args.skip_demos:
            logger.info("Collecting demonstrations...")
            demonstrations = collect_demonstrations(config)
        else:
            logger.info("Loading existing demonstrations...")
            demo_path = Path("data/demonstrations.pkl")
            if demo_path.exists():
                collector = ExpertDemonstrationCollector(None, {})
                demonstrations = collector.load_demonstrations(demo_path)
            else:
                logger.error("No existing demonstrations found")
                return
        
        # Train agent
        if not args.skip_training:
            logger.info("Training agent...")
            agent = train_agent(config, demonstrations)
        else:
            logger.info("Loading existing model...")
            algorithm = config.get("algorithm", "behavioral_cloning")
            model_path = Path(f"models/{algorithm}_model.pth")
            if model_path.exists():
                # Create agent and load model
                env_config = config.get("environment", {})
                env = create_robotics_env(env_config["name"], env_config)
                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.shape[0]
                
                agent = create_agent(
                    algorithm=algorithm,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    config=config,
                    device=get_device(config.get("device", "auto")),
                    seed=config.get("seed", 42)
                )
                agent.load_model(model_path)
            else:
                logger.error("No existing model found")
                return
        
        # Evaluate agent
        if not args.skip_eval:
            logger.info("Evaluating agent...")
            metrics = evaluate_agent(config, agent)
        else:
            logger.info("Skipping evaluation")
            metrics = {}
        
        # Create visualizations
        if not args.skip_viz:
            logger.info("Creating visualizations...")
            create_visualizations(config, agent, metrics)
        else:
            logger.info("Skipping visualization")
        
        logger.info("Training pipeline complete!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
