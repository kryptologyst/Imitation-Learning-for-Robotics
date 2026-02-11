"""Evaluation metrics and leaderboard for imitation learning."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Abstract base class for evaluators."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize evaluator."""
        self.config = config
        
    @abstractmethod
    def evaluate(self, agent, env, num_episodes: int) -> Dict[str, float]:
        """Evaluate agent performance."""
        pass


class ImitationLearningEvaluator(BaseEvaluator):
    """Comprehensive evaluator for imitation learning agents."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize evaluator."""
        super().__init__(config)
        
        # Evaluation parameters
        self.num_episodes = config.get("num_episodes", 10)
        self.success_threshold = config.get("success_threshold", 0.8)
        self.trajectory_error_threshold = config.get("trajectory_error_threshold", 0.1)
        
        # Metrics to compute
        self.compute_success_rate = config.get("compute_success_rate", True)
        self.compute_trajectory_error = config.get("compute_trajectory_error", True)
        self.compute_sample_efficiency = config.get("compute_sample_efficiency", True)
        self.compute_generalization = config.get("compute_generalization", True)
        self.compute_robustness = config.get("compute_robustness", True)
        
    def evaluate(self, agent, env, num_episodes: Optional[int] = None) -> Dict[str, float]:
        """Evaluate agent performance.
        
        Args:
            agent: Trained imitation learning agent
            env: Environment to evaluate on
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        if num_episodes is None:
            num_episodes = self.num_episodes
            
        logger.info(f"Evaluating agent for {num_episodes} episodes")
        
        # Run evaluation episodes
        episode_results = self._run_evaluation_episodes(agent, env, num_episodes)
        
        # Compute metrics
        metrics = {}
        
        if self.compute_success_rate:
            metrics.update(self._compute_success_metrics(episode_results))
        
        if self.compute_trajectory_error:
            metrics.update(self._compute_trajectory_metrics(episode_results))
        
        if self.compute_sample_efficiency:
            metrics.update(self._compute_sample_efficiency_metrics(agent))
        
        if self.compute_generalization:
            metrics.update(self._compute_generalization_metrics(agent, env))
        
        if self.compute_robustness:
            metrics.update(self._compute_robustness_metrics(agent, env))
        
        # Add episode statistics
        metrics.update(self._compute_episode_statistics(episode_results))
        
        logger.info(f"Evaluation complete: {metrics}")
        return metrics
    
    def _run_evaluation_episodes(self, agent, env, num_episodes: int) -> List[Dict[str, Any]]:
        """Run evaluation episodes."""
        episode_results = []
        
        for episode in range(num_episodes):
            logger.info(f"Running evaluation episode {episode + 1}/{num_episodes}")
            
            # Reset environment
            state, info = env.reset()
            
            # Initialize episode data
            states = []
            actions = []
            rewards = []
            expert_actions = []  # For trajectory comparison
            
            done = False
            step = 0
            
            while not done:
                # Get agent action
                action = agent.predict(state)
                
                # Store data
                states.append(state)
                actions.append(action)
                
                # Take step
                next_state, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                
                # Get expert action for comparison (if available)
                if hasattr(env, '_get_expert_action'):
                    expert_action = env._get_expert_action(state)
                    expert_actions.append(expert_action)
                
                # Update state
                state = next_state
                done = terminated or truncated
                step += 1
            
            # Store episode results
            episode_result = {
                "episode": episode,
                "states": np.array(states),
                "actions": np.array(actions),
                "rewards": np.array(rewards),
                "expert_actions": np.array(expert_actions) if expert_actions else None,
                "episode_reward": sum(rewards),
                "episode_length": len(states),
                "success": sum(rewards) >= self.success_threshold,
            }
            episode_results.append(episode_result)
        
        return episode_results
    
    def _compute_success_metrics(self, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute success-related metrics."""
        rewards = [result["episode_reward"] for result in episode_results]
        successes = [result["success"] for result in episode_results]
        
        success_rate = np.mean(successes)
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        
        return {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
        }
    
    def _compute_trajectory_metrics(self, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute trajectory-related metrics."""
        trajectory_errors = []
        action_errors = []
        
        for result in episode_results:
            if result["expert_actions"] is not None:
                # Compute action error
                actions = result["actions"]
                expert_actions = result["expert_actions"]
                
                # Ensure same length
                min_len = min(len(actions), len(expert_actions))
                actions = actions[:min_len]
                expert_actions = expert_actions[:min_len]
                
                # L2 action error
                action_error = np.mean(np.linalg.norm(actions - expert_actions, axis=1))
                action_errors.append(action_error)
                
                # Trajectory error (simplified - using state differences)
                states = result["states"][:min_len]
                if len(states) > 1:
                    state_diff = np.diff(states, axis=0)
                    expert_state_diff = np.diff(expert_actions, axis=0)  # Simplified
                    trajectory_error = np.mean(np.linalg.norm(state_diff - expert_state_diff, axis=1))
                    trajectory_errors.append(trajectory_error)
        
        metrics = {}
        
        if action_errors:
            metrics.update({
                "avg_action_error": np.mean(action_errors),
                "std_action_error": np.std(action_errors),
                "max_action_error": np.max(action_errors),
            })
        
        if trajectory_errors:
            metrics.update({
                "avg_trajectory_error": np.mean(trajectory_errors),
                "std_trajectory_error": np.std(trajectory_errors),
                "max_trajectory_error": np.max(trajectory_errors),
            })
        
        return metrics
    
    def _compute_sample_efficiency_metrics(self, agent) -> Dict[str, float]:
        """Compute sample efficiency metrics."""
        if not hasattr(agent, 'training_history') or not agent.training_history:
            return {}
        
        # Find epoch where validation loss reaches threshold
        target_loss = 0.1  # Threshold for "good" performance
        epochs_to_threshold = -1
        
        for i, metrics in enumerate(agent.training_history):
            if "val_loss" in metrics and metrics["val_loss"] <= target_loss:
                epochs_to_threshold = i + 1
                break
        
        # Compute final performance
        final_loss = agent.training_history[-1].get("val_loss", float('inf'))
        
        return {
            "epochs_to_threshold": epochs_to_threshold,
            "final_loss": final_loss,
            "total_epochs": len(agent.training_history),
        }
    
    def _compute_generalization_metrics(self, agent, env) -> Dict[str, float]:
        """Compute generalization metrics."""
        # Test on different initial conditions
        generalization_results = []
        
        for _ in range(5):  # Test on 5 different initial conditions
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.predict(state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            generalization_results.append(episode_reward)
        
        return {
            "generalization_avg_reward": np.mean(generalization_results),
            "generalization_std_reward": np.std(generalization_results),
        }
    
    def _compute_robustness_metrics(self, agent, env) -> Dict[str, float]:
        """Compute robustness metrics."""
        # Test with noise
        noise_levels = [0.0, 0.1, 0.2, 0.5]
        robustness_results = {}
        
        for noise_level in noise_levels:
            episode_rewards = []
            
            for _ in range(3):  # Test each noise level 3 times
                state, _ = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    # Add noise to state
                    noisy_state = state + np.random.normal(0, noise_level, state.shape)
                    action = agent.predict(noisy_state)
                    state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                
                episode_rewards.append(episode_reward)
            
            robustness_results[f"robustness_noise_{noise_level}"] = np.mean(episode_rewards)
        
        return robustness_results
    
    def _compute_episode_statistics(self, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute episode statistics."""
        episode_lengths = [result["episode_length"] for result in episode_results]
        
        return {
            "avg_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths),
            "min_episode_length": np.min(episode_lengths),
            "max_episode_length": np.max(episode_lengths),
        }


class Leaderboard:
    """Leaderboard for tracking and comparing algorithm performance."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize leaderboard."""
        self.config = config
        self.results_file = Path(config.get("results_file", "results/leaderboard.json"))
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing results
        self.results = self._load_results()
        
    def add_result(self, algorithm: str, config: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """Add a result to the leaderboard.
        
        Args:
            algorithm: Algorithm name
            config: Algorithm configuration
            metrics: Evaluation metrics
        """
        result = {
            "algorithm": algorithm,
            "config": config,
            "metrics": metrics,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        
        self.results.append(result)
        self._save_results()
        
        logger.info(f"Added result for {algorithm} to leaderboard")
    
    def get_leaderboard(self, metric: str = "success_rate", top_k: int = 10) -> pd.DataFrame:
        """Get leaderboard sorted by metric.
        
        Args:
            metric: Metric to sort by
            top_k: Number of top results to return
            
        Returns:
            DataFrame with top results
        """
        if not self.results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Extract metrics
        metrics_df = pd.json_normalize(df["metrics"])
        df = pd.concat([df[["algorithm", "timestamp"]], metrics_df], axis=1)
        
        # Sort by metric
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
        
        return df.head(top_k)
    
    def plot_leaderboard(self, metric: str = "success_rate", save_path: Optional[Path] = None) -> None:
        """Plot leaderboard results.
        
        Args:
            metric: Metric to plot
            save_path: Path to save plot
        """
        df = self.get_leaderboard(metric, top_k=20)
        
        if df.empty:
            logger.warning("No results to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot
        plt.subplot(2, 1, 1)
        sns.barplot(data=df, x="algorithm", y=metric)
        plt.title(f"Leaderboard - {metric}")
        plt.xticks(rotation=45)
        plt.ylabel(metric)
        
        # Create scatter plot of all metrics
        plt.subplot(2, 1, 2)
        if len(df.columns) > 2:
            # Plot multiple metrics
            metrics_to_plot = [col for col in df.columns if col not in ["algorithm", "timestamp"]]
            for i, metric_col in enumerate(metrics_to_plot[:3]):  # Plot up to 3 metrics
                plt.scatter(df.index, df[metric_col], label=metric_col, alpha=0.7)
            plt.xlabel("Rank")
            plt.ylabel("Metric Value")
            plt.title("Multiple Metrics Comparison")
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Leaderboard plot saved to {save_path}")
        
        plt.show()
    
    def _load_results(self) -> List[Dict[str, Any]]:
        """Load results from file."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_results(self) -> None:
        """Save results to file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)


class AblationStudy:
    """Conduct ablation studies for algorithm components."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ablation study."""
        self.config = config
        self.results = []
        
    def run_ablation(self, base_config: Dict[str, Any], ablation_configs: List[Dict[str, Any]], 
                    agent_factory, env_factory, evaluator) -> pd.DataFrame:
        """Run ablation study.
        
        Args:
            base_config: Base configuration
            ablation_configs: List of ablation configurations
            agent_factory: Function to create agent
            env_factory: Function to create environment
            evaluator: Evaluator instance
            
        Returns:
            DataFrame with ablation results
        """
        logger.info(f"Running ablation study with {len(ablation_configs)} configurations")
        
        results = []
        
        for i, ablation_config in enumerate(ablation_configs):
            logger.info(f"Running ablation {i + 1}/{len(ablation_configs)}")
            
            # Merge base config with ablation config
            config = {**base_config, **ablation_config}
            
            # Create agent and environment
            agent = agent_factory(config)
            env = env_factory(config)
            
            # Train agent
            # This would need to be implemented based on the specific training process
            # For now, we'll assume the agent is already trained
            
            # Evaluate agent
            metrics = evaluator.evaluate(agent, env)
            
            # Store results
            result = {
                "ablation_id": i,
                "config": config,
                "metrics": metrics,
            }
            results.append(result)
        
        # Convert to DataFrame
        df_results = []
        for result in results:
            row = {"ablation_id": result["ablation_id"]}
            row.update(result["metrics"])
            df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        logger.info("Ablation study complete")
        return df
    
    def plot_ablation_results(self, df: pd.DataFrame, metric: str = "success_rate", 
                            save_path: Optional[Path] = None) -> None:
        """Plot ablation study results.
        
        Args:
            df: Ablation results DataFrame
            metric: Metric to plot
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(df["ablation_id"], df[metric], marker='o', linewidth=2, markersize=8)
        plt.xlabel("Ablation Configuration")
        plt.ylabel(metric)
        plt.title(f"Ablation Study - {metric}")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Ablation plot saved to {save_path}")
        
        plt.show()
