"""Visualization utilities for imitation learning."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class ImitationLearningVisualizer:
    """Visualization utilities for imitation learning."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize visualizer."""
        self.config = config
        self.style = config.get("style", "seaborn-v0_8")
        self.color_palette = config.get("color_palette", "husl")
        self.figsize = config.get("figsize", (10, 6))
        self.dpi = config.get("dpi", 300)
        
        # Set style
        plt.style.use(self.style)
        sns.set_palette(self.color_palette)
    
    def plot_training_history(self, training_history: List[Dict[str, float]], 
                            save_path: Optional[Path] = None) -> None:
        """Plot training history.
        
        Args:
            training_history: List of training metrics per epoch
            save_path: Path to save plot
        """
        if not training_history:
            logger.warning("No training history to plot")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(training_history)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training and validation loss
        if "loss" in df.columns and "val_loss" in df.columns:
            axes[0, 0].plot(df.index, df["loss"], label="Training Loss", alpha=0.8)
            axes[0, 0].plot(df.index, df["val_loss"], label="Validation Loss", alpha=0.8)
            axes[0, 0].set_title("Training and Validation Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot other metrics
        metric_cols = [col for col in df.columns if col not in ["loss", "val_loss"]]
        if metric_cols:
            for i, metric in enumerate(metric_cols[:3]):  # Plot up to 3 additional metrics
                row = (i + 1) // 2
                col = (i + 1) % 2
                if row < 2 and col < 2:
                    axes[row, col].plot(df.index, df[metric], alpha=0.8)
                    axes[row, col].set_title(f"{metric}")
                    axes[row, col].set_xlabel("Epoch")
                    axes[row, col].set_ylabel(metric)
                    axes[row, col].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metric_cols) + 1, 4):
            row = i // 2
            col = i % 2
            if row < 2 and col < 2:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_trajectory_comparison(self, expert_trajectory: np.ndarray, 
                                 learned_trajectory: np.ndarray,
                                 save_path: Optional[Path] = None) -> None:
        """Plot comparison between expert and learned trajectories.
        
        Args:
            expert_trajectory: Expert trajectory
            learned_trajectory: Learned trajectory
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 2D trajectory (if applicable)
        if expert_trajectory.shape[1] >= 2 and learned_trajectory.shape[1] >= 2:
            axes[0].plot(expert_trajectory[:, 0], expert_trajectory[:, 1], 
                        'b-', label='Expert', linewidth=2, alpha=0.8)
            axes[0].plot(learned_trajectory[:, 0], learned_trajectory[:, 1], 
                        'r--', label='Learned', linewidth=2, alpha=0.8)
            axes[0].set_title("Trajectory Comparison (2D)")
            axes[0].set_xlabel("X")
            axes[0].set_ylabel("Y")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].axis('equal')
        
        # Plot trajectory error over time
        min_len = min(len(expert_trajectory), len(learned_trajectory))
        expert_trunc = expert_trajectory[:min_len]
        learned_trunc = learned_trajectory[:min_len]
        
        error = np.linalg.norm(expert_trunc - learned_trunc, axis=1)
        axes[1].plot(error, 'g-', linewidth=2, alpha=0.8)
        axes[1].set_title("Trajectory Error Over Time")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("L2 Error")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Trajectory comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_action_comparison(self, expert_actions: np.ndarray, 
                             learned_actions: np.ndarray,
                             save_path: Optional[Path] = None) -> None:
        """Plot comparison between expert and learned actions.
        
        Args:
            expert_actions: Expert actions
            learned_actions: Learned actions
            save_path: Path to save plot
        """
        min_len = min(len(expert_actions), len(learned_actions))
        expert_trunc = expert_actions[:min_len]
        learned_trunc = learned_actions[:min_len]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot actions over time
        for i in range(min(expert_trunc.shape[1], learned_trunc.shape[1])):
            axes[0].plot(expert_trunc[:, i], label=f'Expert Action {i+1}', alpha=0.8)
            axes[0].plot(learned_trunc[:, i], '--', label=f'Learned Action {i+1}', alpha=0.8)
        
        axes[0].set_title("Action Comparison Over Time")
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("Action Value")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot action error
        action_error = np.linalg.norm(expert_trunc - learned_trunc, axis=1)
        axes[1].plot(action_error, 'r-', linewidth=2, alpha=0.8)
        axes[1].set_title("Action Error Over Time")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("L2 Error")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Action comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_evaluation_metrics(self, metrics: Dict[str, float], 
                               save_path: Optional[Path] = None) -> None:
        """Plot evaluation metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics
            save_path: Path to save plot
        """
        if not metrics:
            logger.warning("No metrics to plot")
            return
        
        # Separate metrics by category
        success_metrics = {k: v for k, v in metrics.items() if 'success' in k.lower()}
        error_metrics = {k: v for k, v in metrics.items() if 'error' in k.lower()}
        other_metrics = {k: v for k, v in metrics.items() 
                         if 'success' not in k.lower() and 'error' not in k.lower()}
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot success metrics
        if success_metrics:
            axes[0].bar(success_metrics.keys(), success_metrics.values(), alpha=0.7)
            axes[0].set_title("Success Metrics")
            axes[0].set_ylabel("Value")
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)
        
        # Plot error metrics
        if error_metrics:
            axes[1].bar(error_metrics.keys(), error_metrics.values(), alpha=0.7, color='red')
            axes[1].set_title("Error Metrics")
            axes[1].set_ylabel("Value")
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)
        
        # Plot other metrics
        if other_metrics:
            axes[2].bar(other_metrics.keys(), other_metrics.values(), alpha=0.7, color='green')
            axes[2].set_title("Other Metrics")
            axes[2].set_ylabel("Value")
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Evaluation metrics plot saved to {save_path}")
        
        plt.show()
    
    def plot_leaderboard(self, leaderboard_df: pd.DataFrame, 
                        metric: str = "success_rate",
                        save_path: Optional[Path] = None) -> None:
        """Plot leaderboard results.
        
        Args:
            leaderboard_df: Leaderboard DataFrame
            metric: Metric to plot
            save_path: Path to save plot
        """
        if leaderboard_df.empty:
            logger.warning("No leaderboard data to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot
        bars = plt.bar(range(len(leaderboard_df)), leaderboard_df[metric], alpha=0.7)
        
        # Color bars by performance
        colors = plt.cm.viridis(leaderboard_df[metric] / leaderboard_df[metric].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title(f"Leaderboard - {metric}")
        plt.xlabel("Algorithm")
        plt.ylabel(metric)
        plt.xticks(range(len(leaderboard_df)), leaderboard_df["algorithm"], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(leaderboard_df[metric]):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Leaderboard plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_plot(self, data: Dict[str, Any], 
                              plot_type: str = "trajectory") -> go.Figure:
        """Create interactive plot using Plotly.
        
        Args:
            data: Data to plot
            plot_type: Type of plot to create
            
        Returns:
            Plotly figure
        """
        if plot_type == "trajectory":
            return self._create_trajectory_plot(data)
        elif plot_type == "metrics":
            return self._create_metrics_plot(data)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def _create_trajectory_plot(self, data: Dict[str, Any]) -> go.Figure:
        """Create interactive trajectory plot."""
        fig = go.Figure()
        
        if "expert_trajectory" in data and "learned_trajectory" in data:
            expert_traj = data["expert_trajectory"]
            learned_traj = data["learned_trajectory"]
            
            # Add expert trajectory
            fig.add_trace(go.Scatter3d(
                x=expert_traj[:, 0],
                y=expert_traj[:, 1],
                z=expert_traj[:, 2] if expert_traj.shape[1] > 2 else np.zeros(len(expert_traj)),
                mode='lines+markers',
                name='Expert',
                line=dict(color='blue', width=4),
                marker=dict(size=3)
            ))
            
            # Add learned trajectory
            fig.add_trace(go.Scatter3d(
                x=learned_traj[:, 0],
                y=learned_traj[:, 1],
                z=learned_traj[:, 2] if learned_traj.shape[1] > 2 else np.zeros(len(learned_traj)),
                mode='lines+markers',
                name='Learned',
                line=dict(color='red', width=4, dash='dash'),
                marker=dict(size=3)
            ))
        
        fig.update_layout(
            title="Interactive Trajectory Comparison",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            )
        )
        
        return fig
    
    def _create_metrics_plot(self, data: Dict[str, Any]) -> go.Figure:
        """Create interactive metrics plot."""
        if "metrics" not in data:
            return go.Figure()
        
        metrics = data["metrics"]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Success Rate", "Average Reward", "Trajectory Error", "Episode Length"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Success rate
        if "success_rate" in metrics:
            fig.add_trace(
                go.Bar(x=["Success Rate"], y=[metrics["success_rate"]], name="Success Rate"),
                row=1, col=1
            )
        
        # Average reward
        if "avg_reward" in metrics:
            fig.add_trace(
                go.Bar(x=["Average Reward"], y=[metrics["avg_reward"]], name="Average Reward"),
                row=1, col=2
            )
        
        # Trajectory error
        if "avg_trajectory_error" in metrics:
            fig.add_trace(
                go.Bar(x=["Trajectory Error"], y=[metrics["avg_trajectory_error"]], name="Trajectory Error"),
                row=2, col=1
            )
        
        # Episode length
        if "avg_episode_length" in metrics:
            fig.add_trace(
                go.Bar(x=["Episode Length"], y=[metrics["avg_episode_length"]], name="Episode Length"),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=False, title_text="Evaluation Metrics")
        
        return fig
    
    def save_all_plots(self, plots_data: Dict[str, Any], output_dir: Path) -> None:
        """Save all plots to directory.
        
        Args:
            plots_data: Dictionary containing plot data
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training history plot
        if "training_history" in plots_data:
            self.plot_training_history(
                plots_data["training_history"],
                save_path=output_dir / "training_history.png"
            )
        
        # Save trajectory comparison plot
        if "expert_trajectory" in plots_data and "learned_trajectory" in plots_data:
            self.plot_trajectory_comparison(
                plots_data["expert_trajectory"],
                plots_data["learned_trajectory"],
                save_path=output_dir / "trajectory_comparison.png"
            )
        
        # Save action comparison plot
        if "expert_actions" in plots_data and "learned_actions" in plots_data:
            self.plot_action_comparison(
                plots_data["expert_actions"],
                plots_data["learned_actions"],
                save_path=output_dir / "action_comparison.png"
            )
        
        # Save evaluation metrics plot
        if "metrics" in plots_data:
            self.plot_evaluation_metrics(
                plots_data["metrics"],
                save_path=output_dir / "evaluation_metrics.png"
            )
        
        logger.info(f"All plots saved to {output_dir}")
