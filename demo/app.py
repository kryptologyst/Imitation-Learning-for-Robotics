"""Streamlit demo application for imitation learning."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from imitation_learning import create_agent
from environments import create_robotics_env, get_available_environments
from data import ExpertDemonstrationCollector, DemonstrationPreprocessor
from evaluation import ImitationLearningEvaluator, Leaderboard
from visualization import ImitationLearningVisualizer
from imitation_learning.utils import set_seed, get_device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Imitation Learning for Robotics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_path}")
        return {}

def initialize_session_state():
    """Initialize session state variables."""
    if 'config' not in st.session_state:
        st.session_state.config = load_config("config/default.yaml")
    
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if 'env' not in st.session_state:
        st.session_state.env = None
    
    if 'demonstrations' not in st.session_state:
        st.session_state.demonstrations = []
    
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {}

def create_agent_from_config(config: dict) -> any:
    """Create agent from configuration."""
    try:
        # Create environment
        env_config = config.get("environment", {})
        env_name = env_config.get("name", "FetchReach-v1")
        env = create_robotics_env(env_name, env_config)
        
        # Get dimensions
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
        
        return agent, env
    except Exception as e:
        st.error(f"Error creating agent: {e}")
        return None, None

def main():
    """Main Streamlit application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Imitation Learning for Robotics</h1>', unsafe_allow_html=True)
    
    # Safety disclaimer
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ SAFETY DISCLAIMER</h4>
        <p><strong>THIS SOFTWARE IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY.</strong></p>
        <p>DO NOT USE ON REAL ROBOTS WITHOUT PROPER SAFETY REVIEW AND TESTING.</p>
        <p>This framework is designed for simulation environments and educational demonstrations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Algorithm selection
        algorithm = st.selectbox(
            "Algorithm",
            ["behavioral_cloning", "dagger"],
            index=0
        )
        
        # Environment selection
        env_name = st.selectbox(
            "Environment",
            get_available_environments(),
            index=0
        )
        
        # Model parameters
        st.subheader("Model Parameters")
        hidden_size_1 = st.slider("Hidden Layer 1 Size", 64, 512, 256)
        hidden_size_2 = st.slider("Hidden Layer 2 Size", 64, 512, 256)
        dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.1)
        
        # Training parameters
        st.subheader("Training Parameters")
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
        batch_size = st.slider("Batch Size", 16, 128, 64)
        num_epochs = st.slider("Number of Epochs", 10, 500, 100)
        
        # Evaluation parameters
        st.subheader("Evaluation Parameters")
        num_eval_episodes = st.slider("Evaluation Episodes", 5, 50, 10)
        
        # Update configuration
        config = {
            "algorithm": algorithm,
            "environment": {"name": env_name},
            "model": {
                "hidden_sizes": [hidden_size_1, hidden_size_2],
                "dropout": dropout
            },
            "training": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs
            },
            "evaluation": {
                "num_episodes": num_eval_episodes
            },
            "device": "auto",
            "seed": 42
        }
        
        st.session_state.config = config
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Training", "Evaluation", "Visualization", "Leaderboard"])
    
    with tab1:
        st.header("Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Collect Demonstrations")
            
            if st.button("Collect Expert Demonstrations"):
                with st.spinner("Collecting demonstrations..."):
                    try:
                        # Create environment
                        env_config = config.get("environment", {})
                        env = create_robotics_env(env_config["name"], env_config)
                        
                        # Create collector
                        collector_config = {
                            "expert_type": "random",
                            "num_demonstrations": 50,
                            "success_threshold": 0.8
                        }
                        collector = ExpertDemonstrationCollector(env, collector_config)
                        
                        # Collect demonstrations
                        demonstrations = collector.collect_demonstrations(50)
                        st.session_state.demonstrations = demonstrations
                        
                        st.success(f"Collected {len(demonstrations)} demonstrations")
                        
                    except Exception as e:
                        st.error(f"Error collecting demonstrations: {e}")
        
        with col2:
            st.subheader("Train Agent")
            
            if st.button("Train Imitation Learning Agent"):
                if not st.session_state.demonstrations:
                    st.warning("Please collect demonstrations first")
                else:
                    with st.spinner("Training agent..."):
                        try:
                            # Create agent
                            agent, env = create_agent_from_config(config)
                            
                            if agent is not None:
                                # Preprocess demonstrations
                                preprocessor = DemonstrationPreprocessor({})
                                processed_demos = preprocessor.preprocess(st.session_state.demonstrations)
                                
                                # Train agent
                                history = agent.train(
                                    processed_demos,
                                    validation_split=0.2,
                                    num_epochs=config["training"]["num_epochs"],
                                    batch_size=config["training"]["batch_size"],
                                    save_best=True
                                )
                                
                                st.session_state.agent = agent
                                st.session_state.env = env
                                
                                st.success("Agent trained successfully!")
                                
                                # Show training progress
                                if history and "loss" in history:
                                    st.line_chart(pd.DataFrame(history))
                            
                        except Exception as e:
                            st.error(f"Error training agent: {e}")
    
    with tab2:
        st.header("Evaluation")
        
        if st.session_state.agent is None:
            st.warning("Please train an agent first")
        else:
            if st.button("Evaluate Agent"):
                with st.spinner("Evaluating agent..."):
                    try:
                        # Create evaluator
                        evaluator_config = config.get("evaluation", {})
                        evaluator = ImitationLearningEvaluator(evaluator_config)
                        
                        # Evaluate agent
                        metrics = evaluator.evaluate(
                            st.session_state.agent, 
                            st.session_state.env,
                            num_episodes=config["evaluation"]["num_episodes"]
                        )
                        
                        st.session_state.metrics = metrics
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Success Rate", f"{metrics.get('success_rate', 0):.2%}")
                        
                        with col2:
                            st.metric("Average Reward", f"{metrics.get('avg_reward', 0):.2f}")
                        
                        with col3:
                            st.metric("Avg Episode Length", f"{metrics.get('avg_episode_length', 0):.1f}")
                        
                        with col4:
                            st.metric("Trajectory Error", f"{metrics.get('avg_trajectory_error', 0):.3f}")
                        
                        st.success("Evaluation complete!")
                        
                    except Exception as e:
                        st.error(f"Error evaluating agent: {e}")
    
    with tab3:
        st.header("Visualization")
        
        if st.session_state.agent is None:
            st.warning("Please train an agent first")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Plot Training History"):
                    if hasattr(st.session_state.agent, 'training_history') and st.session_state.agent.training_history:
                        # Create training history plot
                        df = pd.DataFrame(st.session_state.agent.training_history)
                        
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("Training Loss", "Validation Loss")
                        )
                        
                        if "loss" in df.columns:
                            fig.add_trace(
                                go.Scatter(x=df.index, y=df["loss"], name="Training Loss"),
                                row=1, col=1
                            )
                        
                        if "val_loss" in df.columns:
                            fig.add_trace(
                                go.Scatter(x=df.index, y=df["val_loss"], name="Validation Loss"),
                                row=1, col=2
                            )
                        
                        fig.update_layout(height=400, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No training history available")
            
            with col2:
                if st.button("Plot Evaluation Metrics"):
                    if st.session_state.metrics:
                        # Create metrics plot
                        metrics_df = pd.DataFrame([st.session_state.metrics])
                        
                        fig = px.bar(
                            metrics_df,
                            title="Evaluation Metrics",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No evaluation metrics available")
    
    with tab4:
        st.header("Leaderboard")
        
        try:
            # Load leaderboard
            leaderboard_config = {"results_file": "results/leaderboard.json"}
            leaderboard = Leaderboard(leaderboard_config)
            
            # Get leaderboard data
            df = leaderboard.get_leaderboard("success_rate", top_k=10)
            
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                
                # Create leaderboard plot
                fig = px.bar(
                    df,
                    x="algorithm",
                    y="success_rate",
                    title="Success Rate Leaderboard",
                    color="success_rate",
                    color_continuous_scale="viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No results in leaderboard yet. Train and evaluate agents to see results.")
                
        except Exception as e:
            st.error(f"Error loading leaderboard: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        <p>Imitation Learning for Robotics - Research and Educational Use Only</p>
        <p>⚠️ Do not use on real robots without proper safety review</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
