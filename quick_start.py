#!/usr/bin/env python3
"""Quick start script for imitation learning framework."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Run quick start setup."""
    print("🚀 Imitation Learning for Robotics - Quick Start")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -e .", "Installing package"):
        print("❌ Failed to install package")
        sys.exit(1)
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but continuing...")
    
    # Run example
    if not run_command("python example.py", "Running example"):
        print("⚠️  Example failed, but continuing...")
    
    print("\n🎉 Quick start complete!")
    print("\nNext steps:")
    print("1. Run the interactive demo: streamlit run demo/app.py")
    print("2. Train a model: python train.py --config config/bc_config.yaml")
    print("3. Explore the code in src/ directory")
    print("4. Read the README.md for detailed documentation")
    print("\n⚠️  Remember: This is for simulation only. Do not use on real robots!")
    
    # Check if Streamlit is available
    try:
        import streamlit
        print("\n🌐 Starting Streamlit demo...")
        subprocess.run("streamlit run demo/app.py", shell=True)
    except ImportError:
        print("\n📦 To run the interactive demo, install Streamlit:")
        print("pip install streamlit")
        print("Then run: streamlit run demo/app.py")


if __name__ == "__main__":
    main()
