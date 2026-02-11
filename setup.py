"""Setup script for the imitation learning package."""

from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="imitation-learning-robotics",
    version="1.0.0",
    author="AI Assistant",
    author_email="ai@example.com",
    description="Modern imitation learning framework for robotics with behavioral cloning, DAgger, and GAIL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/imitation-learning-robotics",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pre-commit>=3.3.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ],
        "ros2": [
            "rclpy>=3.3.0",
            "tf2-ros>=0.25.0",
            "geometry-msgs>=4.2.0",
            "sensor-msgs>=4.2.0",
            "nav-msgs>=4.2.0",
        ],
        "simulation": [
            "pybullet>=3.2.0",
            "mujoco>=2.3.0",
            "gymnasium-robotics>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "imitation-learning-train=train:main",
            "imitation-learning-demo=demo.app:main",
        ],
    },
)
