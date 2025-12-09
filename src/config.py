"""
Configuration management for the robotic AI system.
"""

import os
import yaml
import random
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_PATH = PROJECT_ROOT / "data"
ASSETS_PATH = PROJECT_ROOT / "assets"


def load_config() -> dict:
    """Load configuration from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    return get_default_config()


def get_default_config() -> dict:
    """Return default configuration."""
    return {
        "simulation": {
            "seed": 42,
            "gui": True,
            "time_step": 1.0 / 240.0,
        },
        "robot": {
            "model": "kuka_iiwa",
            "end_effector_link": 6,
        },
        "camera": {
            "width": 640,
            "height": 480,
            "fov": 60,
            "near": 0.1,
            "far": 10.0,
            "position": [0.5, 0.0, 1.2],
            "target": [0.5, 0.0, 0.2],
        },
        "vision": {
            "rust_hsv_lower": [0, 100, 100],
            "rust_hsv_upper": [10, 255, 255],
            "crack_hsv_lower": [0, 0, 0],
            "crack_hsv_upper": [180, 255, 50],
            "dent_hsv_lower": [100, 100, 100],
            "dent_hsv_upper": [130, 255, 255],
            "min_contour_area": 100,
            "morphology_kernel_size": 5,
        },
        "agent": {
            "provider": "ollama",  # "ollama" or "openai"
            "model": os.getenv("OLLAMA_MODEL", "qwen3:14b"),
            "timeout": 30,
            "max_retries": 3,
        },
        "safety": {
            "collision_distance": 0.01,
            "singularity_threshold": 100,
            "workspace_bounds": {
                "x": [0.2, 0.8],
                "y": [-0.4, 0.4],
                "z": [0.05, 0.6],
            },
        },
        "path": {
            "max_velocity": 0.1,
            "max_acceleration": 0.05,
            "collision_check_step": 5,
        },
    }


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


# Load config on import
config = load_config()
set_seeds(config.get("simulation", {}).get("seed", 42))
