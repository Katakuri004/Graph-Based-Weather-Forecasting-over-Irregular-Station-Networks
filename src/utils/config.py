"""
Configuration settings for the project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GRAPHS_DIR = DATA_DIR / "graphs"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
FINAL_MODELS_DIR = MODELS_DIR / "final"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
EVALUATIONS_DIR = RESULTS_DIR / "evaluations"

# Create directories if they don't exist
for dir_path in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, GRAPHS_DIR,
    CHECKPOINTS_DIR, FINAL_MODELS_DIR,
    FIGURES_DIR, TABLES_DIR, EVALUATIONS_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Weather variables to forecast
WEATHER_VARIABLES = [
    'temperature_2m',
    'relative_humidity_2m',
    'wind_speed_10m',
    'wind_direction_10m',
    'surface_pressure'
]

# Forecast lead times (hours)
LEAD_TIMES = [1, 6, 12, 24]

# Random seed for reproducibility
RANDOM_SEED = 42

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
