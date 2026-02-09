# Graph-Based Weather Forecasting over Irregular Station Networks

Research project for forecasting weather variables over irregular station networks using Graph Neural Networks (GNNs).

## Project Structure

```
earth-sgnn/
├── notebooks/              # Jupyter notebooks organized by phase
│   ├── 01_data_acquisition/
│   ├── 02_data_preprocessing/
│   ├── 03_baselines/
│   ├── 04_gnn_models/
│   ├── 05_training/
│   ├── 06_analysis/
│   ├── 07_evaluation/
│   └── 08_documentation/
├── data/                   # Data storage
│   ├── raw/               # Raw downloaded data
│   ├── processed/         # Preprocessed data
│   └── graphs/            # Graph structure files
├── models/                 # Model storage
│   ├── checkpoints/       # Training checkpoints
│   └── final/             # Final trained models
├── results/                # Results and outputs
│   ├── figures/           # Visualization figures
│   ├── tables/            # Result tables
│   └── evaluations/       # Evaluation metrics
├── src/                    # Source code modules
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures
│   └── utils/             # Utility functions
├── requirements.txt        # Python dependencies
├── IMPLEMENTATION_PLAN.md  # Detailed implementation plan
├── REBUTTALS_AND_ALTERNATIVES.md  # Methodology defense
└── PROJECT_SUMMARY.md      # Quick reference guide
```

## Setup

### Virtual Environment Already Created ✅

The virtual environment `venv-earth-sgnn` has been created and all dependencies are installed.

### Activate the Environment

**Windows PowerShell:**
```powershell
.\venv-earth-sgnn\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
venv-earth-sgnn\Scripts\activate.bat
```

### Verify Installation

Run the verification script:
```bash
python verify_installation.py
```

Or manually check:
```python
python -c "import torch; import pandas; import torch_geometric; print('All packages OK!')"
```

### Start Jupyter Lab

Once the environment is activated:
```bash
jupyter lab
```

## Data Sources

### Primary: NOAA ISD
- Integrated Surface Database
- Hourly surface observations from global land stations
- Variables: temperature, humidity, wind, pressure

### Development: Open-Meteo
- Historical Weather API
- Used for rapid prototyping and validation

### Reference: ERA5
- ECMWF reanalysis data
- Used for baseline comparisons

## Getting Started

1. Start with `notebooks/01_data_acquisition/` for data setup
2. Follow the notebooks in numerical order
3. See `IMPLEMENTATION_PLAN.md` for detailed guidance

## Key Features

- **Graph-based approach**: Models weather stations as nodes in a graph
- **Spatio-temporal modeling**: Captures both spatial and temporal dependencies
- **Operational focus**: Designed for real-world station networks
- **Comprehensive evaluation**: Multiple baselines and metrics

## Documentation

- `IMPLEMENTATION_PLAN.md`: Complete 16-week implementation plan
- `REBUTTALS_AND_ALTERNATIVES.md`: Defense of methodology
- `PROJECT_SUMMARY.md`: Quick reference guide

## License

Research project - see LICENSE file for details.
