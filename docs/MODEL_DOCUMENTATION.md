# Earth-SGNN Model Documentation

This document provides comprehensive documentation for integrating Earth-SGNN models into your project.

---

## Table of Contents

1. [Overview](#overview)
2. [Available Models](#available-models)
3. [Model Architectures](#model-architectures)
4. [Input Specifications](#input-specifications)
5. [Output Specifications](#output-specifications)
6. [Loading Pre-trained Models](#loading-pre-trained-models)
7. [Inference Examples](#inference-examples)
8. [Data Preprocessing](#data-preprocessing)
9. [Graph Construction](#graph-construction)
10. [Model Selection Guide](#model-selection-guide)

---

## Overview

Earth-SGNN provides temperature forecasting models for weather station networks. The models predict temperature at forecast horizons of 1, 6, 12, and 24 hours using historical observations and spatial relationships between stations.

### Quick Reference

| Model | Best For | RMSE | R² | Training Samples |
|-------|----------|------|-----|------------------|
| Hybrid GNN | 1h, 6h, 12h forecasts | 0.647-2.151°C | 0.889-0.990 | 4.3M |
| GNN v1 | 24h forecasts | 2.454°C | 0.880 | 6.4K |
| LSTM Baseline | Comparison/fallback | 1.293-3.679°C | 0.686-0.958 | 4.3M |

---

## Available Models

### Model Files

```
results/models/
├── hybrid_gnn_1h.pt      # Best for 1-hour forecasts
├── hybrid_gnn_6h.pt      # Best for 6-hour forecasts
├── hybrid_gnn_12h.pt     # Best for 12-hour forecasts
├── hybrid_gnn_24h.pt     # Hybrid version for 24h (use GNN v1 instead)
├── gnn_model_1h.pt       # GNN v1 - 1 hour
├── gnn_model_6h.pt       # GNN v1 - 6 hours
├── gnn_model_12h.pt      # GNN v1 - 12 hours
├── gnn_model_24h.pt      # Best for 24-hour forecasts
└── lstm_baseline_*.pt    # LSTM baselines (if saved)
```

### Graph Files

```
data/graphs/
├── weather_graph_knn_8.pt    # Primary graph (k=8)
├── weather_graph_knn_4.pt    # Sparse alternative
├── weather_graph_knn_12.pt   # Denser alternative
├── weather_graph_knn_16.pt   # Densest option
└── graph_metadata.json       # Graph statistics
```

---

## Model Architectures

### 1. Hybrid GNN (Recommended for 1h-12h)

```
HybridGNN
├── TemporalEncoder (LSTM)
│   ├── input_dim: 9 features
│   ├── hidden_dim: 64
│   ├── num_layers: 2
│   └── dropout: 0.2
├── SpatialAggregator (Attention)
│   ├── Processes k=8 nearest neighbors
│   ├── Attention-weighted aggregation
│   └── Handles missing neighbors via masking
├── Learnable Fusion
│   ├── alpha = sigmoid(learnable_weight)
│   └── output = (1-alpha) * temporal + alpha * spatial
└── OutputHead (MLP)
    └── 64 → 32 → 1
```

**Key Features:**
- Processes per-station sequences (data efficient)
- Attention-based neighbor aggregation
- Learnable temporal-spatial fusion weight
- Uses 4.3M training samples

### 2. GNN v1 (Recommended for 24h)

```
SpatioTemporalGNN
├── TemporalEncoder (LSTM)
│   ├── input_dim: 9 features
│   ├── hidden_dim: 64
│   └── num_layers: 2
├── SpatialGNN (GCN)
│   ├── hidden_dim: 64
│   ├── num_layers: 2
│   └── Full graph convolution
└── OutputHead (MLP)
    └── 64 → 32 → 1
```

**Key Features:**
- Full graph convolution (all 822 stations)
- Requires synchronized observations
- Uses 6.4K training samples
- Better for long-range forecasts

### 3. LSTM Baseline

```
LSTMModel
├── LSTM
│   ├── input_dim: 9 features
│   ├── hidden_dim: 64
│   └── num_layers: 2
└── OutputHead (Linear)
    └── 64 → 1
```

**Key Features:**
- Per-station independent predictions
- No spatial information
- Fast inference

---

## Input Specifications

### Feature Vector (9 dimensions)

| Index | Feature | Description | Normalization |
|-------|---------|-------------|---------------|
| 0 | `temperature_2m_norm` | Normalized temperature | Z-score |
| 1 | `dewpoint_2m_norm` | Normalized dewpoint | Z-score |
| 2 | `relative_humidity_2m_norm` | Normalized humidity | Z-score |
| 3 | `wind_u_norm` | Normalized U-wind component | Z-score |
| 4 | `wind_v_norm` | Normalized V-wind component | Z-score |
| 5 | `hour_sin` | sin(2π × hour/24) | [-1, 1] |
| 6 | `hour_cos` | cos(2π × hour/24) | [-1, 1] |
| 7 | `doy_sin` | sin(2π × day_of_year/365) | [-1, 1] |
| 8 | `doy_cos` | cos(2π × day_of_year/365) | [-1, 1] |

### Normalization Statistics

Use these values to normalize input features:

```python
NORMALIZATION_STATS = {
    'temperature_2m': {'mean': 7.34, 'std': 8.42},
    'dewpoint_2m': {'mean': 3.71, 'std': 7.85},
    'relative_humidity_2m': {'mean': 79.60, 'std': 15.66},
    'wind_u': {'mean': 1.04, 'std': 4.10},
    'wind_v': {'mean': 0.87, 'std': 4.08},
}
```

### Input Tensor Shapes

| Model | Input Shape | Description |
|-------|-------------|-------------|
| Hybrid GNN | `(batch, seq_len, n_features)` | Per-station sequences |
| | + `(batch, k, seq_len, n_features)` | Neighbor sequences |
| GNN v1 | `(n_stations, seq_len, n_features)` | All stations at once |
| LSTM | `(batch, seq_len, n_features)` | Per-station sequences |

**Parameters:**
- `batch`: Batch size (e.g., 256)
- `seq_len`: Lookback window = 24 hours
- `n_features`: 9 input features
- `k`: Number of neighbors = 8
- `n_stations`: 822 weather stations

---

## Output Specifications

### Output Shape

All models output a single temperature prediction:

| Model | Output Shape | Description |
|-------|--------------|-------------|
| Hybrid GNN | `(batch, 1)` | Temperature in °C |
| GNN v1 | `(n_stations, 1)` | Temperature per station |
| LSTM | `(batch, 1)` | Temperature in °C |

### Denormalization

Model outputs are in **original units (°C)** - no denormalization needed.

---

## Loading Pre-trained Models

### PyTorch Loading

```python
import torch

# Define model architecture (must match saved model)
class HybridGNN(torch.nn.Module):
    def __init__(self, n_features=9, hidden_dim=64, n_layers=2, 
                 max_neighbors=8, dropout=0.2):
        super().__init__()
        
        # Temporal encoder
        self.lstm = torch.nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Spatial attention
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1)
        )
        
        # Learnable fusion weight
        self.spatial_weight = torch.nn.Parameter(torch.tensor(0.0))
        
        # Output head
        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    
    def forward(self, center_seq, neighbor_seqs, neighbor_mask):
        # Encode center station
        center_out, _ = self.lstm(center_seq)
        center_features = center_out[:, -1, :]  # Last timestep
        
        # Encode neighbors
        batch_size, k, seq_len, n_feat = neighbor_seqs.shape
        neighbor_flat = neighbor_seqs.view(-1, seq_len, n_feat)
        neighbor_out, _ = self.lstm(neighbor_flat)
        neighbor_features = neighbor_out[:, -1, :].view(batch_size, k, -1)
        
        # Attention over neighbors
        center_expanded = center_features.unsqueeze(1).expand(-1, k, -1)
        combined = torch.cat([center_expanded, neighbor_features], dim=-1)
        attention_scores = self.attention(combined).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~neighbor_mask, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        spatial_features = (attention_weights.unsqueeze(-1) * neighbor_features).sum(dim=1)
        
        # Fusion
        alpha = torch.sigmoid(self.spatial_weight)
        fused = (1 - alpha) * center_features + alpha * spatial_features
        
        return self.output(fused)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridGNN()
model.load_state_dict(torch.load('results/models/hybrid_gnn_1h.pt', map_location=device))
model.eval()
```

### Quick Load with Checkpoint

```python
# If models are saved as complete checkpoints
checkpoint = torch.load('results/models/hybrid_gnn_1h.pt')
model.load_state_dict(checkpoint['model_state_dict'])
# Additional info may include:
# - checkpoint['optimizer_state_dict']
# - checkpoint['epoch']
# - checkpoint['val_loss']
```

---

## Inference Examples

### Single Station Prediction (Hybrid GNN)

```python
import torch
import numpy as np

def predict_temperature(model, center_sequence, neighbor_sequences, 
                        neighbor_mask, device='cuda'):
    """
    Predict temperature for a single station.
    
    Args:
        model: Loaded HybridGNN model
        center_sequence: np.array of shape (24, 9) - 24 hours of features
        neighbor_sequences: np.array of shape (8, 24, 9) - 8 neighbors
        neighbor_mask: np.array of shape (8,) - True where neighbor exists
        device: 'cuda' or 'cpu'
    
    Returns:
        float: Predicted temperature in °C
    """
    model.eval()
    with torch.no_grad():
        # Add batch dimension
        center = torch.tensor(center_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        neighbors = torch.tensor(neighbor_sequences, dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.tensor(neighbor_mask, dtype=torch.bool).unsqueeze(0).to(device)
        
        prediction = model(center, neighbors, mask)
        return prediction.item()

# Example usage
center_seq = np.random.randn(24, 9)  # Replace with real data
neighbor_seqs = np.random.randn(8, 24, 9)
neighbor_mask = np.array([True, True, True, True, True, True, False, False])

temp_prediction = predict_temperature(model, center_seq, neighbor_seqs, neighbor_mask)
print(f"Predicted temperature: {temp_prediction:.2f}°C")
```

### Batch Prediction

```python
def predict_batch(model, center_batch, neighbor_batch, mask_batch, device='cuda'):
    """
    Predict temperature for a batch of stations.
    
    Args:
        center_batch: torch.Tensor of shape (batch, 24, 9)
        neighbor_batch: torch.Tensor of shape (batch, 8, 24, 9)
        mask_batch: torch.Tensor of shape (batch, 8)
    
    Returns:
        torch.Tensor: Predictions of shape (batch, 1)
    """
    model.eval()
    with torch.no_grad():
        predictions = model(
            center_batch.to(device),
            neighbor_batch.to(device),
            mask_batch.to(device)
        )
    return predictions.cpu()

# Example
batch_size = 32
predictions = predict_batch(
    model,
    torch.randn(batch_size, 24, 9),
    torch.randn(batch_size, 8, 24, 9),
    torch.ones(batch_size, 8, dtype=torch.bool)
)
```

### Multi-Horizon Prediction

```python
def predict_all_horizons(models_dict, center_seq, neighbor_seqs, neighbor_mask):
    """
    Predict temperature at all horizons using optimal models.
    
    Args:
        models_dict: Dict with keys '1h', '6h', '12h', '24h'
        center_seq, neighbor_seqs, neighbor_mask: Input data
    
    Returns:
        dict: Predictions for each horizon
    """
    predictions = {}
    
    for horizon in ['1h', '6h', '12h']:
        # Use Hybrid GNN for 1h-12h
        model = models_dict[f'hybrid_{horizon}']
        predictions[horizon] = predict_temperature(
            model, center_seq, neighbor_seqs, neighbor_mask
        )
    
    # Use GNN v1 for 24h (requires different input format)
    # ... handle separately based on GNN v1 requirements
    
    return predictions
```

---

## Data Preprocessing

### Step 1: Load and Normalize Data

```python
import pandas as pd
import numpy as np

# Load your data
df = pd.read_parquet('your_weather_data.parquet')

# Normalization function
def normalize_features(df, stats=NORMALIZATION_STATS):
    """Normalize meteorological features."""
    df_norm = df.copy()
    
    for var, params in stats.items():
        if var in df.columns:
            df_norm[f'{var}_norm'] = (df[var] - params['mean']) / params['std']
    
    return df_norm

# Add cyclical time features
def add_time_features(df):
    """Add cyclical encoding for time."""
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    return df

# Add wind components
def add_wind_components(df):
    """Convert wind speed/direction to U/V components."""
    wind_rad = np.radians(df['wind_direction_10m'])
    df['wind_u'] = -df['wind_speed_10m'] * np.sin(wind_rad)
    df['wind_v'] = -df['wind_speed_10m'] * np.cos(wind_rad)
    return df
```

### Step 2: Create Sequences

```python
def create_sequences(data, lookback=24):
    """
    Create input sequences for model.
    
    Args:
        data: np.array of shape (time, features)
        lookback: Number of historical hours (default: 24)
    
    Returns:
        np.array of shape (n_samples, lookback, features)
    """
    sequences = []
    for i in range(lookback, len(data)):
        seq = data[i-lookback:i]
        sequences.append(seq)
    return np.array(sequences)

# Feature columns (must be in this order)
FEATURE_COLS = [
    'temperature_2m_norm',
    'dewpoint_2m_norm', 
    'relative_humidity_2m_norm',
    'wind_u_norm',
    'wind_v_norm',
    'hour_sin',
    'hour_cos',
    'doy_sin',
    'doy_cos'
]
```

---

## Graph Construction

### Load Existing Graph

```python
import torch

# Load graph
graph = torch.load('data/graphs/weather_graph_knn_8.pt')

# Graph structure
print(f"Nodes: {graph.num_nodes}")        # 822
print(f"Edges: {graph.edge_index.shape}")  # [2, 7842]
print(f"Node features: {graph.x.shape}")   # [822, 3] (lat, lon, elev)
print(f"Edge weights: {graph.edge_weight.shape}")  # [7842, 1]
```

### Get Neighbors for a Station

```python
def get_neighbors(graph, station_idx, k=8):
    """
    Get k nearest neighbors for a station.
    
    Args:
        graph: PyG Data object
        station_idx: Index of target station
        k: Number of neighbors
    
    Returns:
        list: Indices of neighbor stations
    """
    edge_index = graph.edge_index
    
    # Find edges where source is station_idx
    mask = edge_index[0] == station_idx
    neighbors = edge_index[1][mask].tolist()
    
    return neighbors[:k]

# Example
neighbors = get_neighbors(graph, station_idx=0)
print(f"Station 0 neighbors: {neighbors}")
```

### Create Custom Graph

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

def create_knn_graph(station_coords, k=8, sigma=100):
    """
    Create k-NN graph from station coordinates.
    
    Args:
        station_coords: np.array of shape (n_stations, 2) - [lat, lon]
        k: Number of neighbors
        sigma: Gaussian kernel sigma (km)
    
    Returns:
        edge_index: torch.Tensor of shape [2, n_edges]
        edge_weight: torch.Tensor of shape [n_edges]
    """
    from geopy.distance import geodesic
    
    n_stations = len(station_coords)
    
    # Compute distance matrix
    distances = np.zeros((n_stations, n_stations))
    for i in range(n_stations):
        for j in range(n_stations):
            distances[i, j] = geodesic(station_coords[i], station_coords[j]).km
    
    # k-NN
    edges_src, edges_dst, weights = [], [], []
    for i in range(n_stations):
        nearest = np.argsort(distances[i])[1:k+1]  # Exclude self
        for j in nearest:
            edges_src.append(i)
            edges_dst.append(j)
            # Gaussian kernel weight
            weights.append(np.exp(-distances[i, j]**2 / (2 * sigma**2)))
    
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    
    return edge_index, edge_weight
```

---

## Model Selection Guide

### Decision Tree

```
Which forecast horizon?
│
├── 1 hour  → Use Hybrid GNN (hybrid_gnn_1h.pt)
│             RMSE: 0.647°C, R²: 0.990
│
├── 6 hours → Use Hybrid GNN (hybrid_gnn_6h.pt)
│             RMSE: 1.464°C, R²: 0.948
│
├── 12 hours → Use Hybrid GNN (hybrid_gnn_12h.pt)
│              RMSE: 2.151°C, R²: 0.889
│
└── 24 hours → Use GNN v1 (gnn_model_24h.pt)
               RMSE: 2.454°C, R²: 0.880
```

### Production Ensemble

For best performance across all horizons, use an ensemble:

```python
class WeatherEnsemble:
    def __init__(self, model_dir='results/models/'):
        self.models = {
            '1h': load_hybrid_gnn(f'{model_dir}/hybrid_gnn_1h.pt'),
            '6h': load_hybrid_gnn(f'{model_dir}/hybrid_gnn_6h.pt'),
            '12h': load_hybrid_gnn(f'{model_dir}/hybrid_gnn_12h.pt'),
            '24h': load_gnn_v1(f'{model_dir}/gnn_model_24h.pt'),
        }
    
    def predict(self, data, horizon):
        """Automatically select best model for horizon."""
        model = self.models[horizon]
        return model(data)
```

### Performance Comparison

| Horizon | Hybrid GNN | GNN v1 | LSTM | Persistence | Best |
|---------|-----------|--------|------|-------------|------|
| 1h | **0.647** | 2.163 | 1.293 | 0.772 | Hybrid |
| 6h | **1.464** | 2.312 | 1.541 | 1.971 | Hybrid |
| 12h | **2.151** | 2.326 | 2.292 | 2.697 | Hybrid |
| 24h | 3.085 | **2.454** | 3.679 | 3.534 | GNN v1 |

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use `torch.cuda.empty_cache()`
   - Move to CPU for inference: `model.cpu()`

2. **Missing Neighbors**
   - Ensure `neighbor_mask` correctly marks available neighbors
   - Pad missing neighbors with zeros

3. **NaN Predictions**
   - Check for NaN in input features
   - Ensure normalization statistics match training data

4. **Shape Mismatch**
   - Verify input dimensions match expected shapes
   - Check sequence length is exactly 24

### Minimum Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric 2.0+ (for graph operations)
- CUDA 11.0+ (for GPU acceleration)
- 4GB GPU memory (8GB recommended)

---

## Citation

If you use these models in your research, please cite:

```bibtex
@misc{earthsgnn2026,
  title={Earth-SGNN: Spatio-Temporal Graph Neural Networks for Weather Forecasting},
  author={Earth-SGNN Contributors},
  year={2026},
  url={https://github.com/username/earth-sgnn}
}
```

---

## License

MIT License - See LICENSE file for details.
