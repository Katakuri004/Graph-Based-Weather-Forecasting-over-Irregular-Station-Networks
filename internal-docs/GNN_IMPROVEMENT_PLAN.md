# GNN Improvement Plan: Efficient Spatio-Temporal Weather Forecasting

## Executive Summary

The current GNN implementation shows **33% improvement at 24h horizon** but **underperforms at shorter horizons** due to data efficiency issues. This plan outlines strategies to improve GNN performance across all forecast horizons.

---

## Problem Analysis

### Current Performance
| Horizon | LSTM RMSE | GNN RMSE | Gap |
|---------|-----------|----------|-----|
| 1h | 1.29°C | 2.16°C | +67% worse |
| 6h | 1.54°C | 2.31°C | +50% worse |
| 12h | 2.29°C | 2.33°C | ~equal |
| 24h | 3.68°C | 2.45°C | **33% better** |

### Root Causes

| Issue | Current State | Impact |
|-------|---------------|--------|
| **Data Efficiency** | 6K samples (0.15% of LSTM's 4M) | Severe underfitting |
| **Missing Data** | Drop timesteps with any missing station | 99.85% data loss |
| **Temporal Priority** | GNN processes all stations equally | Short-term = local patterns |
| **Graph Static** | Same k=8 for all horizons | Suboptimal topology |

---

## Implementation Plan

### Phase 1: Efficient Data Loading (Priority: HIGH)

**Goal:** Increase training samples from 6K to ~500K+

#### Strategy 1.1: Station Subset Sampling
Instead of requiring ALL 822 stations, sample random subsets of stations per batch.

```python
class EfficientSpatioTemporalDataset:
    """
    Sample station subsets instead of requiring full coverage.
    
    Key changes:
    - Sample N stations per batch (e.g., 100-200)
    - Build dynamic subgraph for sampled stations
    - Handle missing values with masking, not dropping
    """
    def __init__(self, df, graph, n_sample_stations=150):
        self.n_sample_stations = n_sample_stations
        # Index data by timestamp for fast lookup
        self.time_index = df.groupby('timestamp').groups
        
    def __getitem__(self, idx):
        # Get timestamp
        timestamp = self.valid_times[idx]
        
        # Sample stations that have data at this time
        available_stations = self.get_available_stations(timestamp)
        sampled = np.random.choice(available_stations, 
                                   min(self.n_sample_stations, len(available_stations)))
        
        # Build subgraph for sampled stations
        subgraph = self.extract_subgraph(sampled)
        
        return x, y, subgraph, mask
```

**Expected Impact:** 10-50x more training samples

#### Strategy 1.2: Masked Loss Training
Train on available stations, mask missing ones in loss computation.

```python
def masked_mse_loss(pred, target, mask):
    """Compute MSE only on valid (non-missing) stations."""
    valid_pred = pred[mask]
    valid_target = target[mask]
    return F.mse_loss(valid_pred, valid_target)
```

**Expected Impact:** Use ALL timesteps, not just complete ones

---

### Phase 2: Hybrid Architecture (Priority: HIGH)

**Goal:** Combine LSTM's temporal strength with GNN's spatial strength

#### Strategy 2.1: Pre-trained Temporal Encoder
1. Pre-train LSTM on per-station data (4M samples)
2. Freeze or fine-tune temporal encoder
3. Add spatial GNN on top of learned temporal features

```python
class HybridModel(nn.Module):
    def __init__(self, pretrained_lstm_path):
        # Load pre-trained per-station LSTM
        self.temporal_encoder = load_pretrained_lstm(pretrained_lstm_path)
        self.temporal_encoder.requires_grad_(False)  # Freeze initially
        
        # Spatial aggregation
        self.spatial_gnn = SpatialGNN(...)
        
        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(temporal_dim + spatial_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
```

**Expected Impact:** Leverage 4M samples for temporal learning

#### Strategy 2.2: Horizon-Adaptive Model Selection
Use different models for different horizons:

| Horizon | Primary Model | Spatial Weight |
|---------|--------------|----------------|
| 1h | LSTM (frozen) | 0.1 |
| 6h | LSTM + light GNN | 0.3 |
| 12h | LSTM + GNN | 0.5 |
| 24h | Full GNN | 0.8 |

```python
class AdaptiveHybrid(nn.Module):
    def forward(self, x, edge_index, horizon):
        temporal_out = self.temporal_encoder(x)
        spatial_out = self.spatial_gnn(temporal_out, edge_index)
        
        # Horizon-dependent fusion
        alpha = self.horizon_weights[horizon]  # Learned or fixed
        return (1 - alpha) * temporal_out + alpha * spatial_out
```

---

### Phase 3: Advanced GNN Architecture (Priority: MEDIUM)

#### Strategy 3.1: Graph Attention Network (GAT)
Replace GCN with attention to learn which neighbors matter.

```python
from torch_geometric.nn import GATConv

class SpatialGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4):
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
```

**Benefits:**
- Learns edge importance dynamically
- Can ignore irrelevant neighbors
- Better for heterogeneous station density

#### Strategy 3.2: Multi-Scale Graph
Use different graph topologies for different information scales:

```python
class MultiScaleGNN(nn.Module):
    def __init__(self):
        # Local: k=4 (nearby stations, ~65km mean distance)
        self.local_gnn = GCNConv(...)
        
        # Regional: k=8 (medium range, ~90km)
        self.regional_gnn = GCNConv(...)
        
        # Long-range: k=16 (weather systems, ~127km)
        self.longrange_gnn = GCNConv(...)
        
    def forward(self, x, graphs):
        local = self.local_gnn(x, graphs['k4'])
        regional = self.regional_gnn(x, graphs['k8'])
        longrange = self.longrange_gnn(x, graphs['k16'])
        return torch.cat([local, regional, longrange], dim=-1)
```

#### Strategy 3.3: Temporal Graph Networks
Process time as an explicit dimension in the graph:

```python
class TemporalGNN(nn.Module):
    """
    Unfold time into the graph structure.
    Connect each station to:
    - Its spatial neighbors at same timestep
    - Itself at previous timesteps (temporal edges)
    """
    def build_temporal_graph(self, spatial_edges, n_timesteps):
        # Spatial edges (within each timestep)
        # Temporal edges (across timesteps for same station)
        pass
```

---

### Phase 4: Training Optimizations (Priority: MEDIUM)

#### Strategy 4.1: Curriculum Learning
Start with easy examples, progress to harder ones:

```python
# Epoch 1-5: Train on 1h horizon (easy)
# Epoch 6-10: Mix 1h and 6h
# Epoch 11-15: Mix all horizons
# Epoch 16-20: Focus on 12h and 24h (hard)
```

#### Strategy 4.2: Multi-Task Learning
Predict all horizons simultaneously:

```python
class MultiHorizonGNN(nn.Module):
    def __init__(self):
        self.shared_encoder = SpatioTemporalEncoder(...)
        self.heads = nn.ModuleDict({
            '1h': nn.Linear(hidden, 1),
            '6h': nn.Linear(hidden, 1),
            '12h': nn.Linear(hidden, 1),
            '24h': nn.Linear(hidden, 1)
        })
    
    def forward(self, x, edge_index):
        features = self.shared_encoder(x, edge_index)
        return {h: head(features) for h, head in self.heads.items()}
```

**Benefits:**
- Shared representation learning
- Regularization effect
- Single model for all horizons

#### Strategy 4.3: Gradient Accumulation
Simulate larger batches on limited GPU memory:

```python
accumulation_steps = 4
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### Phase 5: Graph Construction Improvements (Priority: LOW)

#### Strategy 5.1: Learned Edge Weights
Let the model learn optimal edge weights:

```python
class LearnableGraph(nn.Module):
    def __init__(self, n_nodes, n_edges):
        # Initialize with distance-based weights
        self.edge_weight = nn.Parameter(initial_weights)
        
    def forward(self):
        # Softmax to ensure valid weights
        return F.softmax(self.edge_weight, dim=0)
```

#### Strategy 5.2: Dynamic Graph Based on Weather Patterns
Adjust connections based on current weather state:

```python
def dynamic_graph(node_features, base_graph, threshold=0.5):
    """
    Modify edges based on feature similarity.
    Connect stations with similar weather patterns.
    """
    similarity = cosine_similarity(node_features)
    dynamic_edges = (similarity > threshold).nonzero()
    return merge_graphs(base_graph, dynamic_edges)
```

---

## Implementation Priority

| Phase | Strategy | Effort | Expected Impact | Priority |
|-------|----------|--------|-----------------|----------|
| 1 | Station Subset Sampling | Medium | Very High | **P0** |
| 1 | Masked Loss Training | Low | High | **P0** |
| 2 | Pre-trained Temporal Encoder | Medium | Very High | **P0** |
| 2 | Horizon-Adaptive Selection | Low | High | **P1** |
| 3 | GAT instead of GCN | Low | Medium | **P1** |
| 3 | Multi-Scale Graph | Medium | Medium | **P2** |
| 4 | Multi-Task Learning | Medium | High | **P1** |
| 4 | Curriculum Learning | Low | Medium | **P2** |
| 5 | Learned Edge Weights | Medium | Low | **P3** |

---

## Recommended Implementation Order

### Sprint 1: Data Efficiency (1-2 days)
1. Implement `EfficientSpatioTemporalDataset` with station sampling
2. Add masked loss function
3. Verify training samples increase to 100K+
4. Re-run baseline GNN with more data

### Sprint 2: Hybrid Architecture (1-2 days)
1. Save pre-trained LSTM models from baseline notebook
2. Implement `HybridModel` with frozen temporal encoder
3. Train and compare against pure GNN
4. Implement horizon-adaptive fusion

### Sprint 3: Advanced GNN (1 day)
1. Replace GCN with GAT
2. Implement multi-scale graph option
3. A/B test against best hybrid model

### Sprint 4: Final Optimization (1 day)
1. Implement multi-task learning
2. Hyperparameter tuning
3. Final evaluation and documentation

---

## Success Metrics

| Horizon | Current GNN | Target | Stretch Goal |
|---------|-------------|--------|--------------|
| 1h | 2.16°C | < 1.29°C (beat LSTM) | < 1.0°C |
| 6h | 2.31°C | < 1.54°C (beat LSTM) | < 1.3°C |
| 12h | 2.33°C | < 2.29°C (beat LSTM) | < 2.0°C |
| 24h | 2.45°C | < 2.45°C (maintain) | < 2.2°C |

---

## Files to Create

```
notebooks/05_gnn_model/
├── 01_gnn_weather_forecasting.ipynb  (existing - baseline GNN)
├── 02_efficient_gnn.ipynb            (Phase 1: data efficiency)
├── 03_hybrid_model.ipynb             (Phase 2: hybrid architecture)
└── 04_advanced_gnn.ipynb             (Phase 3: GAT, multi-scale)

src/models/
├── __init__.py
├── temporal_encoder.py
├── spatial_gnn.py
├── hybrid_model.py
└── efficient_dataset.py
```

---

*Created: 2026-02-05*
