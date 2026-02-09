# Next Steps: Further Improvements Plan

## Current State (Achieved)

| Horizon | Best Model | RMSE | Status |
|---------|-----------|------|--------|
| 1h | Hybrid GNN | 0.647°C | Beats all baselines |
| 6h | Hybrid GNN | 1.464°C | Beats LSTM by 5% |
| 12h | Hybrid GNN | 2.151°C | Beats LSTM by 6% |
| 24h | GNN v1 | 2.454°C | Beats LSTM by 33% |

---

## Improvement Options

### Option A: Production-Ready Ensemble (Priority: HIGH)

**Goal:** Create a single deployable model that uses best approach per horizon.

**Implementation:**
```python
class EnsembleForecaster:
    def __init__(self):
        self.hybrid_1h = load_model('hybrid_gnn_1h.pt')
        self.hybrid_6h = load_model('hybrid_gnn_6h.pt')
        self.hybrid_12h = load_model('hybrid_gnn_12h.pt')
        self.gnn_24h = load_model('gnn_model_24h.pt')
    
    def forecast(self, data, horizon):
        if horizon <= 12:
            return self.hybrid_models[horizon](data)
        else:
            return self.gnn_24h(data)
```

**Deliverables:**
- `src/models/ensemble.py` - Unified ensemble class
- `notebooks/06_ensemble/01_ensemble_model.ipynb` - Ensemble evaluation
- Inference API endpoint

**Effort:** 1 day

---

### Option B: Multi-Variable Forecasting (Priority: MEDIUM)

**Goal:** Extend beyond temperature to predict multiple weather variables simultaneously.

**Variables to add:**
| Variable | Missing % | Priority |
|----------|-----------|----------|
| dewpoint_2m | 3.3% | High |
| wind_u, wind_v | 6% | Medium |
| relative_humidity | 3.4% | Medium |
| surface_pressure | 56% | Low (too sparse) |

**Implementation:**
- Multi-output head: predict all variables at once
- Shared encoder, variable-specific decoders
- Joint loss function with variable weighting

**Deliverables:**
- `notebooks/07_multi_variable/01_multi_output_gnn.ipynb`
- Extended evaluation metrics

**Effort:** 2-3 days

---

### Option C: Improve 24h Performance (Priority: MEDIUM)

**Problem:** Hybrid GNN (3.085°C) underperforms GNN v1 (2.454°C) at 24h.

**Potential Solutions:**

1. **Hybrid with more spatial weight for 24h:**
   - Initialize spatial_weight = 0.6 instead of 0.4
   - Or use separate model architecture for 24h

2. **Two-stage approach:**
   - Stage 1: Hybrid GNN for temporal features
   - Stage 2: Full GNN for spatial propagation

3. **Increase neighbor count:**
   - Use k=16 neighbors for 24h (vs k=8)
   - More spatial context for long-range forecasts

4. **Deeper temporal encoder:**
   - 3-4 LSTM layers for 24h
   - Capture longer-term patterns

**Effort:** 1-2 days

---

### Option D: Graph Attention Network (Priority: MEDIUM)

**Goal:** Replace static GCN with learnable attention over neighbors.

**Benefits:**
- Learn which neighbors matter dynamically
- Handle varying station densities better
- Potentially better for heterogeneous regions

**Implementation:**
```python
from torch_geometric.nn import GATConv

class SpatialGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4):
        self.gat1 = GATConv(in_dim, hidden_dim // heads, heads=heads)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1)
```

**Effort:** 1 day

---

### Option E: Temporal Attention (Transformer) (Priority: LOW)

**Goal:** Replace LSTM with Transformer for temporal encoding.

**Benefits:**
- Better at capturing long-range dependencies
- Parallelizable (faster training)
- State-of-the-art for sequence modeling

**Challenges:**
- More parameters, may need more data
- Positional encoding design for time series

**Effort:** 2-3 days

---

### Option F: Real-Time Deployment (Priority: LOW)

**Goal:** Deploy model for real-time forecasting.

**Components:**
1. **Data pipeline:** Fetch live NOAA data
2. **Inference service:** FastAPI endpoint
3. **Visualization:** Dashboard with forecasts
4. **Monitoring:** Track prediction accuracy

**Effort:** 3-5 days

---

## Recommended Priority Order

| Priority | Option | Effort | Impact |
|----------|--------|--------|--------|
| 1 | **A: Ensemble** | 1 day | High - production ready |
| 2 | **C: Improve 24h** | 1-2 days | Medium - close performance gap |
| 3 | **D: GAT** | 1 day | Medium - modern architecture |
| 4 | **B: Multi-variable** | 2-3 days | Medium - broader applicability |
| 5 | **E: Transformer** | 2-3 days | Low - research exploration |
| 6 | **F: Deployment** | 3-5 days | Low - if production needed |

---

## Quick Wins (Can do immediately)

1. **Save best models with metadata** - Document exact configuration
2. **Create inference script** - Simple prediction function
3. **Visualization dashboard** - Plot forecasts vs actuals
4. **Update README** - Document final results

---

## Decision Required

Which improvement path would you like to pursue?

- [ ] **A:** Production ensemble (recommended)
- [ ] **B:** Multi-variable forecasting
- [ ] **C:** Improve 24h horizon
- [ ] **D:** Graph Attention Network
- [ ] **E:** Transformer temporal encoder
- [ ] **F:** Real-time deployment
- [ ] **Done:** Project complete, documentation only

---

*Created: 2026-02-05*
