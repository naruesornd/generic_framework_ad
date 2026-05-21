# Hybrid Physics-ML Anomaly Detection for Reverse Osmosis Systems

A generic framework for multivariate anomaly detection and root cause analysis in industrial process plants, combining physics-based modelling with deep learning. Developed as part of ECE498 Group Project at Zhejiang University International Campus.

---

## Overview

This framework targets anomaly detection in Reverse Osmosis (RO) water treatment plants. It addresses the limitations of purely data-driven approaches by fusing first-principles physics equations with an LSTM neural network, enabling interpretable and reliable fault detection without requiring labeled anomaly data.

The framework operates through a dual-path detection architecture:
- **SADS (Slow Anomaly Detection System)** — detects gradual long-term deviations using a hybrid Physics + LSTM model
- **FADS (Fast Anomaly Detection System)** — detects sudden real-time deviations using a gradient-based multivariate linear regression on rate-of-change features (dy/dt)

Both systems feed into an automated **Root Cause Analysis (RCA)** module that ranks contributing sensor features by Z-score deviation at each anomaly timestamp.

---

## Project Structure

```
RO/
├── src/
│   ├── data_processor/
│   │   ├── data_processor.py         # Data loading, pivoting, column normalization
│   │   └── cycle_processor.py        # Operational cycle detection and segmentation
│   ├── data_loader/
│   │   ├── data_loader.py            # PyTorch DataLoader wrappers
│   │   └── data_set.py               # TimeSeriesDataset definitions
│   ├── feature_engineering/
│   │   └── feature_engineering.py   # Cross-features, lag, rolling stats, physics features
│   ├── model/
│   │   ├── coarse_feature_selection/
│   │   │   └── cfs.py               # Stage 1+2: Random Forest + SHAP feature ranking
│   │   ├── fine_feature_selection/
│   │   │   └── ffs.py               # Stage 3: LSTM ablation-based feature validation
│   │   ├── mylstm.py                # LSTM architecture and training loop
│   │   ├── fine_tune/
│   │   │   └── fine_tune.py
│   │   └── load_model/
│   │       └── load_model.py
│   └── utils/
│       ├── IQR/iqr.py               # Outlier detection
│       ├── plot/                    # Visualization utilities
│       └── simulate/               # Plant simulation utilities
├── notebook/
│   ├── 01_exploration.ipynb         # Data EDA and cycle analysis
│   ├── 02_framework.ipynb           # Full pipeline execution
│   └── 03_results.ipynb             # Results, evaluation, paper figures
├── data/
│   ├── raw/                         # Raw plant Excel/CSV files
│   └── physics/                     # Saved feature selection outputs
├── requirements.txt
└── README.md
```

---

## Framework Pipeline

```
Dataset from Plant
      │
      ▼
1. Data Preprocessing          — pivot, normalize column names, drop NaN rows
      │
      ▼
2. Input/Output Segregation    — separate controllable inputs from monitored outputs
      │
      ▼
3. Temporal Cyclic Analysis    — detect operational cycles via FeedFlow change-point detection
      │
      ▼
4. Feature Engineering         — cross-features, cycle-aware lag features, rolling stats,
                                  cyclical time encoding (sin/cos), physics-derived features
      │
      ▼
5. Feature Selection           — 3-stage: RF Gini → SHAP → LSTM ablation (per target)
      │
      ├──────────────────┬──────────────────┐
      ▼                  ▼
6. SADS                 7. FADS
   Physics Model           Derivative-Based Model
   + LSTM Residuals         (dy/dt features, Linear Regression + 3σ threshold)
      │                  │
      └──────────────────┘
                │
                ▼
8. Anomaly Detected?
        │
        Yes → Root Cause Analysis (Z-score ranking of features)
              → Probable Cause Feature + Interactive Dashboard
```

---

## Monitored Targets

| Variable | Physics Model Used |
|---|---|
| Differential Pressure | Viscosity-corrected friction model: `dP = k × exp(-0.024(T-25)) × Q^β` |
| Permeate Flow | Solution-Diffusion: `Q = A × TCF × NDP` |
| Permeate Conductivity | Salt diffusion: `C_perm = (B × C_feed × TCF) / Q` |
| Permeate Pressure | Linear regression on permeate flow (hydraulic) |

All physics model parameters (A, B, k) are auto-calibrated from the first 500 rows of clean baseline data — no labeled anomaly data is required.

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `scikit-learn`, `shap`, `pandas`, `numpy`, `plotly`, `matplotlib`

Python 3.10+ recommended.

---

## Usage

### Running the full pipeline

Open and run `notebook/02_framework.ipynb` in order. The notebook covers:
1. Data loading and preprocessing
2. Cycle segmentation
3. Feature engineering
4. Feature selection (RF + SHAP + LSTM ablation)
5. Physics model calibration and residual computation
6. SADS and FADS anomaly detection
7. Root cause analysis and interactive visualization

### Using individual modules

```python
from src.data_processor.data_processor import DataProcessor
from src.data_processor.cycle_processor import CycleProcessor
from src.feature_engineering.feature_engineering import FeatureEngineering, PhysicsBasedFeatures

# Load and preprocess
dp = DataProcessor("data/raw/plant_data.xlsx")
dp.change_pivot('timestamp', 'param_name', 'value')
dp.drop_NA_with_feature(['FeedFlow', 'FeedTemperature'])

# Detect operational cycles
cp = CycleProcessor(dp.df, column_name='FeedFlow', threshold=10)
cp.identify_cycles()
cp.assign_cycle_features()

# Feature engineering
fe = FeatureEngineering(dp)
fe.generate_cross_features()
fe.lag_engineer(mode='cycle')
fe.rolling_mean_engineer()

# Physics features
phys = PhysicsBasedFeatures(dp)
phys.add_all_physics_features()
```

---

## Key Design Decisions

**Why hybrid physics + ML?**
Pure ML models learn correlations without physical grounding, leading to unreliable explanations. Physics models are interpretable but cannot capture all plant dynamics. The hybrid approach uses physics as a structured prior and ML to correct the systematic residual.

**Why cycle-aware features?**
RO systems undergo periodic backwash cycles. Naively computed lag features across cycle boundaries mix physically distinct operating states, creating false patterns. Cycle-aware lags are computed within each cycle using `groupby(cycle_id)`.

**Why three-stage feature selection?**
After feature engineering, the feature space contains hundreds of candidates. Random Forest provides a fast coarse cut, SHAP provides interaction-aware fine ranking, and LSTM ablation validates that each retained feature genuinely improves sequential prediction — not just correlation.

**Why dual detection (SADS + FADS)?**
Membrane fouling develops over days (slow drift) while sensor faults appear within minutes (sudden spikes). A single model cannot be sensitive to both timescales simultaneously. SADS operates on absolute values with a 12-step LSTM window; FADS operates on first-order derivatives (dy/dt) with a linear model and 3σ threshold.

---

## Data Format

The framework expects plant data in one of two formats:

**Long format (SCADA/historian export):**
```
timestamp        | param_name      | value
2023-01-01 00:00 | FeedFlow        | 285.3
2023-01-01 00:00 | FeedPressure    | 12.4
...
```
Use `dp.change_pivot('timestamp', 'param_name', 'value')` to convert.

**Wide format:**
```
timestamp        | FeedFlow | FeedPressure | FeedTemperature | ...
2023-01-01 00:00 | 285.3    | 12.4         | 24.1            | ...
```
Load directly without pivoting.

---

## Citation

If you use this framework in your research, please cite:

```
[Citation to be added upon publication]
```

---

## License

For academic use only. ECE498 Group Project — Zhejiang University International Campus.
