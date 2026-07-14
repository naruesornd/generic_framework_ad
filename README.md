# A Generic Framework for Multivariate Anomaly Detection and Root Cause Analysis Using Slow and Fast Detection in Process Industries

> Naruesorn Dechnorachai and Anjan K. Tula  
> College of Control Science and Engineering, Zhejiang University, Hangzhou, China  
> 

---

## Overview

This repository contains the full implementation, dataset, and supplementary materials for the paper above. The framework addresses six challenges in industrial anomaly detection: poor generalization, insufficient interpretability, multi-timescale anomalies, label scarcity, lack of physics–data fusion, and inadequate cyclic process handling, through a six-step modular pipeline.

The framework operates through a dual-path detection architecture:

- **SADS (Slow Anomaly Detection System)** — detects gradual, long-term deviations (membrane fouling, sensor drift) in the absolute value domain using a hybrid physics equation + LSTM residual model, operating on 1-hour interval data
- **FADS (Fast Anomaly Detection System)** — detects abrupt, sudden deviations (sensor faults, hydraulic disturbances) in the derivative domain using multivariate linear regression on first-order rate-of-change features, operating on 5-minute interval data

Both systems feed into an automated **Root Cause Analysis (RCA)** module that attributes each anomaly to its most probable contributing input feature through model-weighted Z-score deviation scoring (SADS-RCA) or derivative deviation scoring (FADS-RCA), operating in each detector's respective domain.

The framework is validated on a two-year real-world SCADA dataset from an operational Reverse Osmosis (RO) water treatment plant in China.

---

## Framework Pipeline
```
Raw SCADA Dataset (1-hour and 5-minute intervals)
│
▼
Step 1: Data Preprocessing
— Input/output variable partitioning, listwise deletion of missing values,
physical bounds filtering, uniform sampling interval verification
│
▼
Step 2: Temporal Cyclic Analysis
— Cycle boundary detection via simultaneous drops in key process indicators
(Feed Flow, Feed Pressure, Differential Pressure)
— Cycle ID and Cycle Time features appended to dataset
│
▼
Step 3: Feature Engineering
— Temporal lag features (l = 1, 2, 3, 6, 12)
— Pairwise cross features (pressure × flow rate, etc.)
— Total: 59 candidate features from 7 input variables
│
▼
Step 4: Feature Selection
— Stage 1: Random Forest (RF) → top 4K features by MDI importance
— Stage 2: SHAP TreeExplainer → top K=5 features per output variable
│
├──────────────────────┬──────────────────────┐
▼                      ▼
Step 5a: SADS               Step 5b: FADS
Physics model per output    Derivative-domain linear

LSTM residual model        regression (OLS)
3σ threshold              + 3σ threshold
(1-hour data)               (5-minute data)
│                      │
└──────────────────────┘
│
▼
Step 6: Root Cause Analysis (RCA)
— SADS-RCA: model-weighted Z-score ranking in absolute value domain
— FADS-RCA: regression-coefficient-weighted derivative deviation ranking
— Stacked time-series dashboard for operator verification

---


```

---

## Case Study: RO Water Treatment Plant

| Property | Value |
|---|---|
| Source | SCADA system, operational RO plant, China |
| Duration | ~2 years (2021–2023) |
| Sampling intervals | 1-hour (SADS) and 5-minute (FADS) |
| Total observations | 16,563 hourly timestamps |
| Process variables | 11 variables (7 inputs, 4 outputs) + timestamp |
| Anomaly labels | None (fully unlabelled real-world dataset) |


---

## Key Results

- SADS hybrid model improves Differential Pressure R² from −1.12 (physics-only) to 0.543, demonstrating the value of physics–ML hybridization for fouling-degraded systems
- Both SADS and FADS consistently achieve **AUC 0.85–0.97** across all output variables and severity levels under synthetic anomaly injection evaluation
- SADS achieves **zero false positives** across all tested configurations
- At strong severity (5σ), SADS reaches **F1 = 0.947–1.000** across all outputs; FADS reaches **F1 = 1.000** for Concentrate Pressure and Concentrate Flow
- IForest baseline achieves near-random AUC (0.50–0.53), confirming global outlier detectors are unsuitable for per-timestamp point anomaly detection

---

## Repository Structure

```

## Repository Structure
├── data/
│   ├── raw/                        # Raw SCADA dataset
│   └── processed/                  # Preprocessed datasets (1-hour and 5-minute)
│
├── src/
│   ├── data_processor/
│   │   ├── data_processor.py       # Data loading, variable partitioning, preprocessing
│   │   └── cycle_processor.py      # Operational cycle detection and segmentation
│   │   
│   ├── feature_engineering/
│   │   └── feature_engineering.py  # Lag, cross features, physics-derived features
│   │   
│   ├── model/
│   │   ├── coarse_feature_selection/
│   │   │   └── cfs.py              # RF + SHAP two-stage feature selection
│   │   └── lstm_model/             # LSTM with additive attention architecture
│   │      └── enhanced_lstm.py 
│   │       
│   ├── detection/
│   │   ├── sads.py                 # SADS: hybrid prediction and anomaly flagging
│   │   └── fads.py                 # FADS: derivative-domain detection
│   └── rca/
│       └── rca.py                  # SADS-RCA and FADS-RCA modules
│   
├── notebooks/
│   ├── 01_exploration.ipynb        # Data loading, cycle analysis, preprocessing
│   ├── 02_SADS.ipynb               # Physics models, LSTM training, SADS results, its synthetic anomaly injection results
│   ├── 03_FADS.ipynb               # FADS results and its synthetic anomaly injection results
│   
├── requirements.txt
└── README.md

---
```

**Key dependencies:** `torch`, `scikit-learn`, `shap`, `pandas`, `numpy`, `plotly`, `scipy`

Python 3.10+ recommended.

---

## Quick Start

```python
from src.data_processor.data_processor import DataProcessor
from src.data_processor.cycle_processor import CycleProcessor
from src.feature_engineering.feature_engineering import FeatureEngineering

# Step 1: Load and preprocess
dp = DataProcessor("data/raw/plant_data.csv")
dp.drop_NA_with_feature(['FeedFlow', 'FeedTemperature'])

# Step 2: Detect operational cycles
cp = CycleProcessor(dp.df, column_name='FeedFlow', threshold=0.05)
cp.identify_cycles()
cp.assign_cycle_features()

# Step 3–4: Feature engineering and selection
fe = FeatureEngineering(dp)
fe.generate_lag_features(lags=[1, 2, 3, 6, 12])
fe.generate_cross_features()

# Step 5a: Run SADS
from src.detection.sads import lstm_hybrid_model
model, threshold = lstm_hybrid_model(dp, target_col='DifferentialPressure', ...)

# Step 5b: Run FADS
from src.detection.fads import fast_anomaly_detection_system
fads_results = fast_anomaly_detection_system(dp, target_col='DifferentialPressure', ...)

# Step 6: Root cause analysis
from src.rca.rca import run_automated_rca_sads
rca_results = run_automated_rca_sads(dp, feature_name='DifferentialPressure', ...)
```

---

## Data Availability

The complete datasets are publicly available in this repository under `data/`. The dataset may be used for academic and non-commercial purposes under the repository licence.


---
---

## Licence

This repository is made available for academic research purposes.  
© 2025 Naruesorn Dechnorachai and Anjan K. Tula, Zhejiang University.

---

