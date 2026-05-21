# hbaac-penguinsofmadagascar
Daily sales quantity forecasting for 56 days across 15,972 SKUs of a Vietnamese auto parts distributor.
Competition metric: WRMSSE (Weighted Root Mean Squared Scaled Error, profit-weighted). Lower is better.

## Project Overview
**Task:** Given ~5 years of transaction history (2020-11-17 → 2025-09-05), predict daily net sales quantity for each SKU over two 28-day windows:
- **Validation window** (Public LB): F1–F28 = 2025-09-06 → 2025-10-03
- **Evaluation window** (Private LB): F29–F56 = 2025-10-04 → 2025-10-31

**Core Pipeline:**
1. Preprocessing: Standardized data loading with anomaly handling (clipping returns).

2. Feature Engineering: Extraction of 16 key features capturing sparsity, recency, trend, and intermittency.

3. Intelligent Segmentation: A LightGBM Multi-class Classifier partitions SKUs into three distinct operational clusters.

4. Strategic Routing:

- DEAD: Constant 0 prediction.

- SPARSE: Decayed daily rate modeling.

- ACTIVE: Ensemble of EWM and recent-window trends with day-of-week seasonality adjustment.

5. Post-processing: Forced zero-demand for Sundays (store closed).


## 🏗️ Technical Architecture

| Segment | Forecasting Strategy | Best For |
| :--- | :--- | :--- |
| **DEAD** | Zero Forecast | Items with no sales > 730 days |
| **SPARSE** | Decayed Daily Rate | Low-volume items (< 20 active days) |
| **ACTIVE** | EWM + DOW Scaling | High-frequency, trend-driven items |

---

## ⚙️ Configuration & Hyperparameters

The pipeline is highly tunable via the `CONFIG` section. Key parameters include:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `EWM_ALPHA` | `0.15` | Smoothing factor for EWM; higher values increase reactivity. |
| `EWM_WEIGHT` | `0.60` | Blend weight between EWM and Recent-Window mean. |
| `SPARSE_CUTOFF` | `20` | Minimum active days to qualify a SKU as "ACTIVE". |
| `DEAD_SKU_CUTOFF` | `730` | Days since last sale to categorize as "DEAD". |
| `RECENT_WINDOW` | `56` | Days used to calculate recent mean performance. |

---

## 📂 Directory Structure

```text
├── train.csv               # Raw historical sales data
├── sample_submission.csv   # Template for submission
├── forecast_model_lgbm.py  # Main processing & forecasting pipeline
├── submission_lgbm.csv     # Generated forecast output
└── README.md               # Project documentation

---

# 🛠️ Setup & Execution # 
** Prerequisites **
Python 3.8+, pandas, numpy, lightgbm, scikit-learn
