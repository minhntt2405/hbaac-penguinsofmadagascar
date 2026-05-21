# hbaac-penguinsofmadagascar
Daily sales quantity forecasting for 56 days across 15,972 SKUs of a Vietnamese auto parts distributor.
Competition metric: WRMSSE (Weighted Root Mean Squared Scaled Error, profit-weighted). Lower is better.

# Project Overview
In demand forecasting for retail/distribution, treating all items with a one-size-fits-all model often leads to poor performance, especially for intermittent or slow-moving stock. This solution implements a Segmentation-Routing Pipeline that categorizes items by their demand profile before applying specialized forecasting logic.
Core Pipeline
Preprocessing: Standardized data loading with anomaly handling (clipping returns).

Feature Engineering: Extraction of 16 key features capturing sparsity, recency, trend, and intermittency.

Intelligent Segmentation: A LightGBM Multi-class Classifier partitions SKUs into three distinct operational clusters.

Strategic Routing:

DEAD: Constant 0 prediction.

SPARSE: Decayed daily rate modeling.

ACTIVE: Ensemble of EWM and recent-window trends with day-of-week seasonality adjustment.

Post-processing: Forced zero-demand for Sundays (store closed).
