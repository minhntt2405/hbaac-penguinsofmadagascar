"""
Vietnamese Auto Parts Demand Forecasting
Strategy: LightGBM SKU Segmentation + EWM / Sparse / Dead routing
Target: WRMSSE < 0.48

Pipeline:
  1. Load & clean raw sales data
  2. Compute per-SKU features (sparsity, recency, activity windows, etc.)
  3. Train a LightGBM multi-class classifier to segment SKUs into:
       Class 0 – DEAD        (no sale >730 days)
       Class 1 – SPARSE      (<SPARSE_CUTOFF active days, recent)
       Class 2 – ACTIVE      (≥SPARSE_CUTOFF active days)
  4. Route each SKU to its matched forecasting strategy
  5. Force Sunday predictions to 0 (store closed)
  6. Write submission CSV
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_PATH      = "train.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"
OUTPUT_PATH     = "submission_lgbm.csv"

LAST_TRAIN_DATE = pd.Timestamp("2025-09-05")
VAL_DATES       = pd.date_range("2025-09-06",  periods=28)   # F1..F28
EVAL_DATES      = pd.date_range("2025-10-04",  periods=28)   # F29..F56

HORIZON         = 28
EWM_ALPHA       = 0.15    # smoothing factor – higher = more reactive
EWM_WEIGHT      = 0.60    # blend weight on EWM vs. recent-window mean
RECENT_WINDOW   = 56      # days used for recent mean
DOW_WINDOW      = 112     # days used for day-of-week pattern
SPARSE_CUTOFF   = 20      # min active days to qualify as ACTIVE
DEAD_SKU_CUTOFF = 730     # days since last sale → Dead

# LightGBM classifier settings
LGBM_PARAMS = dict(
    objective        = "multiclass",
    num_class        = 3,
    n_estimators     = 400,
    learning_rate    = 0.05,
    num_leaves       = 31,
    max_depth        = -1,
    min_child_samples= 20,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    verbose          = -1,
    n_jobs           = -1,
    random_state     = 42,
)
LGBM_CV_FOLDS = 5

# Segment label mapping
SEG_DEAD   = 0
SEG_SPARSE = 1
SEG_ACTIVE = 2
SEG_NAMES  = {SEG_DEAD: "DEAD", SEG_SPARSE: "SPARSE", SEG_ACTIVE: "ACTIVE"}


# ─── 1. LOAD & CLEAN ───────────────────────────────────────────────────────────
print("=" * 60)
print("1. Loading training data …")
train = pd.read_csv(
    TRAIN_PATH, low_memory=False,
    usecols=["Date", "ItemCode", "Quantity"],
)
train["Date"]     = pd.to_datetime(train["Date"])
train["Quantity"] = pd.to_numeric(train["Quantity"], errors="coerce").fillna(0)
train["Quantity"] = train["Quantity"].clip(lower=0)   # remove returns

# Daily aggregation per SKU
daily_raw = train.groupby(["Date", "ItemCode"])["Quantity"].sum()

# Build lookup dict  {sku: {date: qty}}
print("  Building per-SKU series …")
sku_series: dict[str, dict] = {}
for (date, sku), qty in daily_raw.items():
    if sku not in sku_series:
        sku_series[sku] = {}
    sku_series[sku][date] = qty

all_skus = list(sku_series.keys())
print(f"  → {len(all_skus):,} unique SKUs")


# ─── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
def build_features(sku: str) -> dict:
    """
    Compute a feature vector for one SKU used by the LightGBM classifier.

    Features (all derived from historical sales only):
    ─ Sparsity / activity counts ─────────────────────────────────────────────
    non_zero_count_14   : number of days with sales in last 14 days
    non_zero_count_30   : number of days with sales in last 30 days
    non_zero_count_90   : number of days with sales in last 90 days
    non_zero_count_all  : total number of days with any sale in history

    ─ Intermittency / gap structure ──────────────────────────────────────────
    zero_streak_length  : number of consecutive zero days before LAST_TRAIN_DATE
    avg_gap_between_sales : mean gap (days) between consecutive non-zero days
    max_gap_between_sales : max gap between consecutive non-zero sales days
    cv_qty              : coefficient of variation of non-zero quantities

    ─ Recency ────────────────────────────────────────────────────────────────
    days_since_last_sale : days from last sale day to LAST_TRAIN_DATE
    global_activation_rate : fraction of all history days that had a sale

    ─ Volume & trend ─────────────────────────────────────────────────────────
    mean_qty_nonzero    : mean quantity on sale days
    mean_qty_30         : mean daily quantity (incl. zeros) last 30 days
    mean_qty_90         : mean daily quantity (incl. zeros) last 90 days
    qty_trend_ratio     : mean_qty_30 / (mean_qty_90 + 1e-6)  — acceleration
    max_qty             : max single-day quantity in history

    ─ History length ─────────────────────────────────────────────────────────
    history_days        : total calendar span of observed history
    """
    s = pd.Series(sku_series[sku]).sort_index()
    s = s[s.index <= LAST_TRAIN_DATE]

    # Full dense series over the observed calendar span
    full_idx  = pd.date_range(s.index.min(), LAST_TRAIN_DATE)
    s_dense   = s.reindex(full_idx, fill_value=0)
    history_days = len(full_idx)

    # ── Non-zero counts over windows ──────────────────────────────────────
    nz_all  = int((s_dense > 0).sum())
    nz_90   = int((s_dense.tail(90)  > 0).sum())
    nz_30   = int((s_dense.tail(30)  > 0).sum())
    nz_14   = int((s_dense.tail(14)  > 0).sum())

    # ── Zero-streak length (trailing zeros before reference date) ─────────
    streak = 0
    for v in reversed(s_dense.values):
        if v == 0:
            streak += 1
        else:
            break
    zero_streak_length = streak

    # ── Days since last sale ───────────────────────────────────────────────
    days_since_last = (LAST_TRAIN_DATE - s.index.max()).days

    # ── Gap statistics between sales days ─────────────────────────────────
    sale_dates = s_dense[s_dense > 0].index
    if len(sale_dates) >= 2:
        gaps = np.diff(sale_dates).astype("timedelta64[D]").astype(int)
        avg_gap = float(gaps.mean())
        max_gap = float(gaps.max())
    else:
        avg_gap = float(history_days)
        max_gap = float(history_days)

    # ── Global activation rate ─────────────────────────────────────────────
    global_activation_rate = nz_all / max(history_days, 1)

    # ── Volume statistics ──────────────────────────────────────────────────
    nonzero_vals = s_dense[s_dense > 0].values
    mean_qty_nonzero = float(nonzero_vals.mean()) if len(nonzero_vals) else 0.0
    cv_qty           = (float(nonzero_vals.std()) / (mean_qty_nonzero + 1e-6)
                        if len(nonzero_vals) > 1 else 0.0)
    max_qty          = float(s_dense.max())

    mean_qty_30 = float(s_dense.tail(30).mean())
    mean_qty_90 = float(s_dense.tail(90).mean())
    qty_trend_ratio = mean_qty_30 / (mean_qty_90 + 1e-6)

    return {
        # activity
        "non_zero_count_14"      : nz_14,
        "non_zero_count_30"      : nz_30,
        "non_zero_count_90"      : nz_90,
        "non_zero_count_all"     : nz_all,
        # intermittency
        "zero_streak_length"     : zero_streak_length,
        "avg_gap_between_sales"  : avg_gap,
        "max_gap_between_sales"  : max_gap,
        "cv_qty"                 : cv_qty,
        # recency
        "days_since_last_sale"   : days_since_last,
        "global_activation_rate" : global_activation_rate,
        # volume & trend
        "mean_qty_nonzero"       : mean_qty_nonzero,
        "mean_qty_30"            : mean_qty_30,
        "mean_qty_90"            : mean_qty_90,
        "qty_trend_ratio"        : qty_trend_ratio,
        "max_qty"                : max_qty,
        # history
        "history_days"           : history_days,
    }


print("\n2. Engineering SKU features …")
feature_rows = []
for sku in all_skus:
    row = build_features(sku)
    row["sku"] = sku
    feature_rows.append(row)

feat_df = pd.DataFrame(feature_rows).set_index("sku")

FEATURE_COLS = [c for c in feat_df.columns]
print(f"  → Feature matrix: {feat_df.shape}  ({len(FEATURE_COLS)} features)")


# ─── 3. RULE-BASED LABELS (used as training signal for LightGBM) ──────────────
print("\n3. Generating rule-based segment labels …")

def rule_label(row) -> int:
    if row["days_since_last_sale"] > DEAD_SKU_CUTOFF:
        return SEG_DEAD
    if row["non_zero_count_all"] < SPARSE_CUTOFF:
        return SEG_SPARSE
    return SEG_ACTIVE

feat_df["label"] = feat_df.apply(rule_label, axis=1)

seg_counts = feat_df["label"].value_counts().sort_index()
for seg_id, cnt in seg_counts.items():
    print(f"  {SEG_NAMES[seg_id]:8s}  ({seg_id}): {cnt:,}")


# ─── 4. TRAIN LightGBM CLASSIFIER ─────────────────────────────────────────────
print("\n4. Training LightGBM segmentation classifier …")

X = feat_df[FEATURE_COLS].values.astype(np.float32)
y = feat_df["label"].values

skf   = StratifiedKFold(n_splits=LGBM_CV_FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), 3), dtype=np.float32)
models    = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    oof_preds[va_idx] = model.predict_proba(X_va)
    models.append(model)
    acc = (model.predict(X_va) == y_va).mean()
    print(f"  Fold {fold}: val-accuracy = {acc:.4f}")

# Final predictions: average ensemble over folds
ensemble_probs = np.mean([m.predict_proba(X) for m in models], axis=0)
lgbm_labels    = ensemble_probs.argmax(axis=1)

feat_df["lgbm_segment"] = lgbm_labels

print("\n  LightGBM segment distribution:")
for seg_id in [SEG_DEAD, SEG_SPARSE, SEG_ACTIVE]:
    cnt = (lgbm_labels == seg_id).sum()
    print(f"    {SEG_NAMES[seg_id]:8s}: {cnt:,}")

# Feature importance (top 10)
imp = np.mean([m.feature_importances_ for m in models], axis=0)
top_feat = sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1])[:10]
print("\n  Top-10 feature importances:")
for fname, fimp in top_feat:
    print(f"    {fname:<30s}  {fimp:.1f}")

# Build SKU → segment mapping
sku_segment = dict(zip(feat_df.index, feat_df["lgbm_segment"].values))


# ─── 5. FORECASTING FUNCTIONS ──────────────────────────────────────────────────

def forecast_active(sku: str, forecast_dates: pd.DatetimeIndex) -> list[int]:
    """EWM + recent-mean + day-of-week scaling for active SKUs."""
    s = pd.Series(sku_series[sku]).sort_index()
    s = s[s.index <= LAST_TRAIN_DATE]

    full_start = max(s.index.min(), LAST_TRAIN_DATE - pd.Timedelta(days=364))
    full_idx   = pd.date_range(full_start, LAST_TRAIN_DATE)
    s_full     = s.reindex(full_idx, fill_value=0)

    d112 = s_full.tail(DOW_WINDOW)
    d56  = s_full.tail(RECENT_WINDOW)
    d28  = s_full.tail(28)

    dow_mean    = d112.groupby(d112.index.dayofweek).mean()
    global_mean = d112.mean()
    ewm_val     = d28.ewm(alpha=EWM_ALPHA).mean().iloc[-1]
    recent_mean = d56.mean()
    base        = EWM_WEIGHT * ewm_val + (1 - EWM_WEIGHT) * recent_mean

    preds = []
    for fd in forecast_dates:
        dow = fd.dayofweek
        if global_mean > 0 and dow in dow_mean.index:
            dow_ratio = dow_mean[dow] / global_mean
        else:
            dow_ratio = 1.0
        preds.append(max(0, round(base * dow_ratio)))

    return preds


def forecast_sparse(sku: str, forecast_dates: pd.DatetimeIndex) -> list[int]:
    """Decayed daily-rate for sparse SKUs."""
    s = pd.Series(sku_series[sku]).sort_index()
    s = s[s.index <= LAST_TRAIN_DATE]

    days_since_last = (LAST_TRAIN_DATE - s.index.max()).days
    total_qty       = float(s.clip(lower=0).sum())
    history_days    = max((s.index.max() - s.index.min()).days + 1, 56)
    daily_rate      = total_qty / history_days

    if days_since_last > 112:
        decay = 0.3
    elif days_since_last > 56:
        decay = 0.6
    else:
        decay = 1.0

    base = daily_rate * decay
    return [max(0, round(base))] * HORIZON


def forecast_sku(sku: str, forecast_dates: pd.DatetimeIndex) -> list[int]:
    """
    Route a single SKU to the appropriate forecast strategy
    based on its LightGBM-assigned segment.
    """
    if sku not in sku_series:
        return [0] * HORIZON

    seg = sku_segment.get(sku, SEG_SPARSE)   # fallback: sparse

    if seg == SEG_DEAD:
        preds = [0] * HORIZON
    elif seg == SEG_ACTIVE:
        preds = forecast_active(sku, forecast_dates)
    else:  # SEG_SPARSE
        preds = forecast_sparse(sku, forecast_dates)

    # ── STORE CLOSED ON SUNDAYS → force 0 ──────────────────────────────────
    preds = [
        0 if fd.dayofweek == 6 else p
        for p, fd in zip(preds, forecast_dates)
    ]

    return preds


# ─── 6. RUN FORECASTS ──────────────────────────────────────────────────────────
print("\n5. Generating submission forecasts …")
sub     = pd.read_csv(SAMPLE_SUB_PATH)
sub_ids = sub["id"].tolist()

results = []
for i, row_id in enumerate(sub_ids):
    if i % 10_000 == 0:
        print(f"  {i:,} / {len(sub_ids):,}")

    sku = row_id.replace("_validation", "").replace("_evaluation", "")
    if "_validation" in row_id:
        preds = forecast_sku(sku, VAL_DATES)
    else:
        preds = forecast_sku(sku, EVAL_DATES)

    results.append(preds)


# ─── 7. OUTPUT ─────────────────────────────────────────────────────────────────
forecast_cols = [f"F{i}" for i in range(1, HORIZON + 1)]
out = pd.DataFrame(results, columns=forecast_cols)
out.insert(0, "id", sub_ids)

vals = out[forecast_cols].values.flatten()
print(f"\n✓ Done! Summary stats:")
print(f"  Output shape : {out.shape}")
print(f"  Mean pred    : {vals.mean():.4f}")
print(f"  % zeros      : {(vals == 0).mean():.1%}")
flat_pct = (out[forecast_cols].nunique(axis=1) == 1).mean()
print(f"  % flat rows  : {flat_pct:.1%}")

out.to_csv(OUTPUT_PATH, index=False)
print(f"\n→ Saved to {OUTPUT_PATH}")


# ─── 8. QUICK SANITY CHECK ─────────────────────────────────────────────────────
print("\nSample predictions (F1-F7) for top SKUs:")
for sku in ["SKU-00003", "SKU-00002", "SKU-09458"]:
    row = out[out["id"] == f"{sku}_validation"]
    if len(row):
        seg_name = SEG_NAMES.get(sku_segment.get(sku, -1), "UNKNOWN")
        vals_7   = row.iloc[0, 1:8].values
        print(f"  {sku} [{seg_name:6s}]: {vals_7}")

# Verify Sunday = 0
sunday_cols = [f"F{i+1}" for i, d in enumerate(VAL_DATES) if d.dayofweek == 6]
if sunday_cols:
    sun_vals = out[sunday_cols].values.flatten()
    print(f"\nSunday columns in VAL window: {sunday_cols}")
    print(f"  All Sunday predictions = 0? {(sun_vals == 0).all()}")
