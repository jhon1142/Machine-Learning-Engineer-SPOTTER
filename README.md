# Machine-Learning-Engineer-SPOTTER
# SPOTTER — Freight Rate Prediction

Predict the `posted_rate` (USD) for US freight loads given route, equipment, weight, date, and market signals.

---

## Repository Contents

| File | Description |
|---|---|
| `spotter_en.ipynb` | Main notebook — EDA, feature engineering, and all models (English) |
| `spotter.ipynb` | Same notebook in Spanish |
| `score.py` | Official validator provided by SPOTTER |
| `validation_predictions.csv` | Final predictions for the 12,000-row validation set |
| `december_chart_inputs_filled.csv` | December 2025 daily rate predictions (Lexington → Fort Wayne) |
| `validation_approach.tex` | LaTeX document describing the train/test split and validation strategy |
| `loom_script.pdf` | Video walkthrough script (2–3 min Loom) |

---

## Problem

Given a freight load with known pickup/delivery cities, equipment type, weight, date, `market_index`, and `quote_signal`, predict the `posted_rate` in USD.

- **Train set:** 48,000 labelled loads — Jan 1 to Oct 31, 2025
- **Validation set:** 12,000 unlabelled loads — Nov 1 to Dec 31, 2025
- **Target:** `posted_rate` (continuous, right-skewed, USD)

---

## Approach

### 1. Data Cleaning

- **Negative weights (~6% of records):** confirmed as sign-entry errors via KDE comparison; corrected with `abs()` and flagged with `weight_sign_corrected`.
- **Missing values:** `weight` (300 nulls) and `market_index` (374 nulls) imputed with the **training-set median** only. A `market_index_missing` flag preserves the missingness signal.
- **EDA-only columns:** `rate_per_mile` and `distance_group` dropped before the split to prevent data leakage.

### 2. Feature Engineering

New features added via `add_features()`:

| Feature | Description |
|---|---|
| `bearing` | Direction of travel (0–360°) |
| `delta_lat`, `delta_lon` | Absolute coordinate differences |
| `haversine_km` | Geographic distance (Haversine formula) |
| `dist_x_market` | `distance × market_index` |
| `dist_x_quote` | `distance × quote_signal` |
| `quote_x_market` | `quote_signal × market_index` |
| `dist_dry_van/flatbed/reefer` | Distance × equipment type |
| `month`, `quarter`, `day_of_week`, `day_of_year` | Temporal features from `date` |

### 3. Train / Test Split

- **80/20 random split** (`random_state=42`) → 38,400 train / 9,600 test
- All transformers (imputer, scaler, encoder) are **fitted on train only** and applied with `.transform()` everywhere else
- The external validation set (Nov–Dec) serves as the true temporal holdout

### 4. Models

| Model | Validation | Notes |
|---|---|---|
| Linear Regression | Internal test 20% | Baseline |
| Random Forest | Internal test 20% | 300 trees |
| XGBoost | Internal test 20% | log-target, 1000 trees |
| LightGBM | Internal test 20% | log-target, num_leaves=63 |
| XGBoost + Optuna | 3-fold CV + test | 50 Bayesian trials |
| LightGBM + Optuna | 3-fold CV + test | 50 Bayesian trials |
| MLP Neural Network | Internal test 20% | 512→256→128→64, BatchNorm, Dropout |
| **Stacking Ensemble** | **5-fold OOF + test** | **Ridge meta-model — final model** |

The target is log-transformed (`log1p`) for all gradient boosting and neural network models to reduce skewness and improve fit on low-rate loads.

### 5. Stacking Ensemble (Final Model)

Base learners generate **out-of-fold (OOF) predictions** across 5 folds — each row is predicted by a model that never saw it during training. A **Ridge meta-model** (`α=1.0`) combines the four base-learner columns in log-space.

---

## Setup

```bash
pip install pandas numpy scikit-learn xgboost lightgbm optuna matplotlib seaborn plotly folium
# Optional — for the PyTorch neural network
pip install torch
```

---

## Running the Notebook

1. Open `spotter_en.ipynb` in Jupyter or Google Colab
2. Run all cells in order — data loads directly from GitHub URLs
3. The final cells generate `validation_predictions.csv` and `december_chart_inputs_filled.csv`
4. The last cell runs `score.py` automatically to validate both output files

---

## Validating the Output

```bash
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions december_chart_inputs_filled.csv \
  --output-dir scorer_results
```

Expected output:

```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
```

---

## Key Findings

- **Route (pickup + delivery + equipment) explains 87% of rate variance** — the dominant signal
- Date explains < 1% of variance → time-series methods add negligible value
- Flatbed commands the highest rates per mile; Dry Van the lowest
- `posted_rate` is strongly right-skewed (skewness ≈ 2.1) → log-transform required
- Autocorrelation of daily average rates at lag-1 = 0.33 — moderate market momentum signal

---

## Data Sources

| Dataset | URL |
|---|---|
| Train | `https://raw.githubusercontent.com/jhon1142/Machine-Learning-Engineer-SPOTTER/refs/heads/main/train-test.csv` |
| Validation | `https://raw.githubusercontent.com/jhon1142/Machine-Learning-Engineer-SPOTTER/refs/heads/main/validation.csv` |
| December inputs | `https://raw.githubusercontent.com/jhon1142/Machine-Learning-Engineer-SPOTTER/refs/heads/main/december-chart-inputs.csv` |
