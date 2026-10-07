# Phase 6.1: System A Modeling Contract & Baseline Strategy
### Supervised Demand Forecasting Model Architecture, Benchmarking Strategy, and Governance

**Contract Version:** 1.0.0  
**Target Sub-System:** System A (Supervised Hourly Demand Forecasting)  
**Input Source:** `data/processed/features_forecasting/` (Partitioned Parquet, Snappy Compressed)  
**Eligible Target Grain:** Exactly one meter-hour: `(district, meter_id, window_start)`  
**Training Filter:** `WHERE is_valid_forecast_instance == 1` (978,031 rows)  
**Target Label ($Y$):** `target_hourly_kwh` ($Y_{t+1}$, $[0.0, \infty)\text{ kWh}$)  
**Engine Compatibility:** Apache Spark 3.5+ / 4.x MLlib  
**Status:** 📋 DESIGN SPECIFICATION (Pre-Implementation Contract)

---

## 1. Problem Formulation & Empirical Scope

System A predicts next-hour active energy consumption for electrical meters across Uttar Pradesh:
$$\hat{Y}_{t+1} = f(\mathbf{X}_t; \mathbf{\Theta})$$
where $\mathbf{X}_t \in \mathbb{R}^{25}$ is the 25-dimensional feature vector available at prediction cutoff $t$, and $\mathbf{\Theta}$ represents learned model parameters.

### Chronological Partition Breakdown (Eligible Modeling Instances)

To prevent lookahead bias and temporal leakage, training and validation adhere strictly to chronological boundaries:

```
├── TRAIN SET:      2019-05-01 → 2020-08-31  (16 Months | 618,735 rows | 63.3% modeling data)
│   ├── Bareilly:   344,456 rows
│   └── Mathura:    274,279 rows
│
├── VALIDATION SET: 2020-09-01 → 2020-12-31  ( 4 Months | 152,833 rows | 15.6% modeling data)
│   ├── Bareilly:    96,749 rows
│   └── Mathura:     56,084 rows
│
└── TEST SET:       2021-01-01 → 2021-10-31  (10 Months | 206,463 rows | 21.1% modeling data)
    ├── Bareilly:   182,198 rows
    └── Mathura:     24,265 rows
```

* **Total Eligible Supervised Observations:** **978,031 rows**
* **Zero Null Guarantee:** 0 null values across all 25 features and target label in this subset.

---

## 2. Benchmark Baselines Specification (Level 0)

In smart meter forecasting, complex machine learning algorithms are unjustified unless they demonstrate statistically significant improvement over domain heuristics and naive persistence. Four canonical baselines are defined:

```
                             BENCHMARK BASELINES (LEVEL 0)
                                           │
         ┌──────────────────┬──────────────┴─────┬──────────────────┐
         ▼                  ▼                    ▼                  ▼
    B1: NAIVE          B2: DIURNAL          B3: WEEKLY         B4: ROLLING
   PERSISTENCE          SEASONAL             SEASONAL          24h MEAN
   Ŷ_{t+1} = Y_t      Ŷ_{t+1} = Y_{t-23}   Ŷ_{t+1} = Y_{t-167} Ŷ_{t+1} = μ_{24}(t)
   (lag_0h)           (lag_24h)            (lag_168h)         (rolling_mean_24h)
```

| Baseline ID | Name | Mathematical Formulation | Feature Source | Physical Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **B1** | **Naive Persistence** | $\hat{Y}_{t+1} = Y_t$ | `lag_0h` | Immediate short-term demand inertia. Assumes consumption in the next hour matches the current hour. |
| **B2** | **Diurnal Seasonal Naive** | $\hat{Y}_{t+1} = Y_{t-23}$ | `lag_24h` | Daily diurnal rhythm. Assumes consumer behavior at hour $h$ tomorrow matches hour $h$ yesterday. |
| **B3** | **Weekly Seasonal Naive** | $\hat{Y}_{t+1} = Y_{t-167}$ | `lag_168h` | Weekly rhythm. Assumes consumption matches the exact same hour and day of the previous week. |
| **B4** | **Rolling 24h Mean** | $\hat{Y}_{t+1} = \mu_{24}(t)$ | `rolling_mean_24h` | Moving average baseline smoothing high-frequency consumer volatility. |

*In order to be considered production-viable, any trained MLlib model must achieve lower RMSE and MAE than all four baselines on the Validation set.*

---

## 3. Evaluation Metrics Contract

Standard mean absolute percentage error ($\text{MAPE} = \frac{1}{N}\sum \frac{|Y - \hat{Y}|}{Y}$) is mathematically pathological on smart meter data because voluntary appliance standby ($Y \approx 0.001\text{ kWh}$) produces division-by-zero explosions ($>10,000\%$). 

The evaluation contract mandates five mathematically well-behaved metrics:

### 1. Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (Y_i - \hat{Y}_i)^2}$$
*Unit:* $\text{kWh}$  
*Utility:* Penalizes large peak load errors quadractically. Essential for grid capacity and transformer overload management.

### 2. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |Y_i - \hat{Y}_i|$$
*Unit:* $\text{kWh}$  
*Utility:* Direct linear measure of energy volume deviation.

### 3. Coefficient of Determination ($R^2$)
$$R^2 = 1 - \frac{\sum_{i=1}^N (Y_i - \hat{Y}_i)^2}{\sum_{i=1}^N (Y_i - \bar{Y})^2}$$
*Domain:* $(-\infty, 1.0]$  
*Utility:* Proportion of aggregate demand variance explained by the model relative to the historical mean.

### 4. Weighted Absolute Percentage Error (WAPE)
$$\text{WAPE} = \frac{\sum_{i=1}^N |Y_i - \hat{Y}_i|}{\sum_{i=1}^N Y_i} \times 100\%$$
*Domain:* $[0\%, \infty)$  
*Utility:* Global percentage deviation normalized by total energy consumed. Immune to near-zero division singularities.

### 5. Normalized Root Mean Squared Error (NRMSE)
$$\text{NRMSE} = \frac{\text{RMSE}}{\bar{Y}} \times 100\%$$
*Domain:* $[0\%, \infty)$  
*Utility:* Scale-free metric enabling cross-district comparisons (Bareilly vs Mathura).

---

## 4. Candidate Machine Learning Model Portfolio

```
                              MODEL CANDIDATE PROGRESSION
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
  LEVEL 1: LINEAR MODELS                                            LEVEL 2: TREE ENSEMBLES
  - Ordinary Least Squares (OLS)                                    - GBDT Regressor (GBTRegressor)
  - Ridge Regression (L2, α=0.0)                                    - Random Forest (RandomForestRegressor)
  - ElasticNet (L1+L2, α=0.5)                                       Captures non-linear power-quality
  High interpretability & lag elasticity                             thresholds & cyclical interactions
```

### Level 1: Linear Models via Spark MLlib
* **Model 1.1: Ordinary Least Squares (OLS Linear Regression)**: Baseline parametric linear combination.
* **Model 1.2: Ridge Regression ($L_2$ Regularization)**: `elasticNetParam = 0.0`, `regParam > 0`. Dampens collinear weights between adjacent autoregressive lags ($Y_t, Y_{t-1}, Y_{t-2}$) and rolling statistics.
* **Model 1.3: ElasticNet Regression ($L_1 + L_2$)**: `elasticNetParam = 0.5`. Combines feature sparsity with collinearity stability.

### Level 2: Non-Linear Tree Ensembles via Spark MLlib
* **Model 2.1: Gradient-Boosted Trees (`GBTRegressor`)**: Sequentially builds shallow regression trees to minimize residual squared loss. Captures non-linear thresholds (e.g., severe brownout causing motor stalls, interactions between afternoon hours and heat loads).
* **Model 2.2: Random Forest (`RandomForestRegressor`)**: Bagged ensemble of decorrelated decision trees, providing robustness against high-variance transient loads.

---

## 5. Spark ML Pipeline Architecture & Feature Assembly

### Feature Vector Specification ($d = 25$ dimensions)

```python
FEATURE_COLUMNS = [
    # Group A: Autoregressive Consumption Lags (5 dims)
    "lag_0h",
    "lag_1h",
    "lag_2h",
    "lag_24h",
    "lag_168h",
    # Group B: Rolling 24h Demand Statistics (4 dims)
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_max_24h",
    "rolling_min_24h",
    # Group C: Cyclical Fourier Temporal Encodings (6 dims)
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    # Group D: Cutoff-Time Power-Quality Context (8 dims)
    "mean_voltage_t",
    "mean_current_t",
    "mean_freq_t",
    "outage_ratio_t",
    "standby_ratio_t",
    "brownout_ratio_t",
    "spike_ratio_t",
    "rolling_outage_ratio_24h",
    # Group E: Spatial Identifier (1 dim)
    "district_idx",
]
```

### End-to-End Pipeline Stages

```
   Raw Features (25 Cols)
             │
             ▼
┌───────────────────────────┐
│      VectorAssembler      │ ──► Assembles 25 columns into raw_features vector
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│       StandardScaler      │ ──► Standardizes to μ=0, σ=1 (Fitted STRICTLY on TRAIN)
└────────────┬──────────────┘     (Mandatory for Linear/Ridge/Lasso; bypassed for Trees)
             │
             ▼
┌───────────────────────────┐
│     Estimator Stage       │ ──► LinearRegression, GBTRegressor, or RandomForest
└────────────┬──────────────┘
             │
             ▼
       Predictions (Ŷ)
```

### Feature Scaling Policy
* **Linear Models:** `StandardScaler(withMean=True, withStd=True)` is **mandatory** because regularized penalties ($\lambda \sum \theta_j^2$) require uniform feature scales. The scaler MUST be fit exclusively on the `TRAIN` set and transformed onto `VAL` and `TEST` to prevent data leakage.
* **Tree Models:** Tree partitions are invariant to monotonic feature scaling; trees operate directly on raw vectors.

---

## 6. Execution & Implementation Roadmap for Phase 6

Phase 6 implementation proceeds through controlled sub-milestones:

* **Phase 6.1 (Current):** Authoritative Modeling Contract & Baseline Architecture (Freeze specifications).
* **Phase 6.2:** Baseline Evaluation Script (`src/evaluate_baselines.py`): Implement and evaluate B1, B2, B3, B4 across TRAIN and VAL to establish mathematical benchmarks.
* **Phase 6.3:** MLlib Linear Modeling Pipeline (`src/train_linear_models.py`): Implement `VectorAssembler`, `StandardScaler`, and fit OLS + Ridge regression.
* **Phase 6.4:** MLlib Non-Linear Tree Pipeline (`src/train_tree_models.py`): Implement GBDT and Random Forest regressors.
* **Phase 6.5:** Validation Model Comparison & Selection: Compare all candidates against baselines on `VAL` (152,833 rows).
* **Phase 6.6:** Final Evaluation on Held-Out `TEST` Set: Evaluate winning model candidate once on `TEST` (206,463 rows).
* **Phase 6.7:** Residual Analysis & Model Freeze: Error distributions, peak hour error analysis, and artifact serialization to `models/system_a/`.

---

## 7. Model Governance & Artifact Contracts

1. **Serialization Directory:** All serialized Spark ML models must be saved under `models/system_a/<model_name>/` using native Spark pipeline persistence:
   ```python
   pipeline_model.write().overwrite().save("models/system_a/ridge_best")
   ```
2. **Metadata Sidecar:** Each persisted model must be accompanied by a JSON metadata file recording:
   - Git commit hash
   - Training duration
   - Feature column list and dimensionality ($d=25$)
   - Hyperparameter configurations (`regParam`, `maxDepth`, etc.)
   - Validation performance metrics (RMSE, MAE, $R^2$, WAPE)
3. **Reproducibility Guarantee:** Random seeds for all stochastic tree partitioning algorithms must be explicitly fixed (`seed=42`).
