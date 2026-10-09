# System A Forecasting — Phase 6.2 Domain Baseline Evaluation Report

**System:** System A (Next-Hour Electricity Demand Forecasting)  
**Milestone:** Phase 6.2 Domain Baseline Evaluation  
**Status:** ✅ Complete & Verified (Pending User Approval)  
**Evaluator Implementation:** [`src/evaluate_baselines.py`](file:///c:/Users/kodal/BDA-ML/src/evaluate_baselines.py)  
**Test Suite:** [`tests/test_baselines.py`](file:///c:/Users/kodal/BDA-ML/tests/test_baselines.py) (14/14 Invariant Tests Passed; 47/47 Full Repo Tests Passed)  
**Deterministic Output:** [`reports/baselines/baseline_metrics.json`](file:///c:/Users/kodal/BDA-ML/reports/baselines/baseline_metrics.json) (under `splits` key)  
**Execution Runtime:** 46.94 seconds  
**Authoritative Contracts:**
- [`docs/SYSTEM_A_MODELING_CONTRACT.md`](file:///c:/Users/kodal/BDA-ML/docs/SYSTEM_A_MODELING_CONTRACT.md) (Phase 6.1)
- [`docs/FORECASTING_FEATURE_STORE_CONTRACT.md`](file:///c:/Users/kodal/BDA-ML/docs/FORECASTING_FEATURE_STORE_CONTRACT.md) (Phase 5.1)
- [`docs/CANONICAL_DATA_CONTRACT.md`](file:///c:/Users/kodal/BDA-ML/docs/CANONICAL_DATA_CONTRACT.md) (Phase 4.1)
- [`docs/DATA_QUALITY_POLICY.md`](file:///c:/Users/kodal/BDA-ML/docs/DATA_QUALITY_POLICY.md) (Phase 4.1)

---

## 1. Executive Summary & Objective

In accordance with Section 2 of [`docs/SYSTEM_A_MODELING_CONTRACT.md`](file:///c:/Users/kodal/BDA-ML/docs/SYSTEM_A_MODELING_CONTRACT.md), complex machine learning models (Linear Models, Ridge, ElasticNet, GBDT, Random Forest) are unjustified unless they demonstrate statistically significant improvement over domain heuristics and naive persistence.

Phase 6.2 establishes the **Level 0 Benchmark Baselines** for next-hour smart-meter electricity demand forecasting. Following the pre-implementation mathematical alignment audit, the evaluation was executed under **Option C, Path C1 (Strict Common Intersection)**:

1. **Exact Mathematical Temporal Alignment:** B2 (Diurnal Seasonal Naive) and B3 (Weekly Seasonal Naive) retrieve the true physical targets ($Y_{t-23}$ and $Y_{t-167}$) via distributed timestamp joins against historical feature-store records, eliminating the 1-hour cutoff offset present in standard row lags.
2. **Strict Data-Quality Verification:** Historical target predecessors are validated against existing data-quality rules (`target_is_valid == 1`, non-null, non-negative, finite). Predecessors from outage or incomplete hours are rejected.
3. **Strict Common Cohort Invariance:** All four baselines (B1–B4) are evaluated on the exact identical row set within each split, guaranteeing zero missing values and uniform sample sizes across baselines.
4. **Chronological Splitting & Test Isolation:** Evaluated strictly across **TRAIN** and **VAL** splits. The **TEST** split target instances are excluded upfront prior to alignment; TEST metrics are neither evaluated nor accessed for model development or benchmark selection.
5. **Upstream Artifact Preservation:** Executed without modifying frozen Silver, Gold, or Feature Store datasets, contracts, or pipelines.

---

## 2. Baseline Formulations & Alignment Architecture

```
                              BENCHMARK BASELINES (LEVEL 0)
                                            │
          ┌──────────────────┬──────────────┴─────┬──────────────────┐
          ▼                  ▼                    ▼                  ▼
     B1: NAIVE          B2: DIURNAL          B3: WEEKLY         B4: ROLLING
    PERSISTENCE          SEASONAL             SEASONAL          24h MEAN
    Ŷ_{t+1} = Y_t      Ŷ_{t+1} = Y_{t-23}   Ŷ_{t+1} = Y_{t-167} Ŷ_{t+1} = μ_{24}(t)
    (lag_0h)        (t-24h origin target) (t-168h origin target)(rolling_mean_24h)
```

### Baseline Specifications

| Baseline ID | Baseline Name | Mathematical Formulation | Source / Alignment Mechanism | Physical Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **B1** | **Naive Persistence** | $\hat{Y}_{t+1} = Y_t$ | Direct column `lag_0h` | Immediate short-term demand inertia. Predicts next-hour consumption matches current hour. |
| **B2** | **Diurnal Seasonal Naive** | $\hat{Y}_{t+1} = Y_{t-23}$ | Distributed join with historical record at $\text{window\_start} - 24\text{h}$, retrieving `target_hourly_kwh` | Daily diurnal rhythm. Predicts consumer behavior matches the exact same clock hour on the previous day. |
| **B3** | **Weekly Seasonal Naive** | $\hat{Y}_{t+1} = Y_{t-167}$ | Distributed join with historical record at $\text{window\_start} - 168\text{h}$, retrieving `target_hourly_kwh` | Weekly rhythm. Predicts consumer behavior matches the exact same clock hour and day of the previous week. |
| **B4** | **Rolling 24h Mean** | $\hat{Y}_{t+1} = \mu_{24}(t)$ | Direct column `rolling_mean_24h` | 24-hour moving average baseline smoothing high-frequency consumer volatility. |

### Temporal Availability & Anti-Leakage Proof
- At forecast origin $T_t$ (representing telemetry window $[T_t, T_t + 1\text{h})$), the decision-maker has observed telemetry up to cutoff timestamp $T_{t+1} = T_t + 1\text{h}$.
- The forecast target is next-hour demand $Y_{t+1}$, measuring consumption over interval $[T_{t+1}, T_{t+2})$.
- For B2, the lookup target $Y_{t-23}$ corresponds to measurement interval $[T_t - 23\text{h}, T_t - 22\text{h})$. This observation concluded at $T_t - 22\text{h}$, which is **22 hours prior to forecast origin $T_t$** and **23 hours prior to cutoff timestamp $T_{t+1}$** ($(T_t + 1\text{h}) - (T_t - 22\text{h}) = 23\text{ hours}$).
- For B3, the lookup target $Y_{t-167}$ corresponds to measurement interval $[T_t - 167\text{h}, T_t - 166\text{h})$. This observation concluded at $T_t - 166\text{h}$, which is **166 hours prior to forecast origin $T_t$** and **167 hours prior to cutoff timestamp $T_{t+1}$** ($(T_t + 1\text{h}) - (T_t - 166\text{h}) = 167\text{ hours}$).
- **Anti-Leakage Verdict:** All predecessor inputs are strictly causal, finalized well before forecast origin $T_t$, and strictly adhere to the no-lookahead invariant. Zero access to future observations $T > T_{t+1}$.

---

## 3. Cohort Retention Audit & Selection Bias Analysis

### Empirical Retention Breakdown by Split

| Split | Original Eligible Instances | Valid B2 Matches ($Y_{t-23}$) | Valid B3 Matches ($Y_{t-167}$) | **Common Cohort Evaluated ($N$)** | Retention % | Excluded Instances | Exclusion % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 618,735 | 538,556 (87.04%) | 513,032 (82.92%) | **461,294** | **74.55%** | 157,441 | 25.45% |
| **VAL** | 152,833 | 129,745 (84.89%) | 124,996 (81.79%) | **111,607** | **73.03%** | 41,226 | 26.97% |

### Exclusion Detail Breakdown

| Exclusion Category | TRAIN Rows | TRAIN % of Eligible | VAL Rows | VAL % of Eligible |
| :--- | :---: | :---: | :---: | :---: |
| **Missing B2 Only** (Valid B3 present, B2 missing/outage) | 51,738 | 8.36% | 13,389 | 8.76% |
| **Missing B3 Only** (Valid B2 present, B3 missing/outage) | 77,262 | 12.49% | 18,138 | 11.87% |
| **Missing Both B2 & B3** (Both predecessors missing/outage) | 28,441 | 4.60% | 9,699 | 6.35% |
| **Total Excluded** | **157,441** | **25.45%** | **41,226** | **26.97%** |

### Empirical Root Cause Decomposition of Predecessor Lookup Misses

Rather than speculating on outage attribution, the underlying feature store and Gold telemetry provide concrete source-data evidence explaining why historical predecessors were unavailable:

#### Validation Split Predecessor Miss Analysis ($N_{\text{excluded}} = 41,226$)

1. **B2 Predecessor Misses ($N = 23,088$ total):**
   - **Absent Historical Timestamp Gaps:** **17,704 instances (76.68%)** had no feature store row at $(T - 24\text{h})$. In the underlying telemetry, these correspond to transmission drops, packet gaps, or unmetered periods where no Gold record was constructed.
   - **Historical Outage Records:** **5,027 instances (21.77%)** had an existing feature store row at $(T - 24\text{h})$, but its historical target was flagged as a grid power outage (`target_is_outage == 1`). Per project contract rules, outage zero-consumption reflects power unavailablity rather than true consumer demand habits and must not be used as a seasonal demand prediction.
   - **Incomplete / Invalid Records:** **357 instances (1.55%)** had a historical record, but failed data-quality validation (`target_is_valid == 0` or null/negative/non-finite).

2. **B3 Predecessor Misses ($N = 27,837$ total):**
   - **Absent Historical Timestamp Gaps:** **21,395 instances (76.86%)** had no feature store row at $(T - 168\text{h})$.
   - **Historical Outage Records:** **5,998 instances (21.55%)** had a historical record with `target_is_outage == 1`.
   - **Incomplete / Invalid Records:** **444 instances (1.59%)** had a historical record with `target_is_valid == 0` or null/negative/non-finite.

### Selection Bias & Domain Implications
1. **Source Data Reality:** In real-world semi-urban Indian distribution grids (MVVNL Bareilly and DVVNL Mathura), physical power outages and telemetry communication drops occur periodically. Over three-quarters of baseline exclusions stem from physical or telemetry data gaps at the specific historical offset, with roughly one-fifth attributable to historical outages.
2. **Selection Bias Profile:** Because inclusion requires valid normal supply during the preceding daily and weekly seasonal anchors, the common evaluation cohort represents periods of **relative grid stability**. Instances occurring immediately after multi-hour grid blackouts (where power was absent 24 hours or 7 days prior) are underrepresented.
3. **Methodological Trade-Off Justification:** Evaluating on the strict common intersection ($N = 111,607$ on VAL) ensures that every comparison between B1, B2, B3, and B4 is conducted on **identical physical events**. Arbitrary zero-imputation or fallback to misaligned lags would distort baseline error distributions and violate Section 6 contract invariants.

---

## 4. Benchmark Baseline Evaluation Results

All metrics were computed via distributed PySpark aggregations adhering strictly to the contract formulations (Section 3 of [`docs/SYSTEM_A_MODELING_CONTRACT.md`](file:///c:/Users/kodal/BDA-ML/docs/SYSTEM_A_MODELING_CONTRACT.md)). Mean Absolute Percentage Error (MAPE) was explicitly omitted due to near-zero standby division instabilities.

### 1. Training Set Evaluation (TRAIN Split)
- **Evaluated Common Cohort ($N$):** 461,294 rows  
- **Target Demand ($\bar{Y}$):** Mean = 0.3869 kWh | Std = 0.4591 kWh | Min = 0.0000 kWh | Max = 5.1090 kWh | Total Energy = 178,459.81 kWh

| Baseline ID & Name | RMSE (kWh) | MAE (kWh) | $R^2$ | WAPE (%) | NRMSE (%) | Sum of Squared Errors (SSE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1: Naive Persistence** (`lag_0h`) | **0.2686** | **0.1254** | **0.6577** | **32.42%** | **69.43%** | 33,276.93 |
| **B2: Diurnal Seasonal** ($Y_{t-23}$) | 0.3023 | 0.1482 | 0.5664 | 38.30% | 78.14% | 42,153.43 |
| **B4: Rolling 24h Mean** (`rolling_mean_24h`) | 0.3055 | 0.1725 | 0.5571 | 44.58% | 78.97% | 43,057.68 |
| **B3: Weekly Seasonal** ($Y_{t-167}$) | 0.3500 | 0.1771 | 0.4186 | 45.77% | 90.48% | 56,518.77 |

---

### 2. Validation Set Evaluation (VAL Split — Primary Benchmark Gate)
- **Evaluated Common Cohort ($N$):** 111,607 rows  
- **Target Demand ($\bar{Y}$):** Mean = 0.3278 kWh | Std = 0.3974 kWh | Min = 0.0000 kWh | Max = 4.2180 kWh | Total Energy = 36,583.45 kWh

| Baseline ID & Name | RMSE (kWh) | MAE (kWh) | $R^2$ | WAPE (%) | NRMSE (%) | Sum of Squared Errors (SSE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1: Naive Persistence** (`lag_0h`) | **0.2594** | **0.1196** | **0.5740** | **36.48%** | **79.12%** | 7,506.93 |
| **B4: Rolling 24h Mean** (`rolling_mean_24h`) | 0.2727 | 0.1534 | 0.5289 | 46.81% | 83.20% | 8,301.78 |
| **B2: Diurnal Seasonal** ($Y_{t-23}$) | 0.2799 | 0.1319 | 0.5039 | 40.25% | 85.38% | 8,742.52 |
| **B3: Weekly Seasonal** ($Y_{t-167}$) | 0.3226 | 0.1612 | 0.3408 | 49.16% | 98.42% | 11,616.27 |

---

## 5. Domain Analysis & Machine Learning Performance Thresholds

### Physical Baseline Hierarchy
1. **Dominance of Short-Term Demand Inertia (B1):**
   - Naive persistence ($\hat{Y}_{t+1} = Y_t$) is the top-performing baseline across both splits (VAL RMSE: **0.2594 kWh**, MAE: **0.1196 kWh**, $R^2$: **0.5740**).
   - In single-meter residential load forecasting, human appliance activity (e.g., air conditioning running, lighting active) exhibits strong hour-to-hour persistence.
2. **Diurnal Rhythms vs Moving Averages (B2 vs B4):**
   - On the validation set, Rolling 24h Mean (B4) achieves lower quadratic error (RMSE: **0.2727 kWh**) than Diurnal Seasonal Naive (B2, RMSE: **0.2799 kWh**), but Diurnal Seasonal achieves lower linear error (MAE: **0.1319 kWh** vs **0.1534 kWh**).
   - This occurs because B4 dampens high-variance transient spikes, while B2 accurately tracks the hour-of-day shape but suffers from daily load level shifts.
3. **Weekly Decay (B3):**
   - Weekly Seasonal Naive exhibits the highest error (VAL RMSE: **0.3226 kWh**, $R^2$: **0.3408**). A 168-hour temporal distance introduces substantial weather and temperature drift between consecutive weeks.

### Production Viability Hurdle for Phase 6.3 & Phase 6.4 ML Models
Per Section 2 of [`docs/SYSTEM_A_MODELING_CONTRACT.md`](file:///c:/Users/kodal/BDA-ML/docs/SYSTEM_A_MODELING_CONTRACT.md), in order to be declared viable, any candidate MLlib regression model (Linear Regression, Ridge, ElasticNet, GBDT, Random Forest) must beat **all four baselines on the Validation set**:

$$\mathbf{\text{Target Hurdle (VAL):}}\quad \text{RMSE} < 0.2594\text{ kWh} \quad \text{and} \quad \text{MAE} < 0.1196\text{ kWh}$$

Any model failing to exceed $R^2 = 0.5740$ on the validation cohort fails the baseline benchmark gate.

---

## 6. Invariant & Test Verification Matrix

A dedicated automated test suite was authored in [`tests/test_baselines.py`](file:///c:/Users/kodal/BDA-ML/tests/test_baselines.py) to guarantee all contract invariants.

| Test Function | Verification Scope | Status |
| :--- | :--- | :---: |
| `test_01_b1_lag0_identity` | Verifies B1 prediction matches `lag_0h` identically with 0 diff rows on Feature Store. | ✅ PASSED |
| `test_02_b4_rolling24_identity` | Verifies B4 prediction matches `rolling_mean_24h` identically with 0 diff rows. | ✅ PASSED |
| `test_03_b2_temporal_alignment_real` | Verifies B2 prediction strictly matches historical target at $T - 24\text{h}$ for the same meter on actual data. | ✅ PASSED |
| `test_04_b3_temporal_alignment_real` | Verifies B3 prediction strictly matches historical target at $T - 168\text{h}$ for the same meter on actual data. | ✅ PASSED |
| `test_05_cross_meter_isolation_synthetic` | Rigorously proves join keys isolate meters: identical timestamps across different meters never cross-contaminate. | ✅ PASSED |
| `test_06_exact_timestamp_offsets_synthetic` | Proves synthetic offsets at $T - 23\text{h}$ and $T - 25\text{h}$ are strictly rejected; only exact $T - 24\text{h}$ aligns for B2. | ✅ PASSED |
| `test_07_missing_predecessors_synthetic` | Proves missing history yields null predictions and is cleanly excluded from the common cohort without crash or zero-fill. | ✅ PASSED |
| `test_08_invalid_and_nonfinite_target_rejection_synthetic` | Proves historical targets with NaN, $\pm\infty$, negative values, or `target_is_valid == 0` are strictly rejected. | ✅ PASSED |
| `test_09_no_join_fan_out` | Verifies join produces zero row fan-out ($N_{\text{aligned}} == N_{\text{eligible}} = 978,031$). | ✅ PASSED |
| `test_10_cross_boundary_history_retrieval` | Verifies VAL rows can legitimately look back into late TRAIN rows for 24h/168h historical targets. | ✅ PASSED |
| `test_11_test_split_evaluation_isolation` | Verifies TEST split target instances are excluded upfront and TEST metrics are absent from output JSON. | ✅ PASSED |
| `test_12_common_cohort_uniformity_and_finite_safety` | Verifies common evaluation cohort has identical row count, zero nulls, and strictly finite non-negative values for all 4 baselines. | ✅ PASSED |
| `test_13_metric_mathematical_invariants` | Verifies domain mathematical axioms: $\text{RMSE} \ge \text{MAE} \ge 0$, $R^2 \le 1.0$, $\text{WAPE} \ge 0$, $\text{NRMSE} \ge 0$. | ✅ PASSED |
| `test_14_primary_key_integrity_rejection_synthetic` | Rigorously proves duplicate primary keys and null primary key fields in lookup source are detected and rejected with ValueError, preventing silent join fan-out and dropped matches. | ✅ PASSED |

### Full Repository Regression Status
- **Baseline Invariant Tests:** 14/14 Passed
- **Feature Store Invariant Tests:** 13/13 Passed
- **Gold Canonical Tests:** 13/13 Passed
- **Silver Ingestion Tests:** 7/7 Passed
- **Total Test Suite:** **47/47 Passed (100% Green, 0 Failures)**

---

## 7. Deliverables & Git Status

1. **Evaluator Script:** [`src/evaluate_baselines.py`](file:///c:/Users/kodal/BDA-ML/src/evaluate_baselines.py)
2. **Test Suite:** [`tests/test_baselines.py`](file:///c:/Users/kodal/BDA-ML/tests/test_baselines.py)
3. **Structured Metrics JSON:** [`reports/baselines/baseline_metrics.json`](file:///c:/Users/kodal/BDA-ML/reports/baselines/baseline_metrics.json)
4. **Documentation Report:** [`docs/BASELINE_EVALUATION_REPORT.md`](file:///c:/Users/kodal/BDA-ML/docs/BASELINE_EVALUATION_REPORT.md)

**Git Working Tree Status:** Untracked files present (`git add` pending user review). Zero commits or pushes performed.
