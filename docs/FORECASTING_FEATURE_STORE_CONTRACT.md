# Phase 5.1: Forecasting Feature Store Contract & Specification
### System A: Supervised Autoregressive Demand Forecasting Feature Store

**Contract Version:** 1.0.0  
**Storage Target:** `data/processed/features_forecasting/` (Partitioned Parquet, Snappy Compressed)  
**Input Source:** `data/processed/canonical_hourly.parquet` (Gold Layer, 1,074,001 rows)  
**Target Engine:** Apache Spark 3.5+ / 4.x & PySpark MLlib  
**Status:** 📋 DESIGN FROZEN (Pre-Implementation Specification)

---

## 1. System Vision & Problem Formulation

System A transforms the hourly canonical smart-meter telemetry into a high-throughput, leakage-safe feature store for distributed supervised regression modeling.

### Mathematical Grain & Prediction Origin
* **Primary Observation Grain:** Exactly one meter $\times$ one clock hour:
  $$\text{Grain} = (district, meter\_id, window\_start)$$
  where $window\_start = T_t$ represents the 1-hour interval $[T_t, T_t + 1\text{h})$.
* **Prediction Origin ($t$):** The cutoff boundary at the conclusion of hour $t$ ($T_{t+1}$). At this point in time, all electrical measurements and quality flags for hour $t$ and prior ($\tau \le t$) are fully finalized and recorded.
* **Canonical Forecast Target ($Y_{t+1}$):** Active energy consumption for the subsequent clock-hour $[T_{t+1}, T_{t+2})$:
  $$\boxed{Y_{t+1} = \text{hourly\_kwh}_{t+1}}$$
* **Multi-Horizon Roadmap:** While the canonical feature pipeline first materializes the immediate step-ahead target ($t+1$), the feature extraction vector $\mathbf{X}_t$ remains mathematically identical and directly extensible to multi-horizon targets ($t+24$ day-ahead, $t+168$ week-ahead).

```
TIME AXIS ─────────────────────────────────────────────────────────────────────────────►
                  [ ... t-24 ... t-2 ... t-1 ]    [      t      ] │ [     t+1     ]
                                                  ▲               │ ▲
Historical Context & Lags ────────────────────────┘               │ │
Known Power Quality Metrics at Cutoff ────────────────────────────┘ │
PREDICTION ORIGIN (Cutoff at T_{t+1}) ─────────────────────────────┘ │
FORECAST TARGET Y_{t+1} ────────────────────────────────────────────┘
```

---

## 2. Strict Anti-Leakage Axioms & Temporal Causality

To ensure strict production validity and prevent data leakage, all feature computations must satisfy three mathematical axioms:

### Axiom 1: Temporal Causality (Past-Only Feature State)
Every feature $f \in \mathbf{X}_t$ must be computable using information available strictly at or before prediction cutoff time $T_{t+1}$:
$$\mathbf{X}_t = \phi\Big( \big\{ Y_\tau, V_\tau, I_\tau, \text{Freq}_\tau, \text{Flags}_\tau \big\}_{\tau \le t}, \text{Calendar}(T_{t+1}) \Big)$$

### Axiom 2: Target & Future Exogenous Independence
No measurement or quality indicator from horizon $t+1$ or later ($\tau \ge t+1$) may enter feature vector $\mathbf{X}_t$. Specifically:
* Target voltage $V_{t+1}$, current $I_{t+1}$, frequency $\text{freq}_{t+1}$, and outage ratio $\text{outage\_ratio}_{t+1}$ are **strictly forbidden** from the feature vector.
* Target consumption $Y_{t+1}$ appears **only** as the supervised regression label (`target_hourly_kwh`).

### Axiom 3: Strict Meter Spatial Isolation
All Window operations (lag shifts, rolling statistics) must be partitioned strictly by `meter_id` (and `district`). Under no circumstances may a Window operation aggregate across different meters or mix meter histories.

---

## 3. Feature Taxonomy & Mathematical Definitions

The feature store is organized into six functional groups totaling 24 features, 5 metadata/identifiers, 4 target/eligibility flags, and 1 split assignment.

```
                      FEATURE STORE SCHEMA TAXONOMY
                                    │
    ┌──────────────┬────────────────┼──────────────┬──────────────┐
    ▼              ▼                ▼              ▼              ▼
 GROUP A        GROUP B          GROUP C        GROUP D        GROUP E & F
Demand Lags   Rolling Stats    Calendar/Fourier Power Quality   Identifiers,
(t, t-1, t-2, (24h Mean, Std,  (Cyclical Sine/  (Voltage, PQ,   Targets &
t-24, t-168)   Min, Max)        Cosine Vector)   Outage Ratio)   Eligibility
```

---

### Group A — Autoregressive Demand Lags
Captures short-term inertia, daily periodicity, and weekly seasonality of active power demand. All lags represent historical hourly energy consumption ($Y_\tau$) known at cutoff $t$:

| Feature Name | Mathematical Definition | Temporal Offset | Intuition / Physical Signal |
| :--- | :--- | :--- | :--- |
| `lag_0h` (or `kwh_t`) | $Y_t$ | Current hour $t$ | Immediate baseline consumption leading into forecast |
| `lag_1h` | $Y_{t-1}$ | $t - 1\text{h}$ (1h before $t$) | 1-step autoregressive lag / immediate velocity |
| `lag_2h` | $Y_{t-2}$ | $t - 2\text{h}$ (2h before $t$) | 2-step autoregressive lag / short-term acceleration |
| `lag_24h` | $Y_{t-24}$ | $t - 24\text{h}$ (Same hour yesterday) | Strong diurnal cycle (24-hour daily periodicity) |
| `lag_168h` | $Y_{t-168}$ | $t - 168\text{h}$ (Same hour last week) | Strong weekly cycle (168-hour day-of-week seasonality) |

*Note on PySpark Window Implementation:*
```python
w_meter = Window.partitionBy("district", "meter_id").orderBy("window_start")
# lag_0h is the current row's hourly_kwh (Y_t)
# lag_1h = F.lag("hourly_kwh", 1).over(w_meter)
# lag_2h = F.lag("hourly_kwh", 2).over(w_meter)
# lag_24h = F.lag("hourly_kwh", 24).over(w_meter)
# lag_168h = F.lag("hourly_kwh", 168).over(w_meter)
```

---

### Group B — Rolling Window Demand Statistics
Captures local demand trend and volatility over the preceding 24 hours. Computed over the backward-looking closed interval $[t-23, t]$ (inclusive of $t$, zero access to $t+1$):

| Feature Name | Mathematical Definition | Window Range | Description |
| :--- | :--- | :--- | :--- |
| `rolling_mean_24h` | $\mu_{24}(t) = \frac{1}{24} \sum_{i=0}^{23} Y_{t-i}$ | $[t-23, t]$ (24 hours) | 24-hour moving average baseline demand |
| `rolling_std_24h` | $\sigma_{24}(t) = \sqrt{\frac{1}{23} \sum_{i=0}^{23} (Y_{t-i} - \mu_{24}(t))^2}$ | $[t-23, t]$ (24 hours) | 24-hour demand dispersion / consumer volatility |
| `rolling_max_24h` | $\max_{i=0..23} Y_{t-i}$ | $[t-23, t]$ (24 hours) | 24-hour peak load demand |
| `rolling_min_24h` | $\min_{i=0..23} Y_{t-i}$ | $[t-23, t]$ (24 hours) | 24-hour baseload floor demand |

*Note on PySpark Window Implementation:*
```python
w_rolling_24 = Window.partitionBy("district", "meter_id").orderBy("window_start").rowsBetween(-23, 0)
# rolling_mean_24h = F.mean("hourly_kwh").over(w_rolling_24)
# rolling_std_24h  = F.stddev("hourly_kwh").over(w_rolling_24)
# rolling_max_24h  = F.max("hourly_kwh").over(w_rolling_24)
# rolling_min_24h  = F.min("hourly_kwh").over(w_rolling_24)
```

---

### Group C — Calendar Features & Cyclical Fourier Encodings
Derived deterministically from the timestamp of the forecast target window $T_{t+1}$ ($[T_{t+1}, T_{t+2})$). Continuous sine/cosine transformations preserve cyclic boundary continuity (e.g., $23:00 \rightarrow 00:00$ and Sunday $\rightarrow$ Monday):

| Feature Name | Data Type | Range / Domain | Formulation / Definition |
| :--- | :--- | :--- | :--- |
| `target_hour` | `IntegerType` | $[0, 23]$ | Clock hour of target window $T_{t+1}$ |
| `target_dow` | `IntegerType` | $[1, 7]$ | Day of week ($1=\text{Sunday} \dots 7=\text{Saturday}$ in Spark) |
| `target_month` | `IntegerType` | $[1, 12]$ | Calendar month of target window $T_{t+1}$ |
| `target_is_weekend` | `IntegerType` | $\{0, 1\}$ | $1$ if `target_dow` $\in \{1, 7\}$ (Sat/Sun), else $0$ |
| `hour_sin` | `DoubleType` | $[-1.0, 1.0]$ | $\sin\left(\frac{2\pi \cdot \text{target\_hour}}{24.0}\right)$ |
| `hour_cos` | `DoubleType` | $[-1.0, 1.0]$ | $\cos\left(\frac{2\pi \cdot \text{target\_hour}}{24.0}\right)$ |
| `dow_sin` | `DoubleType` | $[-1.0, 1.0]$ | $\sin\left(\frac{2\pi \cdot (\text{target\_dow} - 1)}{7.0}\right)$ |
| `dow_cos` | `DoubleType` | $[-1.0, 1.0]$ | $\cos\left(\frac{2\pi \cdot (\text{target\_dow} - 1)}{7.0}\right)$ |
| `month_sin` | `DoubleType` | $[-1.0, 1.0]$ | $\sin\left(\frac{2\pi \cdot (\text{target\_month} - 1)}{12.0}\right)$ |
| `month_cos` | `DoubleType` | $[-1.0, 1.0]$ | $\cos\left(\frac{2\pi \cdot (\text{target\_month} - 1)}{12.0}\right)$ |

---

### Group D — Power Quality & Grid Stress Context (at Cutoff $t$)
Provides operational context regarding supply reliability, grid stress, and device behavior during the observation hour $t$:

| Feature Name | Source Column in Gold | Range / Unit | Description |
| :--- | :--- | :--- | :--- |
| `mean_voltage_t` | `mean_voltage` at $t$ | $[0.0, 700.0]\text{ V}$ | Average line voltage leading up to prediction origin |
| `mean_current_t` | `mean_current` at $t$ | $[0.0, 150.0]\text{ A}$ | Average line current leading up to prediction origin |
| `mean_freq_t` | `mean_freq` at $t$ | $[40.0, 60.0]\text{ Hz}$ | Average grid frequency leading up to prediction origin |
| `outage_ratio_t` | `outage_ratio` at $t$ | $[0.0, 1.0]$ | Outage fraction in hour $t$ |
| `standby_ratio_t` | `standby_ratio` at $t$ | $[0.0, 1.0]$ | Voluntary appliance standby fraction in hour $t$ |
| `brownout_ratio_t` | `brownout_ratio` at $t$ | $[0.0, 1.0]$ | Brownout supply fraction in hour $t$ |
| `spike_ratio_t` | `spike_ratio` at $t$ | $[0.0, 1.0]$ | Overvoltage stress fraction in hour $t$ |
| `rolling_outage_ratio_24h` | Backward 24h Mean | $[0.0, 1.0]$ | Cumulative 24-hour historical outage burden |

---

### Group E — Spatial & Identity Encodings
* **`district`:** Text string (`Bareilly` or `Mathura`).
* **`district_idx`:** Binary numerical encoding ($0.0 = \text{Bareilly}$, $1.0 = \text{Mathura}$) for vector assembly in MLlib.
* **`meter_id`:** Kept as partition key and metadata for cohort grouping, slicing, and evaluation. Excluded from continuous regression weights in the global model to prevent overfitting to specific meter IDs.

---

### Group F — Target Variables & Supervised Eligibility Flags

```
                           TARGET ELIGIBILITY LOGIC
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
     TARGET VALIDITY (t+1)                         HISTORY VALIDITY (≤ t)
     - Complete Hour (sample >= 18)                - lag_1h, lag_2h NOT NULL
     - Normal Supply (outage < 50%)                - lag_24h, lag_168h NOT NULL
     (target_is_valid == 1)                        (has_complete_history == 1)
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      │
                                      ▼
                        is_valid_forecast_instance == 1
                     (Eligible Supervised Training Subset)
```

| Field Name | Type | Range / Domain | Logical Formulation | Description |
| :--- | :--- | :--- | :--- | :--- |
| `target_hourly_kwh` | `DoubleType` | $[0.0, \infty)\text{ kWh}$ | `hourly_kwh` at $t+1$ | Supervised regression label ($Y_{t+1}$) |
| `target_is_complete` | `IntegerType` | $\{0, 1\}$ | `is_complete_hour` at $t+1$ | $1$ if target hour has $\ge 18$ samples |
| `target_is_outage` | `IntegerType` | $\{0, 1\}$ | `is_outage_hour` at $t+1$ | $1$ if target hour had $\ge 50\%$ outage |
| `target_is_valid` | `IntegerType` | $\{0, 1\}$ | `is_valid_forecast_target` at $t+1$ | $1$ if complete and non-outage ($1$ and $0$) |
| `has_complete_history` | `IntegerType` | $\{0, 1\}$ | $\prod_{k \in \{0, 1, 2, 24, 168\}} \mathbb{I}(\text{lag\_kh} \neq \text{null})$ | $1$ if all required historical lags exist |
| `is_valid_forecast_instance` | `IntegerType` | $\{0, 1\}$ | `has_complete_history == 1` $\land$ `target_is_valid == 1` | **Canonical Supervised Training Filter** |

---

## 4. Missing History & Cold-Start Policy

1. **Initial Deployment Cold Start:** The first 168 hours of each meter's time series will naturally have `null` values for `lag_168h` (and first 24h for `lag_24h`).
2. **Multi-Day Telemetry Disruptions:** When a meter drops out for $>24$ hours, subsequent rows upon re-activation will lack valid short-term lags.
3. **Zero-Imputation Prohibition (Strict Invariant):**
   > **Missing lag timestamps MUST NEVER be imputed with `0.0` or mean values.**
   Filling a missing 168h lag with `0.0` creates artificial, non-physical consumption collapses and destroys autoregressive parameter estimation.
4. **Segregation via Metadata Flags:**
   * Rows with incomplete history are retained in the feature store with explicit `null` fields to preserve longitudinal continuity.
   * They receive `has_complete_history = 0` and `is_valid_forecast_instance = 0`.
   * Downstream ML training sets select `WHERE is_valid_forecast_instance = 1`.

---

## 5. Chronological Partitioning & Cross-Boundary Rules

To prevent temporal lookahead bias, all data partitions adhere strictly to the frozen chronological split boundaries based on target timestamp $T_{t+1}$:

```
├── TRAIN SET:      2019-05-01 00:00:00 → 2020-08-31 23:59:59  (16 Months / ~55% timeline)
├── VALIDATION SET: 2020-09-01 00:00:00 → 2020-12-31 23:59:59  ( 4 Months / ~14% timeline)
└── TEST SET:       2021-01-01 00:00:00 → 2021-10-31 23:59:59  (10 Months / ~31% timeline)
```

### The Cross-Boundary Feature Invariant
* **Lag Computation Precedes Split Assignment:** Lags and rolling statistics are computed on the complete, contiguous time series per meter *before* tagging split partitions.
* **Backward Historical Crossing (Permitted & Required):** The first validation instance (`2020-09-01 00:00:00`) legitimately references lag features from August 31, 2020 (Train set). This reflects real-world operational inference where past history is known.
* **Forward Leakage (Strictly Prohibited):** Under no circumstance may any calculation for an instance at $t$ access data from $\tau > t$.

---

## 6. Formal Parquet Schema Contract (`features_forecasting/`)

The feature store is materialized to Parquet partitioned by `split` and `district`:

| Field Name | Spark DataType | Nullable in Store | Nullable in Model Set | Source / Formulation |
| :--- | :--- | :--- | :--- | :--- |
| `district` | `StringType` | `false` | `false` | Gold: `district` |
| `meter_id` | `StringType` | `false` | `false` | Gold: `meter_id` |
| `window_start` | `TimestampType` | `false` | `false` | Gold: `window_start` ($T_t$, Cutoff Reference) |
| `target_window_start` | `TimestampType` | `false` | `false` | Gold: `window_start` at $t+1$ ($T_{t+1}$) |
| `split` | `StringType` | `false` | `false` | `'TRAIN'`, `'VAL'`, or `'TEST'` |
| `district_idx` | `DoubleType` | `false` | `false` | `0.0` (Bareilly) \| `1.0` (Mathura) |
| `lag_0h` | `DoubleType` | `false` | `false` | $Y_t$ (Current hour energy) |
| `lag_1h` | `DoubleType` | `true` | `false` | $Y_{t-1}$ (1-hour lag) |
| `lag_2h` | `DoubleType` | `true` | `false` | $Y_{t-2}$ (2-hour lag) |
| `lag_24h` | `DoubleType` | `true` | `false` | $Y_{t-24}$ (24-hour lag) |
| `lag_168h` | `DoubleType` | `true` | `false` | $Y_{t-168}$ (168-hour lag) |
| `rolling_mean_24h` | `DoubleType` | `true` | `false` | Backward 24h average $[t-23, t]$ |
| `rolling_std_24h` | `DoubleType` | `true` | `false` | Backward 24h stddev $[t-23, t]$ |
| `rolling_max_24h` | `DoubleType` | `true` | `false` | Backward 24h max $[t-23, t]$ |
| `rolling_min_24h` | `DoubleType` | `true` | `false` | Backward 24h min $[t-23, t]$ |
| `target_hour` | `IntegerType` | `false` | `false` | Hour of $T_{t+1}$ $[0..23]$ |
| `target_dow` | `IntegerType` | `false` | `false` | Day of week of $T_{t+1}$ $[1..7]$ |
| `target_month` | `IntegerType` | `false` | `false` | Month of $T_{t+1}$ $[1..12]$ |
| `target_is_weekend` | `IntegerType` | `false` | `false` | Binary $\{0, 1\}$ |
| `hour_sin` | `DoubleType` | `false` | `false` | $\sin(2\pi \cdot \text{target\_hour} / 24.0)$ |
| `hour_cos` | `DoubleType` | `false` | `false` | $\cos(2\pi \cdot \text{target\_hour} / 24.0)$ |
| `dow_sin` | `DoubleType` | `false` | `false` | $\sin(2\pi \cdot (\text{target\_dow}-1) / 7.0)$ |
| `dow_cos` | `DoubleType` | `false` | `false` | $\cos(2\pi \cdot (\text{target\_dow}-1) / 7.0)$ |
| `month_sin` | `DoubleType` | `false` | `false` | $\sin(2\pi \cdot (\text{target\_month}-1) / 12.0)$ |
| `month_cos` | `DoubleType` | `false` | `false` | $\cos(2\pi \cdot (\text{target\_month}-1) / 12.0)$ |
| `mean_voltage_t` | `DoubleType` | `false` | `false` | Gold: `mean_voltage` at $t$ |
| `mean_current_t` | `DoubleType` | `false` | `false` | Gold: `mean_current` at $t$ |
| `mean_freq_t` | `DoubleType` | `false` | `false` | Gold: `mean_freq` at $t$ |
| `outage_ratio_t` | `DoubleType` | `false` | `false` | Gold: `outage_ratio` at $t$ |
| `standby_ratio_t` | `DoubleType` | `false` | `false` | Gold: `standby_ratio` at $t$ |
| `brownout_ratio_t` | `DoubleType` | `false` | `false` | Gold: `brownout_ratio` at $t$ |
| `spike_ratio_t` | `DoubleType` | `false` | `false` | Gold: `spike_ratio` at $t$ |
| `rolling_outage_ratio_24h` | `DoubleType` | `true` | `false` | Backward 24h mean outage ratio |
| `target_hourly_kwh` | `DoubleType` | `false` | `false` | Gold: `hourly_kwh` at $t+1$ ($Y_{t+1}$) |
| `target_is_complete` | `IntegerType` | `false` | `false` | Gold: `is_complete_hour` at $t+1$ |
| `target_is_outage` | `IntegerType` | `false` | `false` | Gold: `is_outage_hour` at $t+1$ |
| `target_is_valid` | `IntegerType` | `false` | `false` | Gold: `is_valid_forecast_target` at $t+1$ |
| `has_complete_history` | `IntegerType` | `false` | `false` | Binary $\{0, 1\}$ |
| `is_valid_forecast_instance` | `IntegerType` | `false` | `false` | Binary $\{0, 1\}$ |

---

## 7. Phase 5.3 Automated Invariant Acceptance Test Suite Plan

The following 12 automated verification tests will be implemented in `tests/test_feature_store.py` during Phase 5.3:

1. **`test_feature_store_primary_key_uniqueness`**: Verifies 0 duplicate `(district, meter_id, window_start)` records.
2. **`test_feature_store_schema_conformance`**: Asserts exact column name and PySpark DataType alignment against the contract.
3. **`test_autoregressive_lag_alignment`**: Samples random continuous sequences to verify `lag_1h == Y(t-1)`, `lag_2h == Y(t-2)`, `lag_24h == Y(t-24)`, and `lag_168h == Y(t-168)`.
4. **`test_rolling_statistics_mathematical_precision`**: Asserts `rolling_mean_24h` and `rolling_std_24h` match numpy/scipy manual calculations on the 24-hour backward window.
5. **`test_cyclical_fourier_invariants`**: Verifies $\sin^2(\theta) + \cos^2(\theta) = 1.0 \pm 10^{-6}$ for all hour, day-of-week, and month encodings.
6. **`test_anti_leakage_target_independence`**: Proves that target $Y_{t+1}$ and target power quality values do not equal cutoff $t$ features when $Y_{t+1} \neq Y_t$.
7. **`test_chronological_split_boundaries`**: Verifies that 100% of rows in `TRAIN`, `VAL`, and `TEST` partitions strictly respect their calendar date ranges.
8. **`test_cross_boundary_lag_integrity`**: Confirms that validation rows in early September 2020 have valid, non-null historical lags sourced from late August 2020.
9. **`test_zero_imputation_prohibition`**: Verifies that initial cold-start rows retain explicit `null` lags rather than artificial zeros, and receive `has_complete_history == 0`.
10. **`test_supervised_subset_zero_nulls`**: Asserts that filtering by `is_valid_forecast_instance == 1` yields exactly zero null values across all modeling feature columns.
11. **`test_meter_cohort_preservation`**: Confirms that all 84 meters (46 Bareilly, 38 Mathura) are present in the feature store.
12. **`test_energy_and_power_physical_bounds`**: Verifies $Y_{t+1} \ge 0.0$, $\text{lags} \ge 0.0$, $V_t \ge 0.0$, and ratios $\in [0.0, 1.0]$.
