# Phase 3B: Canonical Data Contract Specification (Reconciled v1.1)

**Contract Version:** 1.1.0  
**Storage Format:** Apache Parquet (Snappy Compressed)  
**Encoding:** Columnar, Strongly Typed  
**Engine Compatibility:** Apache Spark 3.5+ / 4.x  

---

## 1. Medallion Storage Architecture

```
  data/raw/ (Immutable Source Layer)
  ├── 6 District CSVs (3-min interval, ~21.4M rows)
  └── 2 District Aggregated CSVs (Reference totals)
        │
        ▼ (Spark Ingestion + Strict StructType Validation + 5-State Tagging)
  data/interim/validated_3min.parquet (Silver Layer)
  └── Partitioned by: district, file_year
        │
        ▼ (Spark SQL Clock-Hour Aggregation + Outage Ratios + Quality Flags)
  data/processed/canonical_hourly.parquet (Gold Layer)
  └── Partitioned by: district, file_year
        │
        ├───────────────────────────────────────┐
        ▼                                       ▼
  System A: Forecasting Feature Store     System B: Customer Behavioral Matrix
  (`data/processed/features_forecasting/`) (`data/processed/features_clustering/`)
```

---

## 2. Layer 1: Silver Schema (`validated_3min.parquet`)

Represents clean, validated, atomic 3-minute observations with physical operating state tags.

| Field Name | Spark DataType | Nullable | Physical Range / Unit | Description & Invariants |
|---|---|---|---|---|
| `district` | `StringType` | `false` | `Bareilly` \| `Mathura` | District identifier |
| `meter_id` | `StringType` | `false` | Text (e.g. `BR02`, `MH15`) | Unique meter identifier |
| `ts` | `TimestampType` | `false` | IST (UTC+5:30) | Measurement timestamp (3-minute cadence) |
| `file_year` | `IntegerType` | `false` | `2019`, `2020`, `2021` | Source file year |
| `t_kwh` | `DoubleType` | `false` | $[0.0, 0.500]\text{ kWh}$ | Active energy consumed in 3-min interval |
| `voltage` | `DoubleType` | `false` | $[0.0, 700.0]\text{ V}$ | RMS line voltage (Preserves extreme spikes) |
| `current` | `DoubleType` | `false` | $[0.0, 150.0]\text{ A}$ | RMS line current (Preserves heavy inductive draw) |
| `freq` | `DoubleType` | `false` | $[0.0, 60.0]\text{ Hz}$ | Supply frequency |
| `operating_state` | `IntegerType` | `false` | Enum $[1..5]$ | $1=\text{Outage}, 2=\text{Standby}, 3=\text{Brownout}, 4=\text{Normal}, 5=\text{Overvoltage}$ |
| `is_outage` | `IntegerType` | `false` | Binary $\{0,1\}$ | $1$ if $V = 0.0\text{V}$, else $0$ |
| `is_standby` | `IntegerType` | `false` | Binary $\{0,1\}$ | $1$ if $V \ge 180\text{V}, I \le 0.05\text{A}, \text{kWh}=0$, else $0$ |
| `is_brownout` | `IntegerType` | `false` | Binary $\{0,1\}$ | $1$ if $0 < V < 180\text{V}$, else $0$ |
| `is_voltage_high` | `IntegerType` | `false` | Binary $\{0,1\}$ | $1$ if $260\text{V} < V \le 300\text{V}$, else $0$ |
| `is_voltage_spike` | `IntegerType` | `false` | Binary $\{0,1\}$ | $1$ if $V > 300\text{V}$, else $0$ |
| `is_extreme_load` | `IntegerType` | `false` | Binary $\{0,1\}$ | $1$ if $t\_\text{kWh} > 0.300\text{ kWh}$, else $0$ |

---

## 3. Layer 2: Gold Schema (`canonical_hourly.parquet`)

The canonical modeling foundation aggregated over clock-hour buckets (`window(ts, '1 hour')`).

| Field Name | Spark DataType | Nullable | Physical Range / Unit | Formulation / Aggregation Logic |
|---|---|---|---|---|
| `district` | `StringType` | `false` | Text | District identifier (`Bareilly` or `Mathura`) |
| `meter_id` | `StringType` | `false` | Text | Unique meter identifier |
| `window_start` | `TimestampType` | `false` | IST | Start of clock-hour (e.g. `2020-05-01 14:00:00`) |
| `window_end` | `TimestampType` | `false` | IST | End of clock-hour (e.g. `2020-05-01 15:00:00`) |
| `file_year` | `IntegerType` | `false` | `2019`, `2020`, `2021` | Calendar year of window start |
| `hourly_kwh` | `DoubleType` | `false` | $\text{kWh}$ | Sum of 3-min `t_kwh` scaled by $(20 / \text{sample\_count})$ |
| `raw_kwh_sum` | `DoubleType` | `false` | $\text{kWh}$ | Raw unscaled sum of observed `t_kwh` |
| `sample_count` | `IntegerType` | `false` | Integer $[1..20]$ | Number of 3-min samples observed in this hour |
| `mean_voltage` | `DoubleType` | `false` | $\text{Volts (RMS)}$ | Arithmetic mean of voltage readings |
| `mean_current` | `DoubleType` | `false` | $\text{Amperes (RMS)}$ | Arithmetic mean of current readings |
| `mean_freq` | `DoubleType` | `false` | $\text{Hertz (Hz)}$ | Arithmetic mean of supply frequency |
| `outage_ratio` | `DoubleType` | `false` | Fraction $[0..1]$ | Fraction of 3-min readings where $V=0\text{V}$ |
| `standby_ratio` | `DoubleType` | `false` | Fraction $[0..1]$ | Fraction of 3-min readings in voluntary standby |
| `brownout_ratio` | `DoubleType` | `false` | Fraction $[0..1]$ | Fraction of 3-min readings where $0 < V < 180\text{V}$ |
| `spike_ratio` | `DoubleType` | `false` | Fraction $[0..1]$ | Fraction of 3-min readings where $V > 270\text{V}$ |
| `is_complete_hour` | `IntegerType` | `false` | Binary $\{0,1\}$ | $1$ if $\text{sample\_count} \ge 18$ ($\ge 90\%$), else $0$ |
| `is_outage_hour` | `IntegerType` | `false` | Binary $\{0,1\}$ | $1$ if $\text{outage\_ratio} \ge 0.50$, else $0$ |
| `is_valid_forecast_target` | `IntegerType` | `false` | Binary $\{0,1\}$ | $1$ if `is_complete_hour == 1` and `is_outage_hour == 0` |

---

## 4. Sub-System Data Consumption Contracts

### System A (Forecasting Regression Contract)
* **Target Variable ($Y_t$):** `hourly_kwh` at window $T$.
* **Normal-Supply Demand Subset:** Instances where `is_valid_forecast_target == 1`.
* **Full-System Actual Demand Subset:** All instances where `is_complete_hour == 1` (includes outages with `hourly_kwh = 0.0`).
* **Feature Generation Permitted:**
  * Lags: $Y_{t-1}, Y_{t-2}, Y_{t-24}, Y_{t-168}$
  * Rolling aggregates: $\mu_{24}(Y), \sigma_{24}(Y)$
  * Exogenous power metrics: `mean_voltage`, `mean_current`, `outage_ratio`
  * Cyclical time encodings: $\sin / \cos(\text{hour} / 24)$, $\sin / \cos(\text{day\_of\_week} / 7)$

### System B (Customer Clustering Contract)
* **Aggregation Granularity:** Group by `(district, meter_id)` across the entire active lifespan.
* **Feature Vector ($d$-dimensional):**
  1. Mean daily active load ($\bar{E}_m$)
  2. Median active load ($E_{m,\text{med}}$)
  3. Load volatility / Coefficient of variation ($\text{CV}_m = \sigma_m / \bar{E}_m$)
  4. Peak-to-Average Ratio ($\text{PAR}_m = \max(Y_{m,t}) / \bar{E}_m$)
  5. Diurnal ratio ($\text{Energy}_{\text{06:00-18:00}} / \text{Energy}_{\text{18:00-06:00}}$)
  6. Weekend-to-Weekday ratio ($\bar{E}_{\text{weekend}} / \bar{E}_{\text{weekday}}$)
  7. Power quality stress index: Total lifetime outage hours and brownout fraction ($V < 200\text{V}$).
