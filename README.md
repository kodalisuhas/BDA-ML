# Smart Meter Big Data Analytics & Machine Learning (BDA-ML)
### Energy Demand Forecasting & Longitudinal Customer Archetype Clustering with Apache Spark

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Apache Spark 4.2](https://img.shields.io/badge/Apache%20Spark-4.2.0-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![PyTest](https://img.shields.io/badge/Tests-33%2F33%20Passing-brightgreen?logo=pytest&logoColor=white)](https://pytest.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Medallion%20(Bronze%2FSilver%2FGold%2FFeatures)-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

An enterprise-grade Big Data and Distributed Machine Learning architecture for large-scale electrical smart meter telemetry. The platform ingests, validates, aggregates, and models **21,394,429 high-frequency telemetry readings** across 84 meters in Uttar Pradesh (Bareilly and Mathura districts, India) between May 2019 and October 2021.

---

## 📌 Executive Architecture & System Overview

The system implements a dual-track machine learning platform on top of an Apache Spark Medallion Data Engineering pipeline:

```
                                  RAW SMART METER TELEMETRY (CSV)
                                   21,394,429 rows | 84 meters
                                                │
                                                ▼
                         ┌─────────────────────────────────────────────┐
                         │       BRONZE LAYER (Data Intake)            │
                         │ Strict StructType Schema | Zero Type Coerce │
                         └──────────────────────┬──────────────────────┘
                                                │
                                                ▼
                         ┌─────────────────────────────────────────────┐
                         │       SILVER LAYER (Data Validation)        │
                         │ 3-Tier Quality Policy | 5-State Taxonomy    │
                         │ Partitioned Parquet: (district, file_year)  │
                         └──────────────┬──────────────────────────────┘
                                        │ (21,392,743 valid rows)
                                        │ (1,686 quarantined rows)
                                        ▼
                         ┌─────────────────────────────────────────────┐
                         │       GOLD LAYER (Canonical Modeling)       │
                         │ 1-Hour Aggregation | 20/N Scaling Filter    │
                         │ Quality & Outage Modeling Flags             │
                         └──────────────┬──────────────────────────────┘
                                        │ (1,074,001 hourly rows)
                                        ▼
                         ┌─────────────────────────────────────────────┐
                         │    PHASE 5: FORECASTING FEATURE STORE       │
                         │ 39 Columns | Autoregressive Lags | 24h Roll │
                         │ Fourier Vectors | Power Quality at Cutoff t │
                         │ Explicit NULL Missing Target Preservation   │
                         └──────────────┬──────────────────────────────┘
                                        │ (1,074,001 rows | 978,031 eligible)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
    ┌───────────────────────────────┐       ┌───────────────────────────────┐
    │           SYSTEM A            │       │           SYSTEM B            │
    │ Supervised Demand Forecasting │       │  Longitudinal Archetype Clust.│
    │ Multi-Horizon (1h, 24h, 168h) │       │   Unsupervised MLlib K-Means  │
    │ Lag Features + Fourier Cycles │       │ Diurnal Load Curves & Metrics │
    └───────────────────────────────┘       └───────────────────────────────┘
```

### 1. System A: Supervised Autoregressive Demand Forecasting
* **Objective:** Predict consumer and feeder-level active energy consumption ($Y_{t+h}$) across multi-step horizons: 1-hour ($t+1$), 24-hour ($t+24$, day-ahead), and 168-hour ($t+168$, week-ahead).
* **Feature Engineering:** Autoregressive lag vectors ($Y_{t-1}, Y_{t-2}, Y_{t-24}, Y_{t-168}$), rolling statistics ($\mu_{24}, \sigma_{24}$), cyclical temporal Fourier encodings ($\sin/\cos$ of hour, day-of-week, month), and power quality stress ratios.
* **Supply-Aware Filtering:** Isolates grid outages ($V=0\text{V}$) via `is_valid_forecast_instance` to prevent corrupting baseline demand models with transmission supply failures.

### 2. System B: Unsupervised Customer Archetype Clustering
* **Objective:** Discover persistent consumer load archetypes for tariff structuring, capacity planning, and demand-response targeting.
* **Feature Engineering:** 84 longitudinal behavioral vectors per meter capturing baseload fraction, peak-to-average ratio (PAR), diurnal morning/evening ramp rates, power factor distributions, and brownout vulnerability indices.
* **Algorithm:** Distributed PySpark MLlib K-Means with Silhouette and Davies-Bouldin optimization across $k \in [3, 8]$.

---

## 🔬 Forensic Discoveries & Engineering Decisions

| Forensic Investigation | Metric / Finding | Engineering Resolution |
| :--- | :--- | :--- |
| **Dataset Scale** | **21,394,429 raw records** across 84 meters over 29 months | Processed via distributed PySpark with zero out-of-memory overhead |
| **Temporal Cadence** | **99.98%** exact 3-minute consecutive intervals ($\Delta t = 180\text{s}$) | Regular time-series confirmed; dropouts are long-duration lifespan changes |
| **Resolution Selection** | SNR improves from **$-3.23\text{ dB} \rightarrow -2.21\text{ dB}$**; Zero-inflation drops **$16.38\% \rightarrow 7.96\%$** | Canonical modeling resolution frozen at **1-Hour** ($1,074,001$ records) |
| **Physics of Zeros** | $8.19\%$ Outages ($V=0\text{V}$) vs $7.12\%$ Standby ($V\ge 180\text{V}, I\le 0.05\text{A}$) | Mathematically closed 5-state electrical taxonomy; zero orphan records |
| **Quarantine Audit** | **1,686 records** with non-physical grid frequencies ($f \notin [40, 60]\text{ Hz}$) | Isolated into `quarantine_audit.parquet` with explicit `rejection_reason` |
| **Energy Conservation** | Delta between Silver and Gold: **$\Delta < 0.0001\text{ kWh}$** | $20/N$ linear scaling preserves cumulative energy consumption |
| **Target Preservation** | 4,219 non-consecutive $t+1$ transitions retain explicit `NULL` | Zero synthetic energy fabricated; 978,031 clean eligible training instances |

---

## 📂 Repository Structure

```
BDA-ML/
├── .gitignore                      # Strict exclusion for data, caches, and venv
├── requirements.txt                # Production dependency specification
├── README.md                       # Comprehensive system & architectural guide
│
├── configs/                        # Hyperparameter & pipeline configuration files
├── notebooks/                      # Exploratory Data Analysis & visual notebooks
│
├── src/                            # Production PySpark Pipelines & Utilities
│   ├── dataset_audit.py            # Phase 1: Distributed forensic data profiler
│   ├── temporal_forensics.py       # Phase 2: Cadence & transition delta analyzer
│   ├── signal_and_resolution_forensics.py # Phase 2: SNR & resolution trade-off engine
│   ├── silver_pipeline.py          # Phase 4.1: Silver data cleaner & validator
│   ├── gold_pipeline.py            # Phase 4.2: Hourly aggregator & feature engine
│   ├── feature_pipeline.py         # Phase 5.2: System A feature store materializer
│   └── smoke_test.py               # Environment & JVM configuration smoke test
│
├── tests/                          # Automated Pytest Invariant Test Suite
│   ├── test_silver_layer.py        # 7 automated tests for Silver layer invariants
│   ├── test_gold_layer.py          # 13 automated tests for Gold layer invariants
│   └── test_feature_store.py       # 13 automated tests for Feature Store invariants
│
└── docs/                           # Authoritative Technical Specifications
    ├── CANONICAL_DATA_CONTRACT.md   # Version 1.1 schema contract & Chronological splits
    ├── DATA_QUALITY_POLICY.md       # Version 1.1 validation bounds & state machines
    ├── FORECASTING_FEATURE_STORE_CONTRACT.md # Version 1.0.0 39-column Feature Contract
    ├── DATASET_MANIFEST.md          # File-level row counts, byte sizes, and schemas
    ├── DATASET_AUDIT_REPORT.md      # Comprehensive Phase 1 intake profile
    ├── TEMPORAL_FORENSICS_REPORT.md # Interval cadence & missingness analysis
    ├── SIGNAL_ELECTRICAL_RESOLUTION_REPORT.md # Resolution trade-off analysis
    ├── SILVER_PIPELINE_REPORT.md    # Execution metrics for Silver validation
    ├── GOLD_PIPELINE_REPORT.md      # Execution metrics for Gold aggregation
    └── FEATURE_PIPELINE_REPORT.md   # Execution metrics for Feature Store materialization
```

---

## 🧪 Automated Test Suite (33/33 Passing)

The repository enforces strict data invariant assertions using `pytest`. The test suite verifies:

```powershell
pytest tests/ -v
```

```text
tests/test_feature_store.py::test_01_feature_store_primary_key_uniqueness PASSED [  3%]
tests/test_feature_store.py::test_02_feature_store_exact_schema PASSED   [  6%]
tests/test_feature_store.py::test_03_autoregressive_lag_alignment PASSED [  9%]
tests/test_feature_store.py::test_04_rolling_statistics_mathematical_precision PASSED [ 12%]
tests/test_feature_store.py::test_05_fourier_cyclical_invariants PASSED  [ 15%]
tests/test_feature_store.py::test_06_target_timestamp_alignment PASSED   [ 18%]
tests/test_feature_store.py::test_07_target_independence_anti_leakage PASSED [ 21%]
tests/test_feature_store.py::test_08_chronological_split_correctness PASSED [ 24%]
tests/test_feature_store.py::test_09_cross_boundary_lag_integrity PASSED [ 27%]
tests/test_feature_store.py::test_10_zero_imputation_prohibition PASSED  [ 30%]
tests/test_feature_store.py::test_11_eligible_subset_contains_zero_nulls PASSED [ 33%]
tests/test_feature_store.py::test_12_all_84_meter_cohorts_preserved PASSED [ 36%]
tests/test_feature_store.py::test_13_physical_and_value_bounds PASSED    [ 39%]
tests/test_gold_layer.py::test_gold_row_count PASSED                     [ 42%]
tests/test_gold_layer.py::test_gold_schema_matches_contract PASSED       [ 45%]
tests/test_gold_layer.py::test_gold_zero_nulls PASSED                    [ 48%]
tests/test_gold_layer.py::test_gold_primary_key_uniqueness PASSED        [ 51%]
tests/test_gold_layer.py::test_gold_window_duration PASSED               [ 54%]
tests/test_gold_layer.py::test_gold_sample_count_bounds PASSED           [ 57%]
tests/test_gold_layer.py::test_gold_energy_non_negative PASSED           [ 60%]
tests/test_gold_layer.py::test_gold_complete_20_sample_exact_equality PASSED [ 63%]
tests/test_gold_layer.py::test_gold_scaling_formula_adherence PASSED     [ 66%]
tests/test_gold_layer.py::test_gold_ratios_within_unit_interval PASSED   [ 69%]
tests/test_gold_layer.py::test_gold_flag_logical_consistency PASSED      [ 72%]
tests/test_gold_layer.py::test_gold_energy_conservation_against_silver PASSED [ 75%]
tests/test_gold_layer.py::test_gold_spike_ratio_definition PASSED        [ 78%]
tests/test_silver_layer.py::test_silver_row_count PASSED                 [ 81%]
tests/test_silver_layer.py::test_silver_schema_matches_contract PASSED   [ 84%]
tests/test_silver_layer.py::test_silver_zero_nulls PASSED                [ 87%]
tests/test_silver_layer.py::test_silver_primary_key_uniqueness PASSED    [ 90%]
tests/test_silver_layer.py::test_silver_physical_bounds PASSED           [ 93%]
tests/test_silver_layer.py::test_silver_state_machine_closed PASSED      [ 96%]
tests/test_silver_layer.py::test_silver_flag_logical_consistency PASSED  [100%]

======================== 33 passed in 62.90s (0:01:02) ========================
```

---

## 🚀 Quickstart & Reproduction Guide

### Prerequisites
* **Python:** `3.11.x`
* **Java:** OpenJDK `17` LTS (`JAVA_HOME` configured)
* **Hadoop Binaries (Windows):** `HADOOP_HOME` containing `winutils.exe` and `hadoop.dll` (Hadoop 3.3.5)

### Installation
```bash
# 1. Clone repository
git clone https://github.com/kodalisuhas/BDA-ML.git
cd BDA-ML

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### Pipeline Execution
```bash
# Verify environment & Spark session
python src/smoke_test.py

# Run Silver data validation & quarantine pipeline
python src/silver_pipeline.py

# Run Gold hourly aggregation & feature generation pipeline
python src/gold_pipeline.py

# Run System A Supervised Demand Forecasting Feature Store pipeline
python src/feature_pipeline.py

# Run complete automated verification suite (33 invariant tests)
pytest tests/ -v
```

---

## 📅 Data Partitioning & Split Policy

To prevent lookahead bias and temporal data leakage in time-series forecasting, all data splits adhere to strict chronological boundaries:

```
├── TRAIN SET:      May 01, 2019 → Aug 31, 2020  (16 Months / 681,535 rows / 63.5% timeline)
├── VALIDATION SET: Sep 01, 2020 → Dec 31, 2020  ( 4 Months / 166,793 rows / 15.5% timeline)
└── TEST SET:       Jan 01, 2021 → Oct 31, 2021  (10 Months / 225,673 rows / 21.0% timeline)
```

---

## 📖 Key Project Documents
* [Forecasting Feature Store Contract v1.0.0](docs/FORECASTING_FEATURE_STORE_CONTRACT.md)
* [Feature Pipeline Execution Report](docs/FEATURE_PIPELINE_REPORT.md)
* [Canonical Data Contract v1.1](docs/CANONICAL_DATA_CONTRACT.md)
* [Data Quality & Cleaning Policy v1.1](docs/DATA_QUALITY_POLICY.md)
* [Dataset Manifest & Source Inventory](docs/DATASET_MANIFEST.md)
* [Dataset Intake & Profiling Audit](docs/DATASET_AUDIT_REPORT.md)
* [Temporal Cadence & Transition Forensics](docs/TEMPORAL_FORENSICS_REPORT.md)
* [Signal-to-Noise & Resolution Selection Analysis](docs/SIGNAL_ELECTRICAL_RESOLUTION_REPORT.md)
* [Silver Pipeline Execution Report](docs/SILVER_PIPELINE_REPORT.md)
* [Gold Pipeline Execution Report](docs/GOLD_PIPELINE_REPORT.md)

---

## 👤 Author & Maintainer
* **Author:** Suhas Kodali
* **Repository:** [https://github.com/kodalisuhas/BDA-ML](https://github.com/kodalisuhas/BDA-ML)

