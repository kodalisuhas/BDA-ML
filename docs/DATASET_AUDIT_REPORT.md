# Phase 1: Forensic Dataset Audit Report

**Execution Timestamp:** 2026-10-06  
**Engine:** Apache Spark 4.2.0 (`local[*]`, 16 Cores) on OpenJDK 17 LTS  
**Dataset:** CEEW Smart Meter Data (Mathura & Bareilly)  
**Execution Runtime:** 157.51 seconds  

---

## 1. Executive Summary

A full forensic intake audit was executed across all 6 raw interval telemetry files and 2 reference aggregated files. The audit verified row counts, schema uniformity, temporal bounds, meter consistency, null integrity, duplicate keys, and electrical physics distributions.

| Metric | Measured Value | Architectural Significance |
|---|---|---|
| **Total Telemetry Rows** | **21,394,429** | Large-scale dataset confirming Spark distributed processing requirement |
| **Total Unique Meters** | **84** (46 Bareilly, 38 Mathura) | Provides sufficient customer diversity for K-Means behavioral clustering |
| **Temporal Span** | May 1, 2019 – October 31, 2021 | 2.5-year longitudinal panel enabling temporal out-of-time evaluation |
| **Duplicate Keys `(meter, timestamp)`** | **0** | Perfect primary-key uniqueness across all 21.4M rows |
| **Explicit Null / NaN Cells** | **0** | Clean tabular structure (missingness is implicit temporal gaps) |
| **Grid Blackouts ($V=0, f=0$)** | **1,745,585 (8.16%)** | Must be explicitly flagged as `is_outage = 1` during feature engineering |
| **Voluntary Standby ($V>180V, I=0$)** | **1,094,803 (5.12%)** | Retained as true voluntary zero demand |

---

## 2. File-by-File Inventory & Temporal Coverage

| File Name | District | Year | Row Count | Distinct Meters | Earliest Timestamp | Latest Timestamp |
|---|---|---|---|---|---|---|
| `SM Cleaned Data BR2019.csv` | Bareilly | 2019 | 2,919,315 | 46 | 2019-05-09 00:00:00 | 2019-12-31 23:57:00 |
| `CEEW - Smart meter data Bareilly 2020.csv` | Bareilly | 2020 | 6,627,360 | 46 | 2020-01-01 00:00:00 | 2020-12-31 23:57:00 |
| `CEEW - Smart meter data Bareilly 2021.csv` | Bareilly | 2021 | 3,948,960 | 38 | 2021-01-01 00:00:00 | 2021-10-31 23:57:00 |
| `CEEW - Smart meter data Mathura 2019.csv` | Mathura | 2019 | 3,588,874 | 38 | 2019-05-01 00:00:00 | 2019-12-31 23:57:00 |
| `CEEW - Smart meter data Mathura 2020.csv` | Mathura | 2020 | 3,759,360 | 38 | 2020-01-01 00:00:00 | 2020-12-31 23:57:00 |
| `SM Cleaned Data MH2021.csv` | Mathura | 2021 | 550,560 | 35 | 2021-01-01 00:00:00 | 2021-02-20 23:57:00 |
| **Total Raw Telemetry** | — | — | **21,394,429** | **84** | **2019-05-01** | **2021-10-31** |

---

## 3. Physical Value Distributions & Electrical Sanity

Across all 21,394,429 observations:

### Active Energy (`t_kWh`)
* **Range:** $[0.000, 0.300]\text{ kWh}$ per 3-minute interval.
* **Mean:** $0.0171\text{ kWh}$ (equivalent to an average continuous active load of $\approx 342\text{ W}$).
* **Negative Values:** $0$ (No sensor calibration offsets or polarity inversions found).
* **Exact Zeros:** $3,505,437$ ($16.38\%$).

### Voltage ($V_{\text{RMS}}$)
* **Range:** $[0.0\text{ V}, 654.73\text{ V}]$.
* **Mean:** $223.08\text{ V}$ (nominal standard is 230V).
* **Extreme Spikes ($>300\text{ V}$):** $15,653$ records ($0.073\%$).
* **Outages ($V = 0\text{ V}$):** $1,751,938$ records ($8.19\%$).

### Current ($I_{\text{RMS}}$)
* **Range:** $[0.0\text{ A}, 126.05\text{ A}]$.
* **Mean:** $1.62\text{ A}$.
* **Exact Zeros:** $2,892,849$ records ($13.52\%$).

### Frequency ($f$)
* **Range:** $[0.0\text{ Hz}, 570.33\text{ Hz}]$.
* **Mean:** $45.91\text{ Hz}$ (nominal is 50.0 Hz; pull-down caused by blackout zeros).
* **Zero Frequency ($f = 0\text{ Hz}$):** $1,751,945$ records ($8.19\%$).

---

## 4. Physics of Zeros Validation

The audit directly validated the physical hypothesis regarding zero consumption:

1. **True Voluntary Standby ($V > 180\text{V}, I = 0\text{A}, t\_kWh = 0$):**
   * **Count:** $1,094,803$ rows ($5.12\%$).
   * **Interpretation:** The electrical grid was fully energized, but the consumer was asleep, appliances were idle, or the residence was vacant.
2. **Grid Outages / Blackouts ($V = 0\text{V}, f = 0\text{Hz}, t\_kWh = 0$):**
   * **Count:** $1,745,585$ rows ($8.16\%$).
   * **Interpretation:** Feeder trip or DISCOM load-shedding. In the downstream feature pipeline, these will be assigned an explicit indicator `is_outage = 1`.

---

## 5. Phase 1 Verification Checklist Status

- [x] **Gate 1 (Provenance Verification):** Verified Harvard Dataverse source under CC0 Public Domain Dedication.
- [x] **Gate 2 (File Inventory & Checksum):** Validated 8 primary district CSVs with SHA-256 integrity logs.
- [x] **Gate 3 (Schema Contract):** Confirmed uniform 6-column physical schema across all raw files.
- [x] **Gate 4 (Data Quality Audit):** Verified 0 nulls, 0 duplicate keys, 0 negative values across 21,394,429 rows.
- [x] **Gate 5 (Temporal Regularization Setup):** Quantified blackout outage rates (8.16%) vs standby (5.12%).
- [x] **Gate 6 (Modeling Partition Boundaries):** Validated date bounds across 2019, 2020, and 2021.
- [x] **Gate 7 (Dataset Ready for Stage 2 EDA):** Raw baseline frozen.
