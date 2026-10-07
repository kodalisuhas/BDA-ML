# CEEW Smart Meter Dataset Manifest & File Integrity Log

**Project:** Energy Consumption / Demand Forecasting & Customer Clustering with Apache Spark + MLlib  
**Dataset Provenance:** Council on Energy, Environment and Water (CEEW)  
**Archive Reference:** Harvard Dataverse (`doi:10.7910/DVN/GOCHJH`)  
**License:** Creative Commons CC0 1.0 Universal (Public Domain Dedication)  
**Location:** `data/raw/`  
**Timestamp Recorded:** 2026-10-06  

---

## 1. Raw & Reference File Inventory

The intake directory contains **8 CSV files** representing high-frequency (3-minute interval) smart meter readings from Mathura and Bareilly districts in Uttar Pradesh, India, along with district-level aggregated benchmark reference files.

| # | File Name | District | Temporal Scope | Role | Size (Bytes) | Size (MB) |
|---|---|---|---|---|---|---|
| 1 | `CEEW - Smart meter data Bareilly 2020.csv` | Bareilly | 2020 (Full Year) | Raw Telemetry (3-min) | 323,383,946 | 308.40 MB |
| 2 | `CEEW - Smart meter data Bareilly 2021.csv` | Bareilly | 2021 (Jan – Oct) | Raw Telemetry (3-min) | 192,682,564 | 183.76 MB |
| 3 | `CEEW - Smart meter data Mathura 2019.csv` | Mathura | 2019 (May – Dec) | Raw Telemetry (3-min) | 175,494,264 | 167.36 MB |
| 4 | `CEEW - Smart meter data Mathura 2020.csv` | Mathura | 2020 (Full Year) | Raw Telemetry (3-min) | 182,625,294 | 174.17 MB |
| 5 | `SM Cleaned Data BR2019.csv` | Bareilly | 2019 (May – Dec) | Raw Telemetry (3-min) | 142,557,800 | 135.95 MB |
| 6 | `SM Cleaned Data MH2021.csv` | Mathura | 2021 (Jan – Oct) | Raw Telemetry (3-min) | 26,711,040 | 25.47 MB |
| 7 | `SM Cleaned Data BR Aggregated.csv` | Bareilly | 2019 – 2021 | Reference Aggregation | 978,437 | 0.93 MB |
| 8 | `SM Cleaned Data MH Aggregated.csv` | Mathura | 2019 – 2021 | Reference Aggregation | 578,103 | 0.55 MB |

**Total Raw Data Volume:** ~1,045,011,448 Bytes (~1.045 GB uncompressed).

---

## 2. Cryptographic Integrity (SHA-256 Checksums)

Every file in the raw storage landing zone is cryptographically fingerprinted to enforce immutability:

```text
3ED74D940BC394875C7533D4F14668ADE9780CB061928EA137CF3A2778396384  CEEW - Smart meter data Bareilly 2020.csv
0648BA7AADC5ABE69D3B4427F6E9E9D45C2721D8566A7A78003916DCE2EB4DF5  CEEW - Smart meter data Bareilly 2021.csv
C09380165E114E7028BBD1A937E08BA7938A2F99E9E41C61E1B606B669AFCBE4  CEEW - Smart meter data Mathura 2019.csv
E77DDF845B7D00FF1E206556E5657480507CC25A6D0600D6F129DC0C01470C53  CEEW - Smart meter data Mathura 2020.csv
01746BE1E45B61E175A7097941CAF932188D3E71FC3E8EE0A46B027D2C5AC830  SM Cleaned Data BR2019.csv
1A0CDF746FE85B55A135B2A13A48F1CC0D7D5046DB534F76BA0C6A80846249AB  SM Cleaned Data MH2021.csv
504495933CADD1987C235D3CF7E66E3A56830F4D7679E62DD6AD9EC32063FE40  SM Cleaned Data BR Aggregated.csv
AE79E4775B477D781D317864B2FF60D113F509A034774784F383D82A7EE034F1  SM Cleaned Data MH Aggregated.csv
```

---

## 3. Structural Roles

* **Primary Modeling Files (6 Files):**
  * Bareilly: `SM Cleaned Data BR2019.csv`, `CEEW - Smart meter data Bareilly 2020.csv`, `CEEW - Smart meter data Bareilly 2021.csv`
  * Mathura: `CEEW - Smart meter data Mathura 2019.csv`, `CEEW - Smart meter data Mathura 2020.csv`, `SM Cleaned Data MH2021.csv`
  * These files constitute the raw, atomic 3-minute interval time series feeding Spark data engineering pipelines.

* **Reference Validation Files (2 Files):**
  * `SM Cleaned Data BR Aggregated.csv`, `SM Cleaned Data MH Aggregated.csv`
  * Used exclusively as ground-truth reference benchmarks to audit Spark SQL hourly/daily downsampling logic against original CEEW aggregates. Not used for model training.

---

## 4. Source Immutability Contract
1. The `data/raw/` directory is **strictly read-only**.
2. No script, pipeline, or agent is permitted to write, overwrite, or mutate raw files in place.
3. All data parsing, typing, cleaning, and downstream transformations will output strictly to `data/processed/` in structured Parquet format.
