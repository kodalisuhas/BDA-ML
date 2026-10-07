# Phase 4.1.1: Hardened Silver Pipeline & Quarantine Audit Report

**Execution Timestamp:** 2026-10-06  
**Authoritative Specifications:** `DATA_QUALITY_POLICY.md` v1.1 & `CANONICAL_DATA_CONTRACT.md` v1.1  
**Execution Engine:** Apache Spark 4.2.0 on OpenJDK 17 LTS (`local[*]`, 16 CPU cores)  
**Silver Location:** `data/interim/validated_3min.parquet`  
**Quarantine Location:** `data/interim/quarantine_audit.parquet`  
**Pipeline Runtime:** 122.37 seconds  
**Test Suite Runtime:** 37.43 seconds (7/7 Passed, 100%)  

---

## 1. Hardening Enhancements Implemented

1. **Genuine `StructType` CSV Ingestion:**  
   Replaced loose string casting with an authoritative `RAW_TELEMETRY_SCHEMA` `StructType` passed directly into `spark.read.schema(RAW_TELEMETRY_SCHEMA).csv(...)`.
2. **Explicit Multi-Reason Quarantine Auditing:**  
   All non-conforming rows are evaluated against the three-tier rule matrix and tagged with an explicit, auditable `rejection_reason` string (e.g., `FREQUENCY_OUT_OF_BOUNDS`).
3. **Partitioned Snappy Parquet:**  
   Silver data is stored as columnar Snappy Parquet partitioned by `district` and `file_year`.

---

## 2. Ingestion & Quarantine Accounting

| Stage | Record Count | % Share | Audit Detail |
|---|---|---|---|
| **Raw Telemetry Ingested** | **$21,394,429$** | $100.00\%$ | Across 6 raw CSV files via explicit `StructType` |
| **Timestamp Parsing** | **$0$ parse failures** | $0.000\%$ | Strict multi-pattern parsing (`yyyy-MM-dd HH:mm:ss`, `dd-MM-yyyy HH:mm:ss`) |
| **Quarantined Rows (Tier 3)** | **$1,686$** | **$0.008\%$** | $100\%$ attributed to `FREQUENCY_OUT_OF_BOUNDS` (e.g. sensor artifacts $<40\text{Hz}$ or $>60\text{Hz}$) |
| **Validated Silver Rows** | **$21,392,743$** | **$99.992\%$** | Validated, classified, and deduplicated on `(district, meter_id, ts)` |

### Quarantine Audit Breakdown (`data/interim/quarantine_audit.parquet`)
```text
+-----------------------+-------+-------------------------------------------------------+
| rejection_reason      | Count | Physical Failure Mode                                 |
+-----------------------+-------+-------------------------------------------------------+
| FREQUENCY_OUT_OF_BOUNDS | 1,686 | Grid frequency sensor glitches (e.g. 19.4Hz, 79.1Hz)   |
+-----------------------+-------+-------------------------------------------------------+
```

---

## 3. Partition Inventory in Silver Layer

The dataset is partitioned physically into 6 district-year directory trees under `data/interim/validated_3min.parquet/`:

```
data/interim/validated_3min.parquet/
├── district=Bareilly/
│   ├── file_year=2019/ (2,919,315 validated rows)
│   ├── file_year=2020/ (6,626,448 validated rows)
│   └── file_year=2021/ (3,948,960 validated rows)
└── district=Mathura/
    ├── file_year=2019/ (3,588,100 validated rows)
    ├── file_year=2020/ (3,759,360 validated rows)
    └── file_year=2021/ (550,560 validated rows)
```

---

## 4. Automated Verification Test Suite Results (`tests/test_silver_layer.py`)

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\kodal\BDA-ML

tests/test_silver_layer.py::test_silver_row_count PASSED                 [ 14%]
tests/test_silver_layer.py::test_silver_schema_matches_contract PASSED   [ 28%]
tests/test_silver_layer.py::test_silver_zero_nulls PASSED                [ 42%]
tests/test_silver_layer.py::test_silver_primary_key_uniqueness PASSED    [ 57%]
tests/test_silver_layer.py::test_silver_physical_bounds PASSED           [ 71%]
tests/test_silver_layer.py::test_silver_state_machine_closed PASSED      [ 85%]
tests/test_silver_layer.py::test_silver_flag_logical_consistency PASSED  [100%]

============================= 7 passed in 37.43s ==============================
```

---

## 5. Verification Gate Status

- [x] **Genuine `StructType` Ingestion:** Explicit `RAW_TELEMETRY_SCHEMA` enforced at CSV read boundary.
- [x] **Zero Nulls Across Columns:** All 15 columns non-null and strongly typed.
- [x] **Primary Key Integrity:** Logical key `(district, meter_id, ts)` is 100% unique.
- [x] **Closed 5-State Taxonomy:** Exactly 0 State-6 records.
- [x] **Auditable Quarantine Layer:** All 1,686 invalid rows persisted with explicit `rejection_reason`.
