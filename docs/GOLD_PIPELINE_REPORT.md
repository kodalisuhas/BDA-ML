# Phase 4.2 & 4.3: Gold Hourly Pipeline & Verification Report (Hardened v1.1)

**Execution Timestamp:** 2026-10-06  
**Authoritative Contracts:** `DATA_QUALITY_POLICY.md` v1.1 & `CANONICAL_DATA_CONTRACT.md` v1.1  
**Execution Engine:** Apache Spark 4.2.0 on OpenJDK 17 LTS (`local[*]`, 16 CPU cores)  
**Input Source:** `data/interim/validated_3min.parquet` ($21,392,743$ rows)  
**Gold Target:** `data/processed/canonical_hourly.parquet`  
**Pipeline Runtime:** 31.73 seconds  
**Test Suite Runtime:** 48.55 seconds (20/20 Passed, 100% across Silver & Gold)  

---

## 1. Executive Summary & Record Accounting

The Phase 4.2 Gold data engineering pipeline aggregated all 21.39M Silver 3-minute observations into the canonical 1-hour modeling layer with exact energy scaling, power quality ratios, and modeling eligibility flags.

| Metric / Stage | Measured Ground Truth | Physical & Engineering Meaning |
|---|---|---|
| **Silver Ingestion Count** | **$21,392,743$ rows** | Validated 3-minute atomic observations |
| **Gold Canonical Hourly Rows** | **$1,074,001$ rows** | Exactly matches the deterministic forensic benchmark |
| **Complete Hours ($\ge 18/20$ samples)** | **$1,059,573$ rows ($98.66\%$)** | High temporal completeness across active meter spans |
| **Valid Normal-Supply Forecast Targets** | **$974,272$ rows ($90.71\%$)** | Clean supervised training targets (`is_valid_forecast_target == 1`) |
| **Outage-Heavy Hours ($\ge 50\%$ outage)** | **$85,301$ rows ($7.94\%$)** | Preserved with flag `is_outage_hour = 1` for outage modeling & clustering |
| **Primary Key Uniqueness** | **$0$ duplicate keys** | Composite `(district, meter_id, window_start)` is 100% unique |
| **Energy Conservation Invariant** | **$\Delta < 0.001\text{ kWh}$** | $\sum \text{raw\_kwh\_sum}_{\text{Gold}} \equiv \sum t\_\text{kWh}_{\text{Silver}}$ |

---

## 2. Power-Quality Multi-Level Measurement Specification

The architecture deliberately employs a two-level overvoltage measurement model:
* **Silver Event Flags:**
  * `is_voltage_high = 1`: Mild to moderate overvoltage ($260\text{V} < V \le 300\text{V}$).
  * `is_voltage_spike = 1`: Severe transient spike ($V > 300\text{V}$).
* **Gold Operational Metric:**
  * `spike_ratio`: Measures broader clock-hour overvoltage stress ($\text{fraction with } V > 270.0\text{ V} = \frac{\sum \mathbb{I}(V > 270)}{N}$).

---

## 3. End-to-End Automated Verification Test Suite (`pytest tests/ -v`)

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\kodal\BDA-ML

tests/test_gold_layer.py::test_gold_row_count PASSED                     [  5%]
tests/test_gold_layer.py::test_gold_schema_matches_contract PASSED       [ 10%]
tests/test_gold_layer.py::test_gold_zero_nulls PASSED                    [ 15%]
tests/test_gold_layer.py::test_gold_primary_key_uniqueness PASSED        [ 20%]
tests/test_gold_layer.py::test_gold_window_duration PASSED               [ 25%]
tests/test_gold_layer.py::test_gold_sample_count_bounds PASSED           [ 30%]
tests/test_gold_layer.py::test_gold_energy_non_negative PASSED           [ 35%]
tests/test_gold_layer.py::test_gold_complete_20_sample_exact_equality PASSED [ 40%]
tests/test_gold_layer.py::test_gold_scaling_formula_adherence PASSED     [ 45%]
tests/test_gold_layer.py::test_gold_ratios_within_unit_interval PASSED   [ 50%]
tests/test_gold_layer.py::test_gold_flag_logical_consistency PASSED      [ 55%]
tests/test_gold_layer.py::test_gold_energy_conservation_against_silver PASSED [ 60%]
tests/test_gold_layer.py::test_gold_spike_ratio_definition PASSED        [ 65%]
tests/test_silver_layer.py::test_silver_row_count PASSED                 [ 70%]
tests/test_silver_layer.py::test_silver_schema_matches_contract PASSED   [ 75%]
tests/test_silver_layer.py::test_silver_zero_nulls PASSED                [ 80%]
tests/test_silver_layer.py::test_silver_primary_key_uniqueness PASSED    [ 85%]
tests/test_silver_layer.py::test_silver_physical_bounds PASSED           [ 90%]
tests/test_silver_layer.py::test_silver_state_machine_closed PASSED      [ 95%]
tests/test_silver_layer.py::test_silver_flag_logical_consistency PASSED  [100%]

============================= 20 passed in 48.55s =============================
```

---

## 4. Hardened Verification Invariants Confirmed

1. **Exact Deterministic Count:** Exactly $1,074,001$ rows asserted.
2. **Energy Conservation Precision:** Global delta $\Delta < 1\text{e-3}\text{ kWh}$ verified against Silver.
3. **Exact 20/N Scaling Adherence:** For all $1,074,001$ rows, $\text{hourly\_kwh} = \text{raw\_kwh\_sum} \times (20 / \text{sample\_count})$ holds.
4. **20-Sample Identity:** For all complete 20-sample hours ($N=20$), $\text{hourly\_kwh} \equiv \text{raw\_kwh\_sum}$.
5. **Spike Ratio Definition:** Per-window mathematical verification confirming `spike_ratio` strictly reflects $V > 270.0\text{ V}$.
6. **Zero Nulls:** All 18 columns contain 0 nulls across the entire Gold layer.
