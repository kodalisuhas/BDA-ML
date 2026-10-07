# Phase 3A: Data Quality & Cleaning Policy Specification (Reconciled v1.1)

**Project:** Energy Demand Forecasting & Customer Clustering with Apache Spark + MLlib  
**Domain Asset:** CEEW Mathura & Bareilly High-Frequency Smart Meter Dataset  
**Status:** Frozen Baseline  

---

## 1. Core Engineering Principles

1. **Physical Immutability:** Raw CSV files in `data/raw/` are strictly read-only reference assets.
2. **Physical Event Preservation:** Real-world smart meter data contains legitimate physical zeros ($1.75\text{M}$ blackout records and $1.52\text{M}$ voluntary idle standby records). Deleting zeros or replacing them with means distorts physical load profiles.
3. **Three-Tier Boundary Taxonomy:** Every electrical parameter is bounded by three explicit, non-overlapping thresholds:
   * **Tier 1: Nominal Range** (Standard residential operating window; retained with 0 flags).
   * **Tier 2: Extreme Physical Event Range** (Valid transient/power-quality stress; retained with explicit quality flags).
   * **Tier 3: Rejection / Quarantine Range** (Corrupted telemetry, polarity reversal, or unparseable formats; quarantined).

---

## 2. Unambiguous Validation & Quality Action Matrix

| Parameter | Tier 1: Nominal (Retain, No Flag) | Tier 2: Extreme Event (Retain + Flag) | Tier 3: Rejection (Quarantine) | Associated Quality Flag |
|---|---|---|---|---|
| **Composite Key** | Unique `(district, meter_id, ts)` | — | Duplicate key occurrence | Quarantined / Deduplicated |
| **Timestamp (`ts`)** | `2019-05-01` to `2021-10-31` | — | Unparseable or out-of-bounds | `is_valid_timestamp = 0` (Reject) |
| **Active Energy (`t_kWh`)** | $0.000 \le E \le 0.300\text{ kWh}$ ($0 - 6\text{ kW}$) | $0.300 < E \le 0.500\text{ kWh}$ ($6 - 10\text{ kW}$) | $E < 0.0$ or $E > 0.500\text{ kWh}$ | `is_extreme_load = 1` |
| **Voltage ($V_{\text{RMS}}$)** | $180.0\text{ V} \le V \le 260.0\text{ V}$ | $0 < V < 180\text{V}$ (Brownout)<br>$260 < V \le 300\text{V}$ (High)<br>$300 < V \le 700\text{V}$ (Transient Spike) | $V < 0.0\text{ V}$ or $V > 700.0\text{ V}$ | `is_brownout = 1`<br>`is_voltage_high = 1`<br>`is_voltage_spike = 1` |
| **Current ($I_{\text{RMS}}$)** | $0.0\text{ A} \le I \le 50.0\text{ A}$ | $50.0\text{ A} < I \le 150.0\text{ A}$ | $I < 0.0\text{ A}$ or $I > 150.0\text{ A}$ | `is_overcurrent = 1` |
| **Frequency ($f$)** | $49.5\text{ Hz} \le f \le 50.5\text{ Hz}$ | $0.0\text{ Hz}$ (Outage)<br>$40.0 \le f < 49.5\text{ Hz}$ or $50.5 < f \le 60.0\text{ Hz}$ | $f < 0.0$ or ($f > 0$ and $f < 40\text{ Hz}$) or $f > 60\text{ Hz}$ | `is_frequency_abnormal = 1` |

---

## 3. Mathematically Closed 5-State Electrical State Machine

Every validated 3-minute record is assigned an explicit, mutually exclusive operational state with **zero fallback/orphan records**:

```
                                  ATOMIC RECORD (t_kWh, V, I, f)
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
         Is Voltage V == 0.0V?                                       Is Voltage V > 0.0V?
                 │                                                             │
                 ▼                                             ┌───────────────┼───────────────┐
              State 1                                          ▼               ▼               ▼
          (Grid Outage)                                    V < 180V        180V ≤ V ≤ 260V     V > 260V
        (Power Cut/No V)                                       │               │               │
                                                               ▼               ▼               ▼
                                                            State 3       Is kWh == 0       State 5
                                                           (Brownout)     & I ≤ 0.05A?    (Overvoltage)
                                                                               │
                                                                       ┌───────┴───────┐
                                                                       ▼               ▼
                                                                    State 2         State 4
                                                                   (Standby)    (Normal Active)
```

1. **State 1 (Grid Outage / Blackout):** $V = 0.0\text{ V}$. (Accounts for $1,751,938$ rows, $8.19\%$).
2. **State 2 (Voluntary Standby / Idle):** $V \ge 180.0\text{ V}$, $I \le 0.05\text{ A}$, and $t\_\text{kWh} = 0.0$. (Accounts for $1,523,831$ rows, $7.12\%$).
3. **State 3 (Brownout Supply):** $0.0\text{ V} < V < 180.0\text{ V}$. (Accounts for $175,768$ rows, $0.82\%$).
4. **State 4 (Normal Active Operation):** $180.0\text{ V} \le V \le 260.0\text{ V}$ with active demand ($t\_\text{kWh} > 0.0$ or $I > 0.05\text{ A}$). (Accounts for $15,690,354$ rows, $73.34\%$).
5. **State 5 (Overvoltage Transient / Spike):** $V > 260.0\text{ V}$. (Accounts for $2,252,538$ rows, $10.53\%$).
6. **State 6 (Fallback / Orphan):** **$0$ rows ($0.000\%$)** (Proven exhaustive across all 21.4M rows).

---

## 4. Hourly Aggregation, Scaling & Outage Policies

### Hourly Scaling Policy & Invariant Preservation
* **Storage Invariant:** Both the unadjusted observation sum (`raw_kwh_sum`) and the estimated clock-hour demand (`hourly_kwh`) are preserved in the Gold layer.
* **Missing-at-Random Linear Expectation Assumption:**  
  For windows containing $18 \le \text{sample\_count} < 20$ readings ($90\% - 95\%$ temporal completeness), hourly demand is estimated as:
  $$E_{\text{hourly}} = \left(\sum_{i=1}^N t\_\text{kWh}_i\right) \times \left(\frac{20}{N}\right)$$
  *Note:* This scaling is explicitly documented as a missing-at-random linear expectation, enabling subsequent sensitivity tests against strict $20/20$ complete subsets.
* **Completeness Flag:** `is_complete_hour = 1` if $\text{sample\_count} \ge 18$; else `0`.

### Outage-Heavy Hours Policy
* Compute clock-hour outage fraction:
  $$\text{outage\_ratio} = \frac{\sum \mathbb{I}(\text{State} = 1)}{\text{sample\_count}}$$
* **Data Retention:** Outage-heavy hours ($\text{outage\_ratio} \ge 0.50$) are **never deleted from the Gold dataset**. They are tagged with `is_outage_hour = 1`.
* **Sub-System Modeling Flags:**
  * `is_valid_forecast_target = 1` iff `is_complete_hour == 1` and `is_outage_hour == 0` (enables seamless filtering for "normal-supply demand forecasting" while preserving raw telemetry for full grid load forecasting).

---

## 5. Temporal Boundaries & Leakage Prevention Policy

1. **Chronological Splitting:** Random train/test shuffling across time is strictly prohibited.
2. **Training Partition:** `2019-05-01` to `2020-08-31` (16 months, multi-season coverage).
3. **Validation Partition:** `2020-09-01` to `2020-12-31` (4 months, hyperparameter tuning).
4. **Out-of-Time Test Partition:** `2021-01-01` to `2021-10-31` (Holdout evaluation).
5. **Feature Lag Invariant:** Any lag feature $Y_{T-k}$ at window $T$ must strictly reference windows $t \le T-k$.
