# Phase 2: Temporal Forensics & Meter Coverage Report

**Execution Timestamp:** 2026-10-06  
**Analysis Target:** 21,394,429 raw observations across 84 smart meters  
**Compute Engine:** Apache Spark 4.2.0 (`local[*]`, 16 CPU cores)  
**Execution Runtime:** 84.99 seconds  

---

## 1. Executive Findings: Temporal Continuity & Gaps

A windowed lag analysis ($\Delta t_i = t_i - t_{i-1}$) was evaluated over all **21,394,345 consecutive measurement transitions**:

| Interval Classification | Time Delta ($\Delta t$) | Transition Count | Percentage | Minimum $\Delta t$ | Maximum $\Delta t$ |
|---|---|---|---|---|---|
| **1. Nominal Cadence** | $\Delta t = 3\text{ minutes}$ | **21,390,002** | **99.98%** | 3.0 min | 3.0 min |
| **2. Short Telemetry Jitter** | $3\text{ min} < \Delta t \le 15\text{ min}$ | **70** | $0.0003\%$ | 6.0 min | 15.0 min |
| **3. Medium Communications Gap** | $15\text{ min} < \Delta t \le 1\text{ hour}$ | **120** | $0.0006\%$ | 18.0 min | 48.0 min |
| **4. Long Inactive Gap** | $1\text{ hour} < \Delta t \le 24\text{ hours}$ | **83** | $0.0004\%$ | 63.0 min | 1,440.0 min |
| **5. Extended Meter Dropout** | $\Delta t > 24\text{ hours}$ | **4,070** | **0.02%** | 1,443.0 min (1.0 d) | 205,923.0 min (143 d) |
| **Total Analyzed Transitions** | — | **21,394,345** | **100.00%** | — | — |

### Key Engineering Takeaway
> **99.98% of all consecutive records strictly follow the exact 3-minute sampling frequency.**  
> Grid blackouts ($V=0, f=0$) did not disconnect meter logging; the AMI infrastructure explicitly captured outage rows at regular 3-minute ticks ($1.75\text{M}$ blackout rows logged). Missingness exists almost entirely as multi-day deployment dropouts rather than chaotic intra-hour packet jitter.

---

## 2. Per-Meter Coverage & Lifespan Summary

### Lifespan Distributions
* **Mathura ($N=38$ meters):**
  * Rollout commenced: May 1 – May 31, 2019.
  * Active monitoring endpoint: **February 18–20, 2021** (~630–660 active days).
  * Mean observations per meter: ~207,863 rows.
  * Mean local lifespan coverage: **68.2%** (due to selective feeder deactivations).
* **Bareilly ($N=46$ meters):**
  * Rollout commenced: May 9 – September 24, 2019 (staggered in 3 waves).
  * Active monitoring endpoint: **October 31, 2021** (~780–886 active days).
  * Mean observations per meter: ~293,383 rows.
  * Mean local lifespan coverage: **89.5%** (high temporal density).

---

## 3. District Comparison: Bareilly vs. Mathura

| Metric | Bareilly (`BR`) | Mathura (`MH`) | Combined / Delta |
|---|---|---|---|
| **Monitored Meters** | 46 meters | 38 meters | 84 total |
| **Total Telemetry Rows** | 13,495,635 rows (63.1%) | 7,898,794 rows (36.9%) | 21,394,429 rows |
| **Active Energy Mean ($t\_\text{kWh}$)** | $0.0171\text{ kWh}$ (~$342.0\text{ W}$) | $0.0170\text{ kWh}$ (~$340.6\text{ W}$) | Parity ($\Delta < 0.5\%$) |
| **Mean Supply Voltage ($V_{\text{RMS}}$)** | **$225.22\text{ V}$** | **$219.44\text{ V}$** | Mathura suffers from lower average grid voltage |
| **Mean Supply Frequency ($f$)** | $46.29\text{ Hz}$ | $45.24\text{ Hz}$ | Driven by outage zero pull-down |
| **Grid Outages ($V=0, f=0, \text{kWh}=0$)** | **$1,000,280$ rows ($7.41\%$)** | **$751,652$ rows ($9.52\%$)** | Mathura experienced $+28.5\%$ higher blackout rate |
| **Voluntary Standby ($V>180V, I=0$)** | **$818,816$ rows ($6.07\%$)** | **$275,987$ rows ($3.49\%$)** | Higher idling baseline in Bareilly |

---

## 4. Architectural Implications for Downstream Pipelines

1. **Hourly Resampling Feasibility:**  
   Because intra-hour 3-minute readings are $99.98\%$ regular, downsampling via Spark SQL clock-hour windows (`groupBy(meter_id, window(ts, '1 hour'))`) will aggregate cleanly without requiring complex interpolation of fractional minute gaps.
2. **Outage Flag Preservation:**  
   When calculating hourly aggregated energy $Y_t = \sum t\_\text{kWh}$, we compute the hourly outage fraction $\text{outage\_ratio} = \frac{N(V=0)}{N_\text{total}}$. Hours with $\text{outage\_ratio} > 0.5$ can be explicitly flagged.
3. **Partitioning & Modeling Split:**  
   * **Train / Validation Period:** May 2019 – December 2020 (covers both districts across all seasonal cycles).
   * **Out-of-Time Test Period:** Jan 2021 – Feb 2021 (Dual-district evaluation) and March 2021 – Oct 2021 (Bareilly extended holdout).
