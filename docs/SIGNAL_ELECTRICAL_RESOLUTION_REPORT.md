# Consolidated Forensic Report: Signal Distribution, Electrical Forensics & Empirical Resolution Selection

**Execution Timestamp:** 2026-10-06  
**Analyzed Corpus:** 21,394,429 raw 3-minute records across 84 smart meters in Uttar Pradesh (CEEW)  
**Execution Engine:** Apache Spark 4.2.0 on OpenJDK 17 LTS (`local[*]`, 16 CPU cores)  
**Runtime:** 179.72 seconds  

---

## 1. Explicit Timestamp Parsing & Key Integrity Audit

* **Ingested Rows Across 6 Raw Files:** $21,394,429$
* **Timestamp Parsing Audit:** Parsed via strict multi-pattern parsing (`yyyy-MM-dd HH:mm:ss`, `dd-MM-yyyy HH:mm:ss`).
* **Timestamp Parse Failures (`ts IS NULL`):** **$0$ records ($0.000\%$)**
* **Primary Key Definition:** All records are uniquely and unambiguously indexed by the composite key `(district, meter_id, ts)`.

---

## 2. Phase 2.3: Active Energy Consumption ($t\_\text{kWh}$) Signal Analysis

### Summary Statistics
* **Total Observations:** $21,394,429$
* **Zero Consumption Observations:** $3,505,437$ ($16.38\%$)
* **Active (Non-Zero) Observations:** $17,888,992$ ($83.62\%$)
* **Dynamic Range:** $[0.0000\text{ kWh}, 0.3000\text{ kWh}]$ per 3-minute window (equivalent to $0\text{ W} - 6.0\text{ kW}$ instantaneous active load).
* **Mean Active Energy:** $0.0171\text{ kWh}$ ($\approx 341.5\text{ W}$ continuous active power).
* **Standard Deviation ($\sigma$):** $0.0248\text{ kWh}$
* **Distribution Skewness:** $+3.01$ (pronounced positive right-tail skewness representing intermittent high-power appliance activations such as air conditioners, water heaters, and induction cooktops).

### Exact Percentile Distributions
| Percentile Cut | All Data ($t\_\text{kWh}$) | Equivalent Continuous Power (W) | Non-Zero Data ($t\_\text{kWh}$) | Equivalent Continuous Power (W) |
|---|---|---|---|---|
| **$p_{10}$** | $0.0000\text{ kWh}$ | $0\text{ W}$ | $0.0020\text{ kWh}$ | $40\text{ W}$ |
| **$p_{25}$** | $0.0020\text{ kWh}$ | $40\text{ W}$ | $0.0060\text{ kWh}$ | $120\text{ W}$ |
| **$p_{50}$ (Median)** | **$0.0090\text{ kWh}$** | **$180\text{ W}$** | **$0.0120\text{ kWh}$** | **$240\text{ W}$** |
| **$p_{75}$** | $0.0200\text{ kWh}$ | $400\text{ W}$ | $0.0230\text{ kWh}$ | $460\text{ W}$ |
| **$p_{90}$** | $0.0390\text{ kWh}$ | $780\text{ W}$ | $0.0460\text{ kWh}$ | $920\text{ W}$ |
| **$p_{95}$** | $0.0740\text{ kWh}$ | $1,480\text{ W}$ | $0.0820\text{ kWh}$ | $1,640\text{ W}$ |
| **$p_{99}$** | $0.1200\text{ kWh}$ | $2,400\text{ W}$ | $0.1240\text{ kWh}$ | $2,480\text{ W}$ |
| **Max** | $0.3000\text{ kWh}$ | $6,000\text{ W}$ | $0.3000\text{ kWh}$ | $6,000\text{ W}$ |

---

## 3. Phase 2.4: Electrical Quality & Grid Stability Forensics

### Supply Voltage ($V_{\text{RMS}}$)
* **Global Mean (All rows):** $223.08\text{ V}$ (pulled down by blackout zeros).
* **Energized Grid Mean ($V > 0\text{V}$):** **$242.98\text{ V}$** (nominal standard is $230\text{ V}$).
* **Energized Grid Voltage Quantiles:**
  * $p_5 = 212.53\text{ V}$ | $p_{25} = 234.53\text{ V}$ | $p_{50} = 245.37\text{ V}$ | $p_{75} = 254.34\text{ V}$ | $p_{95} = 265.88\text{ V}$

### Grid Frequency & Current
* **Energized Grid Frequency Mean:** **$50.00\text{ Hz}$**
* **CERC Grid Code Compliant ($49.90\text{ Hz} - 50.05\text{ Hz}$):** $15,308,249$ rows ($71.55\%$).
* **Current Mean ($I_{\text{RMS}}$):** $1.62\text{ A}$ (Max: $126.05\text{ A}$).

---

## 4. Exhaustive 5-State Electrical Operating Taxonomy

Every 3-minute atomic observation was classified into one of 5 mutually exclusive operational states:

| State | Physical Definition | Observation Count | % Share | Mean Energy ($t\_\text{kWh}$) | Mean Voltage ($V$) | Operational Interpretation |
|---|---|---|---|---|---|---|
| **State 1: Grid Blackout** | $V=0\text{V}, f=0\text{Hz}, I=0\text{A}, \text{kWh}=0$ | **$1,751,932$** | **$8.19\%$** | $0.0000$ | $0.0\text{ V}$ | Complete feeder power cut / load-shedding |
| **State 2: Voluntary Standby** | $V \ge 180\text{V}, I \le 0.05\text{A}, \text{kWh}=0$ | **$1,523,831$** | **$7.12\%$** | $0.0000$ | $244.0\text{ V}$ | Grid energized; house vacant or idling |
| **State 3: Brownout Supply** | $0\text{V} < V < 180\text{V}$ | **$175,768$** | **$0.82\%$** | $0.0267$ | $163.9\text{ V}$ | Severe under-voltage; elevated inductive current |
| **State 4: Normal Active Operation** | $180\text{V} \le V \le 260\text{V}, \text{active}$ | **$15,690,354$** | **$73.34\%$** | $0.0212$ | $240.4\text{ V}$ | Standard domestic residential load |
| **State 5: Overvoltage Spikes** | $V > 260\text{V}$ | **$2,252,538$** | **$10.53\%$** | $0.0124$ | $266.4\text{ V}$ | Sustained high voltage; reduced consumer load |

---

## 5. Phase 2.6: Empirical Temporal Resolution Benchmark

Four candidate temporal aggregation windows were evaluated across the entire 21.4M corpus:

| Metric | 3-Minute (Raw) | 15-Minute | 30-Minute | 1-Hour (Selected) |
|---|---|---|---|---|
| **Total Observation Count ($N$)** | $21,394,429$ | $4,279,109$ | $2,139,632$ | **$1,074,001$** |
| **Mean Active Energy** | $0.0171\text{ kWh}$ | $0.0854\text{ kWh}$ | $0.1707\text{ kWh}$ | **$0.3401\text{ kWh}$** |
| **Standard Deviation ($\sigma$)** | $0.0248\text{ kWh}$ | $0.1184\text{ kWh}$ | $0.2292\text{ kWh}$ | **$0.4385\text{ kWh}$** |
| **Variance ($\sigma^2$)** | $0.000614$ | $0.014024$ | $0.052539$ | **$0.192244$** |
| **Coefficient of Variation ($\text{CV} = \sigma / \mu$)** | $1.451$ | $1.387$ | $1.342$ | **$1.289$ (Lowest Relative Dispersion)** |
| **Signal-to-Noise Ratio ($\text{SNR}_{\text{dB}}$)** | $-3.23\text{ dB}$ | $-2.84\text{ dB}$ | $-2.56\text{ dB}$ | **$-2.21\text{ dB}$ (Best Signal Quality)** |
| **Zero-Consumption Prevalence** | $16.38\%$ | $11.56\%$ | $9.86\%$ | **$7.96\%$ ($-51.4\%$ Zero Inflation)** |
| **Lag-1 Autocorrelation ($r_{t, t-1}$)** | **$0.9387$** | **$0.8728$** | **$0.8369$** | **$0.8057$ (Strong Model Predictability)** |

### Quantitative Justification for 1-Hour Resolution
1. **Noise Suppression & Signal Quality:** The 1-hour resolution yields the highest Signal-to-Noise Ratio ($\text{SNR} = -2.21\text{ dB}$) and lowest Coefficient of Variation ($\text{CV} = 1.289$), effectively filtering short-lived compressor and motor start pulses.
2. **Mitigation of Zero-Inflation:** Zero-valued records drop from $16.38\%$ to $7.96\%$, drastically reducing zero-inflation distortion in linear and tree-based regression estimators.
3. **Temporal Autoregressive Persistence:** The lag-1 autocorrelation remains exceptionally strong at $r = 0.8057$, guaranteeing that autoregressive features ($Y_{t-1}, Y_{t-2}, Y_{t-24}, Y_{t-168}$) provide robust predictive power.
4. **Computational Scale:** Exactly **$1,074,001$ regularized hourly records** provides sufficient scale to demonstrate Spark distributed execution, caching, and MLlib regression training without driver memory overflow.

---

## 6. District Comparison: Bareilly vs. Mathura

| Characteristic | Bareilly (`BR`) | Mathura (`MH`) |
|---|---|---|
| **Raw Telemetry Observations** | $13,495,635$ ($63.1\%$) | $7,898,794$ ($36.9\%$) |
| **Mean Energized Voltage ($V > 0$)** | $243.25\text{ V}$ | $242.51\text{ V}$ |
| **Brownouts & Outages ($V < 180\text{V}$)** | $1,046,792$ rows ($7.76\%$) | **$880,914$ rows ($11.15\%$)** |
| **Active Energy Mean (3-min)** | $0.0171\text{ kWh}$ ($\sigma = 0.0246$) | $0.0170\text{ kWh}$ ($\sigma = 0.0251$) |
| **Zero Consumption Prevalence** | $16.57\%$ | $16.07\%$ |

*Conclusion:* While mean consumption is nearly identical ($\approx 341\text{ W}$ continuous), Mathura suffered from **$+43.7\%$ higher relative power quality stress** ($11.15\%$ vs $7.76\%$ under-voltage/blackouts).
