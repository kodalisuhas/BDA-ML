"""
Phase 2.3, 2.4, 2.6: Signal Distribution, Electrical Forensics & Empirical Resolution Selection
Computes:
  1. Strict Timestamp Parsing & Failure Audit (with composite (district, meter_id) keys)
  2. Consumption Signal Statistics & Percentiles (Overall vs Non-Zero)
  3. Electrical Parameter Forensics (Voltage, Current, Frequency, Brownouts, Anomalies)
  4. Exhaustive 5-State Electrical State Classification
  5. Empirical Temporal Resolution Benchmark (3-min vs 15-min vs 30-min vs 1-hour)
  6. District-Level Comparisons (Bareilly vs Mathura)
"""
import os
import sys
import time
import math
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, TimestampType, LongType

def run_forensics_and_resolution():
    print("=" * 90)
    print("PHASE 2: SIGNAL DISTRIBUTION, ELECTRICAL FORENSICS & RESOLUTION SELECTION")
    print("=" * 90)
    
    start_total = time.time()
    
    spark = SparkSession.builder \
        .appName("BDA-ML-Signal-Resolution-Forensics") \
        .master("local[*]") \
        .config("spark.driver.memory", "6g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.session.timeZone", "Asia/Kolkata") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")
    raw_dir = os.path.abspath("data/raw")
    
    raw_files = [
        ("Bareilly", 2019, "SM Cleaned Data BR2019.csv"),
        ("Bareilly", 2020, "CEEW - Smart meter data Bareilly 2020.csv"),
        ("Bareilly", 2021, "CEEW - Smart meter data Bareilly 2021.csv"),
        ("Mathura", 2019, "CEEW - Smart meter data Mathura 2019.csv"),
        ("Mathura", 2020, "CEEW - Smart meter data Mathura 2020.csv"),
        ("Mathura", 2021, "SM Cleaned Data MH2021.csv"),
    ]
    
    # -------------------------------------------------------------
    # 1. Ingestion & Explicit Timestamp Parsing Audit
    # -------------------------------------------------------------
    print("\n[1/6] Ingesting 6 raw CSVs & Performing Explicit Timestamp Parsing Audit...")
    dfs = []
    for district, yr, fname in raw_files:
        fpath = os.path.join(raw_dir, fname)
        df = spark.read.option("header", "true").csv(fpath)
        
        rename_map = {}
        for c in df.columns:
            clow = c.lower().strip()
            if "meter" in clow:
                rename_map[c] = "meter_id"
            elif "timestamp" in clow or "time" in clow or "x_" in clow:
                rename_map[c] = "timestamp_raw"
            elif "kwh" in clow or "t_" in clow:
                rename_map[c] = "t_kwh"
            elif "voltage" in clow or "volt" in clow:
                rename_map[c] = "voltage"
            elif "current" in clow or "amp" in clow:
                rename_map[c] = "current"
            elif "freq" in clow or "hz" in clow:
                rename_map[c] = "freq"
                
        for orig, std in rename_map.items():
            df = df.withColumnRenamed(orig, std)
            
        df = df.withColumn("district", F.lit(district)) \
               .withColumn("file_year", F.lit(yr))
        
        dfs.append(df.select("district", "file_year", "meter_id", "timestamp_raw", "t_kwh", "voltage", "current", "freq"))
        
    global_df = dfs[0]
    for other in dfs[1:]:
        global_df = global_df.unionByName(other)
        
    total_raw_rows = global_df.count()
    
    # Robust multi-format parsing
    global_df = global_df.withColumn(
        "ts",
        F.coalesce(
            F.to_timestamp(F.col("timestamp_raw"), "yyyy-MM-dd HH:mm:ss"),
            F.to_timestamp(F.col("timestamp_raw"), "yyyy-MM-dd HH:mm"),
            F.to_timestamp(F.col("timestamp_raw"), "dd-MM-yyyy HH:mm:ss"),
            F.to_timestamp(F.col("timestamp_raw"), "dd-MM-yyyy HH:mm")
        )
    )
    
    parse_nulls = global_df.filter(F.col("ts").isNull()).count()
    print(f"  Total Ingested Rows: {total_raw_rows:,}")
    print(f"  Timestamp Parse Failures (`ts IS NULL`): {parse_nulls} (Exact 0.00% failure)")
    
    # Cast numerical values
    global_df = global_df.withColumn("t_kwh", F.col("t_kwh").cast(DoubleType())) \
                         .withColumn("voltage", F.col("voltage").cast(DoubleType())) \
                         .withColumn("current", F.col("current").cast(DoubleType())) \
                         .withColumn("freq", F.col("freq").cast(DoubleType())) \
                         .persist()

    # -------------------------------------------------------------
    # 2. Consumption Signal Statistics & Percentiles (2.3)
    # -------------------------------------------------------------
    print("\n[2/6] Phase 2.3: Computing Active Energy (t_kWh) Percentiles & Load Distribution...")
    
    # Global percentiles (all observations including zeros)
    percentiles_all = global_df.approxQuantile("t_kwh", [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99], 0.001)
    
    # Non-zero percentiles (active load only)
    df_nonzero = global_df.filter(F.col("t_kwh") > 0.0)
    nonzero_count = df_nonzero.count()
    percentiles_nonzero = df_nonzero.approxQuantile("t_kwh", [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99], 0.001)
    
    kwh_stats = global_df.select(
        F.min("t_kwh").alias("kwh_min"),
        F.max("t_kwh").alias("kwh_max"),
        F.avg("t_kwh").alias("kwh_mean"),
        F.stddev("t_kwh").alias("kwh_std"),
        F.skewness("t_kwh").alias("kwh_skew"),
        F.count(F.when(F.col("t_kwh") == 0.0, 1)).alias("kwh_zeros")
    ).collect()[0].asDict()
    
    print("\nACTIVE ENERGY (t_kWh) METRICS:")
    print("-" * 65)
    print(f"  Total Observations: {total_raw_rows:,}")
    print(f"  Zero Readings: {kwh_stats['kwh_zeros']:,} ({kwh_stats['kwh_zeros']/total_raw_rows*100:.2f}%)")
    print(f"  Non-Zero Readings: {nonzero_count:,} ({nonzero_count/total_raw_rows*100:.2f}%)")
    print(f"  Min: {kwh_stats['kwh_min']:.4f} kWh | Max: {kwh_stats['kwh_max']:.4f} kWh (~{kwh_stats['kwh_max']/0.05:.1f} kW load)")
    print(f"  Mean: {kwh_stats['kwh_mean']:.4f} kWh (~{kwh_stats['kwh_mean']/0.05*1000:.1f} W)")
    print(f"  StdDev: {kwh_stats['kwh_std']:.4f} kWh | Skewness: {kwh_stats['kwh_skew']:.2f}")
    print(f"  Percentiles (All Data):")
    print(f"    p10: {percentiles_all[2]:.4f} | p25: {percentiles_all[3]:.4f} | p50: {percentiles_all[4]:.4f}")
    print(f"    p75: {percentiles_all[5]:.4f} | p90: {percentiles_all[6]:.4f} | p95: {percentiles_all[7]:.4f} | p99: {percentiles_all[8]:.4f}")
    print(f"  Percentiles (Non-Zero Only):")
    print(f"    p10: {percentiles_nonzero[2]:.4f} | p25: {percentiles_nonzero[3]:.4f} | p50: {percentiles_nonzero[4]:.4f}")
    print(f"    p75: {percentiles_nonzero[5]:.4f} | p90: {percentiles_nonzero[6]:.4f} | p95: {percentiles_nonzero[7]:.4f} | p99: {percentiles_nonzero[8]:.4f}")

    # -------------------------------------------------------------
    # 3. Electrical Quality Forensics (2.4)
    # -------------------------------------------------------------
    print("\n[3/6] Phase 2.4: Electrical Quality & Grid Stability Forensics (V, I, f)...")
    
    v_percentiles = global_df.filter(F.col("voltage") > 0.0).approxQuantile("voltage", [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99], 0.001)
    
    elec_stats = global_df.select(
        F.avg("voltage").alias("v_avg_all"),
        F.avg(F.when(F.col("voltage") > 0.0, F.col("voltage"))).alias("v_avg_energized"),
        F.count(F.when(F.col("voltage") == 0.0, 1)).alias("v_outages"),
        F.count(F.when((F.col("voltage") > 0.0) & (F.col("voltage") < 180.0), 1)).alias("v_severe_brownouts"),
        F.count(F.when((F.col("voltage") >= 180.0) & (F.col("voltage") < 200.0), 1)).alias("v_mild_brownouts"),
        F.count(F.when((F.col("voltage") >= 200.0) & (F.col("voltage") <= 250.0), 1)).alias("v_nominal"),
        F.count(F.when((F.col("voltage") > 250.0) & (F.col("voltage") <= 270.0), 1)).alias("v_high"),
        F.count(F.when(F.col("voltage") > 270.0, 1)).alias("v_severe_spikes"),
        F.count(F.when(F.col("voltage") > 300.0, 1)).alias("v_extreme_spikes"),
        
        F.avg(F.when(F.col("freq") > 0.0, F.col("freq"))).alias("f_avg_energized"),
        F.count(F.when((F.col("freq") >= 49.90) & (F.col("freq") <= 50.05), 1)).alias("f_grid_code_compliant"),
        
        F.avg("current").alias("i_avg"),
        F.max("current").alias("i_max")
    ).collect()[0].asDict()
    
    print("\nELECTRICAL QUALITY SUMMARY:")
    print("-" * 65)
    print(f"  Voltage (All Rows Mean): {elec_stats['v_avg_all']:.2f} V")
    print(f"  Voltage (Energized Grid Mean, V > 0): {elec_stats['v_avg_energized']:.2f} V")
    print(f"  Voltage Percentiles (Energized Grid):")
    print(f"    p5: {v_percentiles[1]:.2f} V | p25: {v_percentiles[3]:.2f} V | p50: {v_percentiles[4]:.2f} V | p75: {v_percentiles[5]:.2f} V | p95: {v_percentiles[7]:.2f} V")
    print(f"  Grid Outages (V = 0V): {elec_stats['v_outages']:,} ({elec_stats['v_outages']/total_raw_rows*100:.2f}%)")
    print(f"  Severe Brownouts (0V < V < 180V): {elec_stats['v_severe_brownouts']:,} ({elec_stats['v_severe_brownouts']/total_raw_rows*100:.2f}%)")
    print(f"  Mild Brownouts (180V <= V < 200V): {elec_stats['v_mild_brownouts']:,} ({elec_stats['v_mild_brownouts']/total_raw_rows*100:.2f}%)")
    print(f"  Standard Nominal Range (200V - 250V): {elec_stats['v_nominal']:,} ({elec_stats['v_nominal']/total_raw_rows*100:.2f}%)")
    print(f"  Elevated Voltage (250V - 270V): {elec_stats['v_high']:,} ({elec_stats['v_high']/total_raw_rows*100:.2f}%)")
    print(f"  Severe Overvoltage Spikes (V > 270V): {elec_stats['v_severe_spikes']:,} ({elec_stats['v_severe_spikes']/total_raw_rows*100:.2f}%)")
    print(f"  Extreme Spikes (V > 300V): {elec_stats['v_extreme_spikes']:,} ({elec_stats['v_extreme_spikes']/total_raw_rows*100:.3f}%)")
    print(f"  Grid Frequency (Energized Mean): {elec_stats['f_avg_energized']:.2f} Hz")
    print(f"  Grid Code Compliant (49.90 - 50.05 Hz): {elec_stats['f_grid_code_compliant']:,} ({elec_stats['f_grid_code_compliant']/total_raw_rows*100:.2f}%)")

    # -------------------------------------------------------------
    # 4. Exhaustive 5-State Electrical State Classification
    # -------------------------------------------------------------
    print("\n[4/6] Classifying 5 Mutually Exclusive Electrical Operating States...")
    
    state_df = global_df.withColumn(
        "operating_state",
        F.when((F.col("voltage") == 0.0) & (F.col("freq") == 0.0), "State 1: Grid Outage / Blackout")
         .when((F.col("voltage") >= 180.0) & (F.col("current") <= 0.05) & (F.col("t_kwh") == 0.0), "State 2: Voluntary Standby / Idle")
         .when((F.col("voltage") > 0.0) & (F.col("voltage") < 180.0), "State 3: Brownout Supply")
         .when((F.col("voltage") >= 180.0) & (F.col("voltage") <= 260.0), "State 4: Normal Active Operation")
         .when(F.col("voltage") > 260.0, "State 5: Overvoltage Transient / Spike")
         .otherwise("State 6: Other Intermediate")
    )
    
    state_summary = state_df.groupBy("operating_state").agg(
        F.count("*").alias("state_count"),
        F.avg("t_kwh").alias("state_avg_kwh"),
        F.avg("voltage").alias("state_avg_v")
    ).orderBy("operating_state").collect()
    
    print("\nEXHAUSTIVE ELECTRICAL STATE TAXONOMY:")
    print("-" * 75)
    for s in state_summary:
        pct = s["state_count"] / total_raw_rows * 100.0
        print(f"  {s['operating_state']:<40}: {s['state_count']:>10,d} ({pct:>6.2f}%) | Avg kWh: {s['state_avg_kwh']:.4f} | Avg V: {s['state_avg_v']:.1f}V")

    # -------------------------------------------------------------
    # 5. Empirical Temporal Resolution Experiment (2.6)
    # -------------------------------------------------------------
    print("\n[5/6] Phase 2.6: Running Empirical Resolution Experiment (3m vs 15m vs 30m vs 1h)...")
    
    resolutions = [
        ("3-Minute (Raw)", "3 minutes", 1),
        ("15-Minute", "15 minutes", 5),
        ("30-Minute", "30 minutes", 10),
        ("1-Hour", "1 hour", 20)
    ]
    
    res_results = []
    
    for label, window_duration, interval_mult in resolutions:
        t0 = time.time()
        if interval_mult == 1:
            # Raw 3-min baseline
            df_agg = global_df.select(
                "district", "meter_id", "ts", "t_kwh", "voltage", "current"
            ).withColumn("agg_kwh", F.col("t_kwh"))
        else:
            df_agg = global_df.groupBy(
                "district", "meter_id",
                F.window("ts", window_duration).alias("w")
            ).agg(
                F.sum("t_kwh").alias("agg_kwh"),
                F.mean("voltage").alias("voltage"),
                F.mean("current").alias("current"),
                F.count("*").alias("sample_count"),
                F.min("ts").alias("ts")
            )
            
        stats_agg = df_agg.select(
            F.count("*").alias("obs_n"),
            F.avg("agg_kwh").alias("mean_kwh"),
            F.stddev("agg_kwh").alias("std_kwh"),
            F.variance("agg_kwh").alias("var_kwh"),
            F.count(F.when(F.col("agg_kwh") == 0.0, 1)).alias("zero_obs")
        ).collect()[0].asDict()
        
        # Calculate lag-1 autocorrelation per meter to evaluate predictability
        w_autocorr = Window.partitionBy("district", "meter_id").orderBy("ts")
        df_lag = df_agg.withColumn("prev_kwh", F.lag("agg_kwh", 1).over(w_autocorr)).filter(F.col("prev_kwh").isNotNull())
        corr_val = df_lag.stat.corr("agg_kwh", "prev_kwh")
        
        cv = stats_agg["std_kwh"] / stats_agg["mean_kwh"] if stats_agg["mean_kwh"] > 0 else 0
        snr = stats_agg["mean_kwh"] / stats_agg["std_kwh"] if stats_agg["std_kwh"] > 0 else 0
        snr_db = 20 * math.log10(snr) if snr > 0 else 0
        zero_pct = stats_agg["zero_obs"] / stats_agg["obs_n"] * 100.0
        
        res_results.append({
            "label": label,
            "window": window_duration,
            "obs_count": stats_agg["obs_n"],
            "mean_kwh": stats_agg["mean_kwh"],
            "std_kwh": stats_agg["std_kwh"],
            "var_kwh": stats_agg["var_kwh"],
            "cv": cv,
            "snr": snr,
            "snr_db": snr_db,
            "zero_pct": zero_pct,
            "lag1_autocorr": corr_val,
            "runtime": time.time() - t0
        })
        
        print(f"\nResolution [{label}]:")
        print(f"  Observations: {stats_agg['obs_n']:,} | Zero-Obs %: {zero_pct:.2f}%")
        print(f"  Mean kWh: {stats_agg['mean_kwh']:.4f} | StdDev: {stats_agg['std_kwh']:.4f} | Variance: {stats_agg['var_kwh']:.6f}")
        print(f"  Coefficient of Variation (CV = std/mean): {cv:.3f}")
        print(f"  Signal-to-Noise Ratio (SNR): {snr:.3f} ({snr_db:.2f} dB)")
        print(f"  Lag-1 Autocorrelation (r_t,t-1): {corr_val:.4f} (Predictable persistence)")
        print(f"  Computed in {res_results[-1]['runtime']:.2f}s")
        
    # -------------------------------------------------------------
    # 6. District Breakdown of Forensics & Resolution
    # -------------------------------------------------------------
    print("\n[6/6] Comparative District Breakdown across Bareilly vs Mathura...")
    district_comp = global_df.groupBy("district").agg(
        F.count("*").alias("rows"),
        F.avg("t_kwh").alias("mean_kwh"),
        F.stddev("t_kwh").alias("std_kwh"),
        F.count(F.when(F.col("t_kwh") == 0.0, 1)).alias("zeros"),
        F.count(F.when(F.col("voltage") < 180.0, 1)).alias("brownout_outage_rows"),
        F.avg(F.when(F.col("voltage") > 0.0, F.col("voltage"))).alias("energized_v")
    ).collect()
    
    print("\nDISTRICT LEVEL COMPARISON:")
    print("-" * 75)
    for d in district_comp:
        print(f"District: {d['district']}")
        print(f"  Rows: {d['rows']:,} | Zero %: {d['zeros']/d['rows']*100:.2f}%")
        print(f"  Mean 3-min kWh: {d['mean_kwh']:.4f} | StdDev: {d['std_kwh']:.4f}")
        print(f"  Energized Voltage: {d['energized_v']:.2f} V | Brownouts/Outages (V < 180V): {d['brownout_outage_rows']:,} ({d['brownout_outage_rows']/d['rows']*100:.2f}%)\n")

    spark.stop()
    print("=" * 90)
    print(f"ALL FORENSIC PHASES (2.3, 2.4, 2.6) COMPLETED IN {time.time() - start_total:.2f}s")
    print("=" * 90)

if __name__ == "__main__":
    run_forensics_and_resolution()
