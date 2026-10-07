"""
Phase 2: Temporal Forensics & Meter Coverage Analysis
Tasks 2.1 & 2.2:
  - Per-meter active lifespan (first/last timestamp, total observations, active days)
  - Expected vs actual observations (local coverage vs global coverage)
  - Consecutive timestamp delta distribution (gap analysis: nominal 3m, short, medium, long, multi-day)
  - District-level comparison (Bareilly vs Mathura)
"""
import os
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, TimestampType, LongType

def run_temporal_forensics():
    print("=" * 85)
    print("PHASE 2: TEMPORAL FORENSICS & PER-METER GAP ANALYSIS (TASKS 2.1 & 2.2)")
    print("=" * 85)
    
    start_total = time.time()
    
    spark = SparkSession.builder \
        .appName("BDA-ML-Temporal-Forensics") \
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
    
    print("\n[1/4] Ingesting all 6 raw files and standardizing schemas...")
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
        
    # Parse timestamp rigorously
    # Format pattern is YYYY-MM-DD HH:mm:ss
    global_df = global_df.withColumn(
        "ts", F.to_timestamp(F.col("timestamp_raw"), "yyyy-MM-dd HH:mm:ss")
    )
    
    # Cast electrical attributes for analysis
    global_df = global_df.withColumn("t_kwh", F.col("t_kwh").cast(DoubleType())) \
                         .withColumn("voltage", F.col("voltage").cast(DoubleType())) \
                         .withColumn("current", F.col("current").cast(DoubleType())) \
                         .withColumn("freq", F.col("freq").cast(DoubleType()))
                         
    print(f"Total rows ingested: {global_df.count():,}")
    
    # -------------------------------------------------------------
    # 2. Per-Meter Lifespan & Coverage Analysis
    # -------------------------------------------------------------
    print("\n[2/4] Computing per-meter lifespan, active span, and local/global coverage...")
    
    # Global bounds
    global_bounds = global_df.select(
        F.min("ts").alias("global_min_ts"),
        F.max("ts").alias("global_max_ts")
    ).collect()[0]
    
    g_min = global_bounds["global_min_ts"]
    g_max = global_bounds["global_max_ts"]
    global_span_seconds = (g_max.timestamp() - g_min.timestamp())
    global_expected_intervals = int(global_span_seconds / 180) + 1
    
    print(f"Global Time Bounds: {g_min} to {g_max} ({global_span_seconds / 86400:.1f} days)")
    print(f"Global Theoretical 3-Minute Slots: {global_expected_intervals:,} per meter")
    
    meter_summary_df = global_df.groupBy("district", "meter_id").agg(
        F.count("*").alias("obs_count"),
        F.min("ts").alias("first_ts"),
        F.max("ts").alias("last_ts"),
        F.avg("t_kwh").alias("avg_kwh"),
        F.count(F.when((F.col("voltage") == 0.0) & (F.col("freq") == 0.0), 1)).alias("blackout_count"),
        F.count(F.when((F.col("voltage") > 180.0) & (F.col("current") == 0.0) & (F.col("t_kwh") == 0.0), 1)).alias("standby_count")
    )
    
    # Calculate local span, expected local intervals, local coverage % and global coverage %
    meter_summary_df = meter_summary_df.withColumn(
        "lifespan_seconds", F.unix_timestamp("last_ts") - F.unix_timestamp("first_ts")
    ).withColumn(
        "lifespan_days", F.round(F.col("lifespan_seconds") / 86400.0, 1)
    ).withColumn(
        "expected_local_obs", F.floor(F.col("lifespan_seconds") / 180.0) + 1
    ).withColumn(
        "local_coverage_pct", F.round((F.col("obs_count") / F.col("expected_local_obs")) * 100.0, 2)
    ).withColumn(
        "global_coverage_pct", F.round((F.col("obs_count") / F.lit(global_expected_intervals)) * 100.0, 2)
    ).withColumn(
        "blackout_pct", F.round((F.col("blackout_count") / F.col("obs_count")) * 100.0, 2)
    ).withColumn(
        "standby_pct", F.round((F.col("standby_count") / F.col("obs_count")) * 100.0, 2)
    )
    
    meter_summaries = meter_summary_df.orderBy("district", "meter_id").collect()
    
    print("\n" + "=" * 115)
    print(f"{'District':<10} {'Meter':<8} {'First Seen':<20} {'Last Seen':<20} {'Days':<6} {'Obs Count':<11} {'LocalCov%':<10} {'GlobalCov%':<11} {'Outage%':<9}")
    print("-" * 115)
    for m in meter_summaries:
        print(f"{m['district']:<10} {m['meter_id']:<8} {str(m['first_ts']):<20} {str(m['last_ts']):<20} {m['lifespan_days']:<6} {m['obs_count']:<11,d} {m['local_coverage_pct']:<10.2f} {m['global_coverage_pct']:<11.2f} {m['blackout_pct']:<9.2f}")
    print("=" * 115)
    
    # -------------------------------------------------------------
    # 3. Consecutive Timestamp Delta & Gap Analysis
    # -------------------------------------------------------------
    print("\n[3/4] Computing consecutive timestamp deltas (Delta t = t_i - t_{i-1}) per meter...")
    
    window_spec = Window.partitionBy("meter_id").orderBy("ts")
    
    gap_df = global_df.select("district", "meter_id", "ts") \
                      .withColumn("prev_ts", F.lag("ts", 1).over(window_spec)) \
                      .filter(F.col("prev_ts").isNotNull())
                      
    gap_df = gap_df.withColumn(
        "delta_seconds", F.unix_timestamp("ts") - F.unix_timestamp("prev_ts")
    ).withColumn(
        "delta_minutes", F.col("delta_seconds") / 60.0
    )
    
    # Categorize Gaps
    gap_df = gap_df.withColumn(
        "gap_category",
        F.when(F.col("delta_seconds") == 180, "1. Exact 3-min (Nominal)")
         .when((F.col("delta_seconds") > 180) & (F.col("delta_seconds") <= 900), "2. Short Gap (3m - 15m)")
         .when((F.col("delta_seconds") > 900) & (F.col("delta_seconds") <= 3600), "3. Medium Gap (15m - 1h)")
         .when((F.col("delta_seconds") > 3600) & (F.col("delta_seconds") <= 86400), "4. Long Gap (1h - 24h)")
         .when(F.col("delta_seconds") > 86400, "5. Extended Dropout (> 24h)")
         .otherwise("0. Anomaly (< 3 min)")
    )
    
    print("\nGLOBAL GAP DISTRIBUTION:")
    print("-" * 60)
    gap_stats = gap_df.groupBy("gap_category").agg(
        F.count("*").alias("interval_count"),
        F.min("delta_minutes").alias("min_delta_m"),
        F.max("delta_minutes").alias("max_delta_m"),
        F.avg("delta_minutes").alias("avg_delta_m")
    ).orderBy("gap_category").collect()
    
    total_intervals = sum(g["interval_count"] for g in gap_stats)
    for g in gap_stats:
        pct = (g["interval_count"] / total_intervals) * 100.0
        print(f"  {g['gap_category']:<30}: {g['interval_count']:>10,d} ({pct:>6.2f}%) | Range: [{g['min_delta_m']:.1f}m - {g['max_delta_m']:.1f}m]")
        
    print(f"\nTotal Consecutive Step Transitions Analyzed: {total_intervals:,}")
    
    # -------------------------------------------------------------
    # 4. District-Level Comparative Forensics
    # -------------------------------------------------------------
    print("\n[4/4] District-Level Breakdown (Bareilly vs Mathura):")
    print("-" * 80)
    district_df = global_df.groupBy("district").agg(
        F.count("*").alias("total_rows"),
        F.countDistinct("meter_id").alias("meters_count"),
        F.avg("t_kwh").alias("avg_kwh"),
        F.avg("voltage").alias("avg_v"),
        F.avg("current").alias("avg_i"),
        F.avg("freq").alias("avg_f"),
        F.count(F.when((F.col("voltage") == 0.0) & (F.col("freq") == 0.0), 1)).alias("blackout_rows"),
        F.count(F.when((F.col("voltage") > 180.0) & (F.col("current") == 0.0) & (F.col("t_kwh") == 0.0), 1)).alias("standby_rows")
    ).collect()
    
    for d in district_df:
        b_pct = (d["blackout_rows"] / d["total_rows"]) * 100.0
        s_pct = (d["standby_rows"] / d["total_rows"]) * 100.0
        print(f"District: {d['district']}")
        print(f"  Total Rows: {d['total_rows']:,} | Meter Count: {d['meters_count']}")
        print(f"  Mean Active Energy: {d['avg_kwh']:.4f} kWh (~{d['avg_kwh']/0.05*1000:.1f} W)")
        print(f"  Mean Voltage: {d['avg_v']:.2f} V | Mean Current: {d['avg_i']:.2f} A | Mean Freq: {d['avg_f']:.2f} Hz")
        print(f"  Blackouts (Outages): {d['blackout_rows']:,} ({b_pct:.2f}%)")
        print(f"  Voluntary Standby: {d['standby_rows']:,} ({s_pct:.2f}%)\n")
        
    spark.stop()
    print(f"Temporal forensics completed in {time.time() - start_total:.2f} seconds.")

if __name__ == "__main__":
    run_temporal_forensics()
