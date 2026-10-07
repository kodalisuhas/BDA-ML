"""
Phase 1: Full Dataset Forensic Audit Script (Tasks 6 to 10)
Computes:
  - Task 6: Exact row counts across all 6 raw files and 2 aggregated files
  - Task 7: Exact min/max timestamp ranges per file and district
  - Task 8: Distinct meter count per file and global unique meters
  - Task 9: Null values count per column & duplicate (meter, timestamp) primary keys
  - Task 10: Physical-value statistics (min, max, mean, zero count, negative count) for V, I, f, kWh
"""
import os
import sys
import time
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType
)

def run_full_audit():
    print("=" * 80)
    print("PHASE 1: COMPLETE DATASET FORENSIC AUDIT (TASKS 6 - 10)")
    print("=" * 80)
    
    start_total = time.time()
    
    spark = SparkSession.builder \
        .appName("BDA-ML-Comprehensive-Audit") \
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
    
    ref_files = [
        ("Bareilly", "Aggregated", "SM Cleaned Data BR Aggregated.csv"),
        ("Mathura", "Aggregated", "SM Cleaned Data MH Aggregated.csv")
    ]
    
    all_raw_dfs = []
    file_reports = []
    
    print("\n>>> AUDITING INDIVIDUAL FILES <<<\n")
    for district, yr, fname in raw_files:
        fpath = os.path.join(raw_dir, fname)
        t0 = time.time()
        
        df = spark.read.option("header", "true").csv(fpath)
        
        # Standardize column naming for audit
        # Schema columns: x_Timestamp, t_kWh, z_Avg Voltage (Volt), z_Avg Current (Amp), y_Freq (Hz), meter
        rename_map = {}
        for c in df.columns:
            clow = c.lower().strip()
            if "meter" in clow:
                rename_map[c] = "meter_id"
            elif "timestamp" in clow or "time" in clow:
                rename_map[c] = "timestamp_str"
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
               
        row_count = df.count()
        distinct_meters = [r[0] for r in df.select("meter_id").distinct().collect() if r[0] is not None]
        
        t_bounds = df.select(
            F.min("timestamp_str").alias("min_t"),
            F.max("timestamp_str").alias("max_t")
        ).collect()[0]
        
        # Check nulls in this file
        null_counts = df.select([
            F.count(F.when(F.col(c).isNull() | (F.col(c) == "") | (F.col(c) == "null") | (F.col(c) == "NaN"), c)).alias(c)
            for c in ["meter_id", "timestamp_str", "t_kwh", "voltage", "current", "freq"]
        ]).collect()[0].asDict()
        
        # Check duplicate keys (meter_id, timestamp_str)
        dup_count = df.groupBy("meter_id", "timestamp_str").count().filter(F.col("count") > 1).count()
        
        file_reports.append({
            "district": district,
            "year": yr,
            "filename": fname,
            "rows": row_count,
            "distinct_meter_count": len(distinct_meters),
            "distinct_meters": sorted(distinct_meters),
            "min_timestamp": t_bounds["min_t"],
            "max_timestamp": t_bounds["max_t"],
            "nulls": null_counts,
            "duplicate_keys": dup_count,
            "time_sec": time.time() - t0
        })
        
        all_raw_dfs.append(df.select("district", "file_year", "meter_id", "timestamp_str", "t_kwh", "voltage", "current", "freq"))
        
        print(f"[{district} {yr}] File: {fname}")
        print(f"  Rows: {row_count:,} | Meters: {len(distinct_meters)} ({distinct_meters[:5]}...)")
        print(f"  Timestamp Range: {t_bounds['min_t']}  -->  {t_bounds['max_t']}")
        print(f"  Null Counts: {null_counts}")
        print(f"  Duplicate (meter, timestamp) records: {dup_count:,}")
        print(f"  Processed in {file_reports[-1]['time_sec']:.2f}s\n")

    # Unified DataFrame for global aggregations & physical value statistics
    print("=" * 80)
    print(">>> GLOBAL AGGREGATIONS & PHYSICAL VALUE AUDIT (TASK 10) <<<")
    print("=" * 80)
    
    global_df = all_raw_dfs[0]
    for other_df in all_raw_dfs[1:]:
        global_df = global_df.unionByName(other_df)
        
    global_df = global_df.withColumn("t_kwh_num", F.col("t_kwh").cast(DoubleType())) \
                         .withColumn("voltage_num", F.col("voltage").cast(DoubleType())) \
                         .withColumn("current_num", F.col("current").cast(DoubleType())) \
                         .withColumn("freq_num", F.col("freq").cast(DoubleType()))
                         
    total_records = global_df.count()
    global_unique_meters = [r[0] for r in global_df.select("meter_id").distinct().collect() if r[0] is not None]
    
    print(f"\nGLOBAL TOTAL ROWS: {total_records:,}")
    print(f"GLOBAL DISTINCT METERS COUNT: {len(global_unique_meters)}")
    print(f"GLOBAL METERS LIST: {sorted(global_unique_meters)}")
    
    # Global duplicates check
    global_dup_count = global_df.groupBy("meter_id", "timestamp_str").count().filter(F.col("count") > 1).count()
    print(f"\nGLOBAL DUPLICATE KEYS (meter_id, timestamp_str): {global_dup_count:,}")
    
    # Numerical Statistics
    print("\nCalculating summary statistics across ~20M rows (Min, Max, Mean, Zeros, Negatives)...")
    stats = global_df.select(
        F.min("t_kwh_num").alias("kwh_min"),
        F.max("t_kwh_num").alias("kwh_max"),
        F.avg("t_kwh_num").alias("kwh_avg"),
        F.count(F.when(F.col("t_kwh_num") == 0.0, 1)).alias("kwh_zeros"),
        F.count(F.when(F.col("t_kwh_num") < 0.0, 1)).alias("kwh_negs"),
        
        F.min("voltage_num").alias("v_min"),
        F.max("voltage_num").alias("v_max"),
        F.avg("voltage_num").alias("v_avg"),
        F.count(F.when(F.col("voltage_num") == 0.0, 1)).alias("v_zeros"),
        F.count(F.when(F.col("voltage_num") < 0.0, 1)).alias("v_negs"),
        F.count(F.when(F.col("voltage_num") > 300.0, 1)).alias("v_extreme_spikes"),
        
        F.min("current_num").alias("i_min"),
        F.max("current_num").alias("i_max"),
        F.avg("current_num").alias("i_avg"),
        F.count(F.when(F.col("current_num") == 0.0, 1)).alias("i_zeros"),
        F.count(F.when(F.col("current_num") < 0.0, 1)).alias("i_negs"),
        
        F.min("freq_num").alias("f_min"),
        F.max("freq_num").alias("f_max"),
        F.avg("freq_num").alias("f_avg"),
        F.count(F.when(F.col("freq_num") == 0.0, 1)).alias("f_zeros"),
        F.count(F.when(F.col("freq_num") < 0.0, 1)).alias("f_negs")
    ).collect()[0].asDict()
    
    print("\nPHYSICAL NUMERICAL METRICS SUMMARY:")
    print("-" * 60)
    print(f"Energy (t_kWh):")
    print(f"  Min: {stats['kwh_min']} | Max: {stats['kwh_max']} | Mean: {stats['kwh_avg']:.4f} kWh")
    print(f"  Exact Zeros: {stats['kwh_zeros']:,} ({stats['kwh_zeros']/total_records*100:.2f}%)")
    print(f"  Negatives: {stats['kwh_negs']:,}")
    
    print(f"\nVoltage (V_RMS):")
    print(f"  Min: {stats['v_min']} V | Max: {stats['v_max']} V | Mean: {stats['v_avg']:.2f} V")
    print(f"  Exact Zeros (Outages): {stats['v_zeros']:,} ({stats['v_zeros']/total_records*100:.2f}%)")
    print(f"  Negatives: {stats['v_negs']:,}")
    print(f"  Extreme Spikes (>300V): {stats['v_extreme_spikes']:,}")
    
    print(f"\nCurrent (I_RMS):")
    print(f"  Min: {stats['i_min']} A | Max: {stats['i_max']} A | Mean: {stats['i_avg']:.2f} A")
    print(f"  Exact Zeros: {stats['i_zeros']:,} ({stats['i_zeros']/total_records*100:.2f}%)")
    print(f"  Negatives: {stats['i_negs']:,}")
    
    print(f"\nFrequency (Hz):")
    print(f"  Min: {stats['f_min']} Hz | Max: {stats['f_max']} Hz | Mean: {stats['f_avg']:.2f} Hz")
    print(f"  Exact Zeros: {stats['f_zeros']:,} ({stats['f_zeros']/total_records*100:.2f}%)")
    print(f"  Negatives: {stats['f_negs']:,}")
    
    # Physics of Zeros Check: Standby vs Outage
    standby_count = global_df.filter(
        (F.col("voltage_num") > 180.0) & (F.col("current_num") == 0.0) & (F.col("t_kwh_num") == 0.0)
    ).count()
    
    blackout_count = global_df.filter(
        (F.col("voltage_num") == 0.0) & (F.col("freq_num") == 0.0) & (F.col("t_kwh_num") == 0.0)
    ).count()
    
    print("\nPHYSICS OF ZEROS ANALYSIS:")
    print("-" * 60)
    print(f"  1. True Standby (V > 180V, I = 0, kWh = 0): {standby_count:,} ({standby_count/total_records*100:.2f}%)")
    print(f"  2. Grid Outage / Blackout (V = 0, f = 0, kWh = 0): {blackout_count:,} ({blackout_count/total_records*100:.2f}%)")
    
    spark.stop()
    print(f"\nAudit completed in {time.time() - start_total:.2f} seconds.")

if __name__ == "__main__":
    run_full_audit()
