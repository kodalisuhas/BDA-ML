"""
Spark Environment Smoke Test & Single File Inspection Script
Phase 1: Dataset Intake & Verification
"""
import sys
import os
import time

def run_smoke_test(file_path: str):
    print("=" * 60)
    print("TASK 3 & 4: SPARK SMOKE TEST & FIRST FILE INSPECTION")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. Start SparkSession
    print("\n[1/5] Initializing SparkSession (local[*])...")
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    
    spark = SparkSession.builder \
        .appName("BDA-ML-Intake-Verification") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.session.timeZone", "Asia/Kolkata") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark Version: {spark.version}")
    print(f"Master: {spark.sparkContext.master}")
    
    # 2. Inspect Target File Exists
    if not os.path.exists(file_path):
        print(f"Error: Target file not found at {file_path}")
        spark.stop()
        sys.exit(1)
        
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\n[2/5] Target file: {os.path.basename(file_path)} ({file_size_mb:.2f} MB)")
    
    # 3. Read Single File (without expensive full type inference first)
    print("\n[3/5] Reading sample with header=true...")
    df_raw = spark.read.option("header", "true").csv(file_path)
    
    print("\nColumns detected (count: {}):".format(len(df_raw.columns)))
    for idx, col_name in enumerate(df_raw.columns, 1):
        print(f"  {idx}. '{col_name}'")
        
    # 4. Display Schema & Sample Records
    print("\n[4/5] Inferred String Schema:")
    df_raw.printSchema()
    
    print("\nFirst 5 Records:")
    df_raw.show(5, truncate=False)
    
    # 5. Fast Row Count & Distinct Meters on this single file
    print("[5/5] Computing quick row count and distinct meter IDs...")
    total_rows = df_raw.count()
    meter_col = [c for c in df_raw.columns if "meter" in c.lower()][0]
    distinct_meters = df_raw.select(meter_col).distinct().count()
    
    print(f"\nSingle File Summary for '{os.path.basename(file_path)}':")
    print(f"  - Total Rows: {total_rows:,}")
    print(f"  - Distinct Meters: {distinct_meters}")
    
    elapsed = time.time() - start_time
    print(f"\nElapsed execution time: {elapsed:.2f} seconds")
    print("=" * 60)
    print("Spark smoke test completed successfully!")
    print("=" * 60)
    
    spark.stop()

if __name__ == "__main__":
    target = os.path.abspath("data/raw/CEEW - Smart meter data Mathura 2019.csv")
    run_smoke_test(target)
