"""
Phase 4.2: Gold Hourly Aggregation Pipeline
Authoritative Specifications:
  - DATA_QUALITY_POLICY.md v1.1
  - CANONICAL_DATA_CONTRACT.md v1.1

Input:
  data/interim/validated_3min.parquet (Silver Layer)

Output:
  data/processed/canonical_hourly.parquet (Gold Layer)
  Partitioned by: district, file_year
"""
import os
import sys
import time

# Ensure Windows Hadoop binaries are active
hadoop_home = r"C:\Users\kodal\hadoop"
os.environ["HADOOP_HOME"] = hadoop_home
os.environ["PATH"] = os.path.join(hadoop_home, "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
)
import pyspark.sql.functions as F

def build_gold_layer():
    print("=" * 90)
    print("PHASE 4.2: GOLD HOURLY PIPELINE (CLOCK-HOUR AGGREGATION & METRICS)")
    print("=" * 90)
    
    start_total = time.time()
    
    # 1. Initialize SparkSession
    spark = SparkSession.builder \
        .appName("BDA-ML-Gold-Pipeline") \
        .master("local[*]") \
        .config("spark.driver.memory", "6g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.session.timeZone", "Asia/Kolkata") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")
    
    silver_path = os.path.abspath("data/interim/validated_3min.parquet")
    gold_path = os.path.abspath("data/processed/canonical_hourly.parquet")
    
    if not os.path.exists(silver_path):
        raise FileNotFoundError(f"Missing Silver layer input at {silver_path}")
        
    print(f"\n[1/4] Reading Silver Layer from: {silver_path}...")
    silver_df = spark.read.parquet(silver_path)
    silver_row_count = silver_df.count()
    print(f"  Loaded Silver Records: {silver_row_count:,}")
    
    # -------------------------------------------------------------
    # 2. Clock-Hour Aggregation via Spark window()
    # -------------------------------------------------------------
    print("\n[2/4] Aggregating 3-minute records into 1-hour clock windows...")
    
    # Clock hour aggregation grouped by (district, meter_id, window(ts, '1 hour'))
    hourly_agg_df = silver_df.groupBy(
        F.col("district"),
        F.col("meter_id"),
        F.window(F.col("ts"), "1 hour").alias("hour_window")
    ).agg(
        F.min("file_year").alias("file_year"),
        F.sum("t_kwh").alias("raw_kwh_sum"),
        F.count("*").cast(IntegerType()).alias("sample_count"),
        F.mean("voltage").alias("mean_voltage"),
        F.mean("current").alias("mean_current"),
        F.mean("freq").alias("mean_freq"),
        F.sum("is_outage").alias("outage_samples"),
        F.sum("is_standby").alias("standby_samples"),
        F.sum("is_brownout").alias("brownout_samples"),
        F.sum(F.when(F.col("voltage") > 270.0, 1).otherwise(0)).alias("spike_samples")
    )
    
    # -------------------------------------------------------------
    # 3. Formulate Ratios, Hourly Scaling, and Quality Flags
    # -------------------------------------------------------------
    print("\n[3/4] Formulating Linear Scaling, Quality Ratios, and Sub-System Flags...")
    
    # Calculate window timestamps
    gold_df = hourly_agg_df.withColumn(
        "window_start", F.col("hour_window.start")
    ).withColumn(
        "window_end", F.col("hour_window.end")
    )
    
    # Calculate Missing-at-Random scaled hourly_kwh = raw_kwh_sum * (20 / sample_count)
    gold_df = gold_df.withColumn(
        "hourly_kwh",
        F.round(F.col("raw_kwh_sum") * (F.lit(20.0) / F.col("sample_count")), 6)
    ).withColumn(
        "raw_kwh_sum", F.round(F.col("raw_kwh_sum"), 6)
    ).withColumn(
        "mean_voltage", F.round(F.col("mean_voltage"), 4)
    ).withColumn(
        "mean_current", F.round(F.col("mean_current"), 4)
    ).withColumn(
        "mean_freq", F.round(F.col("mean_freq"), 4)
    )
    
    # Compute Ratios [0.0, 1.0]
    gold_df = gold_df.withColumn(
        "outage_ratio", F.round(F.col("outage_samples") / F.col("sample_count"), 4)
    ).withColumn(
        "standby_ratio", F.round(F.col("standby_samples") / F.col("sample_count"), 4)
    ).withColumn(
        "brownout_ratio", F.round(F.col("brownout_samples") / F.col("sample_count"), 4)
    ).withColumn(
        "spike_ratio", F.round(F.col("spike_samples") / F.col("sample_count"), 4)
    )
    
    # Quality & Sub-System Modeling Flags
    gold_df = gold_df.withColumn(
        "is_complete_hour",
        F.when(F.col("sample_count") >= 18, F.lit(1)).otherwise(F.lit(0))
    ).withColumn(
        "is_outage_hour",
        F.when(F.col("outage_ratio") >= 0.50, F.lit(1)).otherwise(F.lit(0))
    ).withColumn(
        "is_valid_forecast_target",
        F.when((F.col("sample_count") >= 18) & (F.col("outage_ratio") < 0.50), F.lit(1)).otherwise(F.lit(0))
    )
    
    # Final Select conforming to CANONICAL_DATA_CONTRACT.md v1.1
    final_gold_df = gold_df.select(
        "district",
        "meter_id",
        "window_start",
        "window_end",
        "file_year",
        "hourly_kwh",
        "raw_kwh_sum",
        "sample_count",
        "mean_voltage",
        "mean_current",
        "mean_freq",
        "outage_ratio",
        "standby_ratio",
        "brownout_ratio",
        "spike_ratio",
        "is_complete_hour",
        "is_outage_hour",
        "is_valid_forecast_target"
    )
    
    # -------------------------------------------------------------
    # 4. Persist Gold Layer to Parquet
    # -------------------------------------------------------------
    print(f"\n[4/4] Serializing Gold Layer to: {gold_path} (Partitioned by district, file_year)...")
    
    final_gold_df.write \
        .mode("overwrite") \
        .partitionBy("district", "file_year") \
        .parquet(gold_path)
        
    gold_row_count = final_gold_df.count()
    elapsed = time.time() - start_total
    
    print("\n" + "=" * 90)
    print(f"GOLD PIPELINE COMPLETE! Persisted {gold_row_count:,} canonical hourly records in {elapsed:.2f}s")
    print("=" * 90)
    
    spark.stop()

if __name__ == "__main__":
    build_gold_layer()
