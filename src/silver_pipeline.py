"""
Phase 4.1.1: Silver Ingestion & Validation Pipeline (Hardened Production Version)
Authoritative Specifications:
  - DATA_QUALITY_POLICY.md v1.1
  - CANONICAL_DATA_CONTRACT.md v1.1

Key Enhancements:
  1. Genuine StructType-based CSV ingestion passed directly to spark.read.schema(RAW_TELEMETRY_SCHEMA).csv(...)
  2. Multi-rule Quarantine Auditing with explicit rejection_reason tags in data/interim/quarantine_audit.parquet
  3. Strict Primary Key Deduplication on (district, meter_id, ts)
  4. Closed 5-State Electrical State Machine (0 State-6 records)
  5. Partitioned Parquet Serialization to data/interim/validated_3min.parquet (district, file_year)
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

# Authoritative Explicit StructType Schema for all CEEW Raw CSV Files
RAW_TELEMETRY_SCHEMA = StructType([
    StructField("x_Timestamp", StringType(), False),
    StructField("t_kWh", DoubleType(), True),
    StructField("z_Avg Voltage (Volt)", DoubleType(), True),
    StructField("z_Avg Current (Amp)", DoubleType(), True),
    StructField("y_Freq (Hz)", DoubleType(), True),
    StructField("meter", StringType(), False)
])

def build_silver_layer():
    print("=" * 90)
    print("PHASE 4.1.1: HARDENED SILVER PIPELINE (STRUCTTYPE INGESTION & QUARANTINE REASONS)")
    print("=" * 90)
    
    start_total = time.time()
    
    # 1. Initialize SparkSession
    spark = SparkSession.builder \
        .appName("BDA-ML-Silver-Pipeline-Hardened") \
        .master("local[*]") \
        .config("spark.driver.memory", "6g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.session.timeZone", "Asia/Kolkata") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")
    
    raw_dir = os.path.abspath("data/raw")
    silver_dir = os.path.abspath("data/interim/validated_3min.parquet")
    quarantine_dir = os.path.abspath("data/interim/quarantine_audit.parquet")
    
    raw_manifest = [
        ("Bareilly", 2019, "SM Cleaned Data BR2019.csv"),
        ("Bareilly", 2020, "CEEW - Smart meter data Bareilly 2020.csv"),
        ("Bareilly", 2021, "CEEW - Smart meter data Bareilly 2021.csv"),
        ("Mathura", 2019, "CEEW - Smart meter data Mathura 2019.csv"),
        ("Mathura", 2020, "CEEW - Smart meter data Mathura 2020.csv"),
        ("Mathura", 2021, "SM Cleaned Data MH2021.csv"),
    ]
    
    print("\n[1/5] Ingesting 6 raw CSV files via explicit StructType schema...")
    dfs = []
    total_raw_count = 0
    
    for district, yr, fname in raw_manifest:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing raw data file: {fpath}")
            
        # Strict StructType ingestion — no runtime type inference
        df_file = spark.read \
            .option("header", "true") \
            .schema(RAW_TELEMETRY_SCHEMA) \
            .csv(fpath)
            
        file_row_cnt = df_file.count()
        total_raw_count += file_row_cnt
        
        # Standardize column naming according to Canonical Data Contract
        df_file = df_file \
            .withColumnRenamed("x_Timestamp", "ts_str") \
            .withColumnRenamed("t_kWh", "t_kwh") \
            .withColumnRenamed("z_Avg Voltage (Volt)", "voltage") \
            .withColumnRenamed("z_Avg Current (Amp)", "current") \
            .withColumnRenamed("y_Freq (Hz)", "freq") \
            .withColumnRenamed("meter", "meter_id_raw") \
            .withColumn("district", F.lit(district)) \
            .withColumn("file_year", F.lit(yr).cast(IntegerType())) \
            .withColumn("meter_id", F.trim(F.col("meter_id_raw")))
            
        dfs.append(df_file.select(
            "district", "file_year", "meter_id", "ts_str", 
            "t_kwh", "voltage", "current", "freq"
        ))
        print(f"  [StructType Ingested] {fname} ({district} {yr}): {file_row_cnt:,} rows")
        
    unified_raw = dfs[0]
    for other in dfs[1:]:
        unified_raw = unified_raw.unionByName(other)
        
    print(f"\nTotal Unified Raw Ingestion Count: {total_raw_count:,}")
    
    # -------------------------------------------------------------
    # 2. Strict Timestamp Parsing
    # -------------------------------------------------------------
    print("\n[2/5] Parsing Timestamps into TimestampType...")
    
    parsed_df = unified_raw.withColumn(
        "ts",
        F.coalesce(
            F.to_timestamp(F.col("ts_str"), "yyyy-MM-dd HH:mm:ss"),
            F.to_timestamp(F.col("ts_str"), "yyyy-MM-dd HH:mm"),
            F.to_timestamp(F.col("ts_str"), "dd-MM-yyyy HH:mm:ss"),
            F.to_timestamp(F.col("ts_str"), "dd-MM-yyyy HH:mm")
        )
    )
     
    # -------------------------------------------------------------
    # 3. Three-Tier Validation & Explicit Quarantine Auditing
    # -------------------------------------------------------------
    print("\n[3/5] Evaluating Validation Rules & Attaching Quarantine Audit Reasons...")
    
    # Build explicit rejection reason strings for any violating record
    evaluated_df = parsed_df.withColumn(
        "rejection_reason",
        F.concat_ws("; ",
            F.when(F.col("ts").isNull(), F.lit("INVALID_OR_UNPARSEABLE_TIMESTAMP")),
            F.when((F.col("ts") < F.to_timestamp(F.lit("2019-05-01 00:00:00"))) | (F.col("ts") > F.to_timestamp(F.lit("2021-10-31 23:59:59"))), F.lit("TIMESTAMP_OUT_OF_BOUNDS")),
            F.when(F.col("meter_id").isNull() | (F.col("meter_id") == ""), F.lit("EMPTY_OR_NULL_METER_ID")),
            F.when(F.col("t_kwh").isNull(), F.lit("NULL_ACTIVE_ENERGY")),
            F.when((F.col("t_kwh") < 0.0) | (F.col("t_kwh") > 0.500), F.lit("ACTIVE_ENERGY_OUT_OF_BOUNDS")),
            F.when(F.col("voltage").isNull(), F.lit("NULL_VOLTAGE")),
            F.when((F.col("voltage") < 0.0) | (F.col("voltage") > 700.0), F.lit("VOLTAGE_OUT_OF_BOUNDS")),
            F.when(F.col("current").isNull(), F.lit("NULL_CURRENT")),
            F.when((F.col("current") < 0.0) | (F.col("current") > 150.0), F.lit("CURRENT_OUT_OF_BOUNDS")),
            F.when(F.col("freq").isNull(), F.lit("NULL_FREQUENCY")),
            F.when((F.col("freq") < 0.0) | ((F.col("freq") > 0.0) & (F.col("freq") < 40.0)) | (F.col("freq") > 60.0), F.lit("FREQUENCY_OUT_OF_BOUNDS"))
        )
    )
    
    valid_rows_df = evaluated_df.filter(F.col("rejection_reason") == "").drop("rejection_reason", "ts_str")
    quarantine_rows_df = evaluated_df.filter(F.col("rejection_reason") != "")
    
    quarantine_count = quarantine_rows_df.count()
    print(f"  Quarantine / Rejection Count (Tier 3 Violations): {quarantine_count:,}")
    
    if quarantine_count > 0:
        print(f"  Persisting {quarantine_count} invalid records with explicit audit reasons to {quarantine_dir}...")
        quarantine_rows_df.write.mode("overwrite").parquet(quarantine_dir)
        
    # Deduplicate on composite logical key (district, meter_id, ts)
    dedup_df = valid_rows_df.dropDuplicates(["district", "meter_id", "ts"])
    valid_count = dedup_df.count()
    print(f"  Validated & Deduplicated Valid Rows: {valid_count:,}")

    # -------------------------------------------------------------
    # 4. Closed 5-State Electrical Machine & Quality Flags
    # -------------------------------------------------------------
    print("\n[4/5] Executing Closed 5-State Electrical Classifier & Attaching Quality Flags...")
    
    # Closed 5-state partition (Zero State-6 records invariant)
    silver_df = dedup_df.withColumn(
        "operating_state",
        F.when(F.col("voltage") == 0.0, F.lit(1))
         .when((F.col("voltage") >= 180.0) & (F.col("current") <= 0.05) & (F.col("t_kwh") == 0.0), F.lit(2))
         .when((F.col("voltage") > 0.0) & (F.col("voltage") < 180.0), F.lit(3))
         .when((F.col("voltage") >= 180.0) & (F.col("voltage") <= 260.0), F.lit(4))
         .when(F.col("voltage") > 260.0, F.lit(5))
         .otherwise(F.lit(6))
    )
    
    # Specific Quality Indicator Flags conforming strictly to CANONICAL_DATA_CONTRACT.md v1.1
    silver_df = silver_df.withColumn(
        "is_outage", F.when(F.col("voltage") == 0.0, F.lit(1)).otherwise(F.lit(0))
    ).withColumn(
        "is_standby", F.when((F.col("voltage") >= 180.0) & (F.col("current") <= 0.05) & (F.col("t_kwh") == 0.0), F.lit(1)).otherwise(F.lit(0))
    ).withColumn(
        "is_brownout", F.when((F.col("voltage") > 0.0) & (F.col("voltage") < 180.0), F.lit(1)).otherwise(F.lit(0))
    ).withColumn(
        "is_voltage_high", F.when((F.col("voltage") > 260.0) & (F.col("voltage") <= 300.0), F.lit(1)).otherwise(F.lit(0))
    ).withColumn(
        "is_voltage_spike", F.when(F.col("voltage") > 300.0, F.lit(1)).otherwise(F.lit(0))
    ).withColumn(
        "is_extreme_load", F.when(F.col("t_kwh") > 0.300, F.lit(1)).otherwise(F.lit(0))
    )
    
    # Select exact columns conforming to CANONICAL_DATA_CONTRACT.md v1.1
    final_silver_df = silver_df.select(
        "district",
        "meter_id",
        "ts",
        "file_year",
        "t_kwh",
        "voltage",
        "current",
        "freq",
        "operating_state",
        "is_outage",
        "is_standby",
        "is_brownout",
        "is_voltage_high",
        "is_voltage_spike",
        "is_extreme_load"
    )
    
    # -------------------------------------------------------------
    # 5. Persist Silver Layer to Parquet
    # -------------------------------------------------------------
    print(f"\n[5/5] Writing Hardened Silver Parquet to: {silver_dir} (Partitioned by district, file_year)...")
    
    final_silver_df.write \
        .mode("overwrite") \
        .partitionBy("district", "file_year") \
        .parquet(silver_dir)
        
    elapsed = time.time() - start_total
    print("\n" + "=" * 90)
    print(f"HARDENED SILVER PIPELINE COMPLETE! Persisted {valid_count:,} validated records in {elapsed:.2f}s")
    print("=" * 90)
    
    spark.stop()

if __name__ == "__main__":
    build_silver_layer()
