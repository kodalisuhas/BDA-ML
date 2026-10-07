"""
Automated Verification Test Suite for Phase 4.2 Gold Layer (Hardened v1.1)
Verifies compliance against DATA_QUALITY_POLICY.md v1.1 and CANONICAL_DATA_CONTRACT.md v1.1
"""
import os
import sys

# Ensure Windows Hadoop binaries are active
hadoop_home = r"C:\Users\kodal\hadoop"
os.environ["HADOOP_HOME"] = hadoop_home
os.environ["PATH"] = os.path.join(hadoop_home, "bin") + os.pathsep + os.environ.get("PATH", "")

import pytest
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
)

@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder \
        .appName("Test-Gold-Layer-Hardened") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.session.timeZone", "Asia/Kolkata") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()

@pytest.fixture(scope="session")
def gold_df(spark):
    gold_path = os.path.abspath("data/processed/canonical_hourly.parquet")
    assert os.path.exists(gold_path), f"Gold parquet directory not found at {gold_path}"
    df = spark.read.parquet(gold_path)
    return df

@pytest.fixture(scope="session")
def silver_df(spark):
    silver_path = os.path.abspath("data/interim/validated_3min.parquet")
    assert os.path.exists(silver_path), f"Silver parquet directory not found at {silver_path}"
    df = spark.read.parquet(silver_path)
    return df

def test_gold_row_count(gold_df):
    """Verify exact total row count matches the established deterministic ground truth (1,074,001)."""
    count = gold_df.count()
    assert count == 1074001, f"Expected exactly 1,074,001 rows in Gold layer, got {count}"

def test_gold_schema_matches_contract(gold_df):
    """Verify exact column names and data types conform to CANONICAL_DATA_CONTRACT.md v1.1."""
    expected_types = {
        "district": StringType(),
        "meter_id": StringType(),
        "window_start": TimestampType(),
        "window_end": TimestampType(),
        "file_year": IntegerType(),
        "hourly_kwh": DoubleType(),
        "raw_kwh_sum": DoubleType(),
        "sample_count": IntegerType(),
        "mean_voltage": DoubleType(),
        "mean_current": DoubleType(),
        "mean_freq": DoubleType(),
        "outage_ratio": DoubleType(),
        "standby_ratio": DoubleType(),
        "brownout_ratio": DoubleType(),
        "spike_ratio": DoubleType(),
        "is_complete_hour": IntegerType(),
        "is_outage_hour": IntegerType(),
        "is_valid_forecast_target": IntegerType()
    }
    
    actual_fields = {f.name: f.dataType for f in gold_df.schema.fields}
    
    for name, expected_dtype in expected_types.items():
        assert name in actual_fields, f"Missing required column in Gold schema: {name}"
        assert isinstance(actual_fields[name], type(expected_dtype)), \
            f"Column {name} data type mismatch: expected {expected_dtype}, got {actual_fields[name]}"

def test_gold_zero_nulls(gold_df):
    """Assert zero nulls across all 18 Gold columns."""
    for col_name in gold_df.columns:
        null_count = gold_df.filter(F.col(col_name).isNull()).count()
        assert null_count == 0, f"Found {null_count} nulls in column '{col_name}'"

def test_gold_primary_key_uniqueness(gold_df):
    """Assert zero duplicate composite primary keys (district, meter_id, window_start)."""
    dup_count = gold_df.groupBy("district", "meter_id", "window_start").count().filter(F.col("count") > 1).count()
    assert dup_count == 0, f"Found {dup_count} duplicate (district, meter_id, window_start) keys in Gold layer"

def test_gold_window_duration(gold_df):
    """Assert window_end > window_start and window duration is exactly 3600 seconds (1 hour)."""
    invalid_windows = gold_df.filter(
        (F.unix_timestamp("window_end") - F.unix_timestamp("window_start")) != 3600
    ).count()
    assert invalid_windows == 0, f"Found {invalid_windows} windows that are not exactly 3600 seconds long"

def test_gold_sample_count_bounds(gold_df):
    """Assert sample_count is strictly bounded between 1 and 20 observations."""
    invalid_samples = gold_df.filter((F.col("sample_count") < 1) | (F.col("sample_count") > 20)).count()
    assert invalid_samples == 0, f"Found {invalid_samples} records with sample_count outside [1, 20]"

def test_gold_energy_non_negative(gold_df):
    """Assert both raw_kwh_sum and hourly_kwh are non-negative."""
    invalid_raw = gold_df.filter(F.col("raw_kwh_sum") < 0.0).count()
    assert invalid_raw == 0, f"Found {invalid_raw} records with negative raw_kwh_sum"
    
    invalid_scaled = gold_df.filter(F.col("hourly_kwh") < 0.0).count()
    assert invalid_scaled == 0, f"Found {invalid_scaled} records with negative hourly_kwh"

def test_gold_complete_20_sample_exact_equality(gold_df):
    """Assert that for all complete 20/20 hours, hourly_kwh == raw_kwh_sum."""
    unequal_count = gold_df.filter(
        (F.col("sample_count") == 20) & (F.abs(F.col("hourly_kwh") - F.col("raw_kwh_sum")) > 1e-4)
    ).count()
    assert unequal_count == 0, f"Found {unequal_count} 20-sample hours where hourly_kwh != raw_kwh_sum"

def test_gold_scaling_formula_adherence(gold_df):
    """Assert hourly_kwh == raw_kwh_sum * (20 / sample_count) for all rows."""
    expected_expr = F.col("raw_kwh_sum") * (F.lit(20.0) / F.col("sample_count"))
    violating_scaling = gold_df.filter(
        F.abs(F.col("hourly_kwh") - expected_expr) > 1e-4
    ).count()
    assert violating_scaling == 0, f"Found {violating_scaling} records violating 20/N scaling formula"

def test_gold_ratios_within_unit_interval(gold_df):
    """Assert all quality and operational ratios are in [0.0, 1.0]."""
    for ratio_col in ["outage_ratio", "standby_ratio", "brownout_ratio", "spike_ratio"]:
        invalid_ratios = gold_df.filter((F.col(ratio_col) < 0.0) | (F.col(ratio_col) > 1.0)).count()
        assert invalid_ratios == 0, f"Found {invalid_ratios} records where {ratio_col} is outside [0.0, 1.0]"

def test_gold_flag_logical_consistency(gold_df):
    """Assert is_complete_hour, is_outage_hour, and is_valid_forecast_target match logical predicates."""
    # is_complete_hour == 1 iff sample_count >= 18
    complete_mismatches = gold_df.filter((F.col("is_complete_hour") == 1) != (F.col("sample_count") >= 18)).count()
    assert complete_mismatches == 0, f"Found {complete_mismatches} mismatches in is_complete_hour flag"
    
    # is_outage_hour == 1 iff outage_ratio >= 0.50
    outage_mismatches = gold_df.filter((F.col("is_outage_hour") == 1) != (F.col("outage_ratio") >= 0.50)).count()
    assert outage_mismatches == 0, f"Found {outage_mismatches} mismatches in is_outage_hour flag"
    
    # is_valid_forecast_target == 1 iff is_complete_hour == 1 and is_outage_hour == 0
    target_mismatches = gold_df.filter(
        (F.col("is_valid_forecast_target") == 1) != ((F.col("is_complete_hour") == 1) & (F.col("is_outage_hour") == 0))
    ).count()
    assert target_mismatches == 0, f"Found {target_mismatches} mismatches in is_valid_forecast_target flag"

def test_gold_energy_conservation_against_silver(gold_df, silver_df):
    """Assert that total raw energy sum in Gold exactly matches total energy sum in Silver with high precision (< 1e-3 kWh)."""
    gold_total_raw_energy = gold_df.select(F.sum("raw_kwh_sum")).collect()[0][0]
    silver_total_raw_energy = silver_df.select(F.sum("t_kwh")).collect()[0][0]
    
    delta = abs(gold_total_raw_energy - silver_total_raw_energy)
    assert delta < 1e-3, f"Energy conservation mismatch: delta={delta:.6f} kWh exceeds 1e-3 tolerance"

def test_gold_spike_ratio_definition(gold_df, silver_df):
    """Assert that Gold spike_ratio strictly aggregates the V > 270V condition across Silver observations."""
    silver_hourly = silver_df.filter(F.col("district") == "Mathura").filter(F.col("file_year") == 2021) \
        .groupBy("district", "meter_id", F.window("ts", "1 hour").alias("w")) \
        .agg(
            F.sum(F.when(F.col("voltage") > 270.0, 1).otherwise(0)).alias("expected_spike_samples"),
            F.count("*").alias("expected_sample_count")
        ).withColumn(
            "expected_spike_ratio", F.round(F.col("expected_spike_samples") / F.col("expected_sample_count"), 4)
        ).withColumn("window_start", F.col("w.start"))
        
    joined = gold_df.filter(F.col("district") == "Mathura").filter(F.col("file_year") == 2021) \
        .join(silver_hourly, on=["district", "meter_id", "window_start"])
        
    mismatches = joined.filter(F.abs(F.col("spike_ratio") - F.col("expected_spike_ratio")) > 1e-4).count()
    assert mismatches == 0, f"Found {mismatches} hourly rows where spike_ratio does not match V > 270.0 aggregation"
