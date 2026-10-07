"""
Automated Verification Test Suite for Phase 4.1 Silver Layer
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
        .appName("Test-Silver-Layer") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.session.timeZone", "Asia/Kolkata") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()

@pytest.fixture(scope="session")
def silver_df(spark):
    silver_path = os.path.abspath("data/interim/validated_3min.parquet")
    assert os.path.exists(silver_path), f"Silver parquet directory not found at {silver_path}"
    df = spark.read.parquet(silver_path)
    return df

def test_silver_row_count(silver_df):
    """Verify total row count matches validated rows (21,392,743 valid rows + 1,686 quarantine = 21,394,429)."""
    count = silver_df.count()
    assert count == 21392743, f"Expected 21,392,743 rows in Silver layer, got {count}"

def test_silver_schema_matches_contract(silver_df):
    """Verify exact column names and data types conform to CANONICAL_DATA_CONTRACT.md v1.1."""
    expected_types = {
        "district": StringType(),
        "meter_id": StringType(),
        "ts": TimestampType(),
        "file_year": IntegerType(),
        "t_kwh": DoubleType(),
        "voltage": DoubleType(),
        "current": DoubleType(),
        "freq": DoubleType(),
        "operating_state": IntegerType(),
        "is_outage": IntegerType(),
        "is_standby": IntegerType(),
        "is_brownout": IntegerType(),
        "is_voltage_high": IntegerType(),
        "is_voltage_spike": IntegerType(),
        "is_extreme_load": IntegerType()
    }
    
    actual_fields = {f.name: f.dataType for f in silver_df.schema.fields}
    
    for name, expected_dtype in expected_types.items():
        assert name in actual_fields, f"Missing required column in Silver schema: {name}"
        assert isinstance(actual_fields[name], type(expected_dtype)), \
            f"Column {name} data type mismatch: expected {expected_dtype}, got {actual_fields[name]}"

def test_silver_zero_nulls(silver_df):
    """Assert zero nulls across all columns."""
    for col_name in silver_df.columns:
        null_count = silver_df.filter(F.col(col_name).isNull()).count()
        assert null_count == 0, f"Found {null_count} nulls in column '{col_name}'"

def test_silver_primary_key_uniqueness(silver_df):
    """Assert zero duplicate primary keys (district, meter_id, ts)."""
    dup_count = silver_df.groupBy("district", "meter_id", "ts").count().filter(F.col("count") > 1).count()
    assert dup_count == 0, f"Found {dup_count} duplicate (district, meter_id, ts) keys in Silver layer"

def test_silver_physical_bounds(silver_df):
    """Assert all numerical parameters strictly adhere to valid Tier 1 & Tier 2 bounds."""
    invalid_kwh = silver_df.filter((F.col("t_kwh") < 0.0) | (F.col("t_kwh") > 0.500)).count()
    assert invalid_kwh == 0, f"Found {invalid_kwh} records violating energy bounds [0.0, 0.500]"
    
    invalid_v = silver_df.filter((F.col("voltage") < 0.0) | (F.col("voltage") > 700.0)).count()
    assert invalid_v == 0, f"Found {invalid_v} records violating voltage bounds [0.0, 700.0]"
    
    invalid_i = silver_df.filter((F.col("current") < 0.0) | (F.col("current") > 150.0)).count()
    assert invalid_i == 0, f"Found {invalid_i} records violating current bounds [0.0, 150.0]"
    
    invalid_f = silver_df.filter((F.col("freq") < 0.0) | ((F.col("freq") > 0.0) & (F.col("freq") < 40.0)) | (F.col("freq") > 60.0)).count()
    assert invalid_f == 0, f"Found {invalid_f} records violating frequency bounds [0.0 or 40-60 Hz]"

def test_silver_state_machine_closed(silver_df):
    """Assert 5-state electrical taxonomy is closed: operating_state in [1..5], exactly 0 State 6."""
    state6_count = silver_df.filter(F.col("operating_state") == 6).count()
    assert state6_count == 0, f"Found {state6_count} unclassified State 6 records! State machine is not closed."
    
    distinct_states = [r[0] for r in silver_df.select("operating_state").distinct().collect()]
    assert set(distinct_states).issubset({1, 2, 3, 4, 5}), f"Unexpected states in Silver: {distinct_states}"

def test_silver_flag_logical_consistency(silver_df):
    """Assert perfect logical consistency between numerical conditions and quality indicator flags."""
    # is_outage == 1 iff voltage == 0.0
    outage_mismatches = silver_df.filter((F.col("is_outage") == 1) != (F.col("voltage") == 0.0)).count()
    assert outage_mismatches == 0, f"Found {outage_mismatches} mismatches between is_outage and voltage == 0.0"
    
    # is_brownout == 1 iff 0 < voltage < 180.0
    brownout_mismatches = silver_df.filter((F.col("is_brownout") == 1) != ((F.col("voltage") > 0.0) & (F.col("voltage") < 180.0))).count()
    assert brownout_mismatches == 0, f"Found {brownout_mismatches} mismatches between is_brownout and 0 < voltage < 180"
    
    # is_voltage_spike == 1 iff voltage > 300.0
    spike_mismatches = silver_df.filter((F.col("is_voltage_spike") == 1) != (F.col("voltage") > 300.0)).count()
    assert spike_mismatches == 0, f"Found {spike_mismatches} mismatches between is_voltage_spike and voltage > 300"
    
    # is_extreme_load == 1 iff t_kwh > 0.300
    load_mismatches = silver_df.filter((F.col("is_extreme_load") == 1) != (F.col("t_kwh") > 0.300)).count()
    assert load_mismatches == 0, f"Found {load_mismatches} mismatches between is_extreme_load and t_kwh > 0.300"
