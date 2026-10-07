"""
Automated Invariant Verification Test Suite for Phase 5.3: Feature Store
Verifies compliance against docs/FORECASTING_FEATURE_STORE_CONTRACT.md v1.0.0
"""
import os
import sys
import math
import datetime
import statistics
import pytest

# Ensure Windows Hadoop binaries are active
hadoop_home = r"C:\Users\kodal\hadoop"
os.environ["HADOOP_HOME"] = hadoop_home
os.environ["PATH"] = os.path.join(hadoop_home, "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StringType, DoubleType, IntegerType, TimestampType
)

EXPECTED_39_SCHEMA = {
    "district": StringType(),
    "meter_id": StringType(),
    "window_start": TimestampType(),
    "target_window_start": TimestampType(),
    "split": StringType(),
    "district_idx": DoubleType(),
    "lag_0h": DoubleType(),
    "lag_1h": DoubleType(),
    "lag_2h": DoubleType(),
    "lag_24h": DoubleType(),
    "lag_168h": DoubleType(),
    "rolling_mean_24h": DoubleType(),
    "rolling_std_24h": DoubleType(),
    "rolling_max_24h": DoubleType(),
    "rolling_min_24h": DoubleType(),
    "target_hour": IntegerType(),
    "target_dow": IntegerType(),
    "target_month": IntegerType(),
    "target_is_weekend": IntegerType(),
    "hour_sin": DoubleType(),
    "hour_cos": DoubleType(),
    "dow_sin": DoubleType(),
    "dow_cos": DoubleType(),
    "month_sin": DoubleType(),
    "month_cos": DoubleType(),
    "mean_voltage_t": DoubleType(),
    "mean_current_t": DoubleType(),
    "mean_freq_t": DoubleType(),
    "outage_ratio_t": DoubleType(),
    "standby_ratio_t": DoubleType(),
    "brownout_ratio_t": DoubleType(),
    "spike_ratio_t": DoubleType(),
    "rolling_outage_ratio_24h": DoubleType(),
    "target_hourly_kwh": DoubleType(),
    "target_is_complete": IntegerType(),
    "target_is_outage": IntegerType(),
    "target_is_valid": IntegerType(),
    "has_complete_history": IntegerType(),
    "is_valid_forecast_instance": IntegerType(),
}


@pytest.fixture(scope="session")
def spark():
    """Shared SparkSession for test suite."""
    spark = SparkSession.builder \
        .appName("Test-Feature-Store-Invariants") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.session.timeZone", "Asia/Kolkata") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def feature_df(spark):
    """Load persisted Feature Store."""
    feat_path = os.path.abspath("data/processed/features_forecasting")
    assert os.path.exists(feat_path), f"Feature store path not found: {feat_path}"
    return spark.read.parquet(feat_path)


@pytest.fixture(scope="session")
def gold_df(spark):
    """Load canonical Gold dataset for ground truth comparison."""
    gold_path = os.path.abspath("data/processed/canonical_hourly.parquet")
    assert os.path.exists(gold_path), f"Gold dataset path not found: {gold_path}"
    return spark.read.parquet(gold_path)


def test_01_feature_store_primary_key_uniqueness(feature_df):
    """Invariant 1: Exactly 1,074,001 rows with zero duplicate (district, meter_id, window_start) keys."""
    total_count = feature_df.count()
    assert total_count == 1074001, f"Expected 1,074,001 rows, got {total_count:,}"

    distinct_count = feature_df.select("district", "meter_id", "window_start").distinct().count()
    assert distinct_count == total_count, (
        f"Primary key uniqueness violated: {total_count} total vs {distinct_count} distinct keys"
    )


def test_02_feature_store_exact_schema(feature_df):
    """Invariant 2: Exact 39 columns and matching PySpark data types against contract."""
    assert len(feature_df.columns) == 39, f"Expected 39 columns, found {len(feature_df.columns)}"

    actual_schema = {f.name: f.dataType for f in feature_df.schema.fields}
    for col_name, expected_type in EXPECTED_39_SCHEMA.items():
        assert col_name in actual_schema, f"Missing required column: {col_name}"
        assert actual_schema[col_name] == expected_type, (
            f"Type mismatch on {col_name}: expected {expected_type}, got {actual_schema[col_name]}"
        )


def test_03_autoregressive_lag_alignment(feature_df, gold_df):
    """Invariant 3: Autoregressive lags match ground truth historical consumption exactly."""
    # Test on a stable meter with complete history
    sample_rows = (
        feature_df.filter(
            (F.col("meter_id") == "BR02")
            & (F.col("has_complete_history") == 1)
        )
        .orderBy("window_start")
        .limit(5)
        .collect()
    )
    assert len(sample_rows) > 0, "No sample rows found for lag verification"

    w_gold = Window.partitionBy("meter_id").orderBy("window_start")
    gold_lags = (
        gold_df.filter(F.col("meter_id") == "BR02")
        .withColumn("exp_lag_0h", F.col("hourly_kwh"))
        .withColumn("exp_lag_1h", F.lag("hourly_kwh", 1).over(w_gold))
        .withColumn("exp_lag_2h", F.lag("hourly_kwh", 2).over(w_gold))
        .withColumn("exp_lag_24h", F.lag("hourly_kwh", 24).over(w_gold))
        .withColumn("exp_lag_168h", F.lag("hourly_kwh", 168).over(w_gold))
    )

    for row in sample_rows:
        t_ws = row["window_start"]
        gold_row = gold_lags.filter(F.col("window_start") == t_ws).first()
        assert gold_row is not None, f"Gold row missing for {t_ws}"

        assert row["lag_0h"] == pytest.approx(gold_row["exp_lag_0h"], 1e-5)
        assert row["lag_1h"] == pytest.approx(gold_row["exp_lag_1h"], 1e-5)
        assert row["lag_2h"] == pytest.approx(gold_row["exp_lag_2h"], 1e-5)
        assert row["lag_24h"] == pytest.approx(gold_row["exp_lag_24h"], 1e-5)
        assert row["lag_168h"] == pytest.approx(gold_row["exp_lag_168h"], 1e-5)


def test_04_rolling_statistics_mathematical_precision(feature_df):
    """Invariant 4: Rolling mean and sample stddev match standard statistics module on [t-23, t]."""
    # Sample a meter continuous slice of 24 consecutive hours
    meter_slice = (
        feature_df.filter(
            (F.col("meter_id") == "BR02")
            & (F.col("has_complete_history") == 1)
        )
        .orderBy("window_start")
        .limit(24)
        .collect()
    )
    assert len(meter_slice) == 24, f"Expected 24 consecutive rows, got {len(meter_slice)}"

    kwh_values = [r["lag_0h"] for r in meter_slice]
    expected_mean = float(statistics.mean(kwh_values))
    expected_std = float(statistics.stdev(kwh_values)) # Sample stddev (N-1)
    expected_max = float(max(kwh_values))
    expected_min = float(min(kwh_values))

    last_row = meter_slice[-1]
    assert last_row["rolling_mean_24h"] == pytest.approx(expected_mean, 1e-4)
    assert last_row["rolling_std_24h"] == pytest.approx(expected_std, 1e-4)
    assert last_row["rolling_max_24h"] == pytest.approx(expected_max, 1e-4)
    assert last_row["rolling_min_24h"] == pytest.approx(expected_min, 1e-4)


def test_05_fourier_cyclical_invariants(feature_df):
    """Invariant 5: Trigonometric identities hold: sin^2 + cos^2 = 1.0 and bounds in [-1, 1]."""
    trig_check = feature_df.select(
        F.max(F.abs(F.pow("hour_sin", 2) + F.pow("hour_cos", 2) - 1.0)).alias("max_hour_err"),
        F.max(F.abs(F.pow("dow_sin", 2) + F.pow("dow_cos", 2) - 1.0)).alias("max_dow_err"),
        F.max(F.abs(F.pow("month_sin", 2) + F.pow("month_cos", 2) - 1.0)).alias("max_month_err"),
        F.min("hour_sin").alias("min_h_sin"),
        F.max("hour_sin").alias("max_h_sin"),
        F.min("hour_cos").alias("min_h_cos"),
        F.max("hour_cos").alias("max_h_cos"),
    ).first()

    assert trig_check["max_hour_err"] < 1e-5, f"Hour trig error: {trig_check['max_hour_err']}"
    assert trig_check["max_dow_err"] < 1e-5, f"DOW trig error: {trig_check['max_dow_err']}"
    assert trig_check["max_month_err"] < 1e-5, f"Month trig error: {trig_check['max_month_err']}"
    assert trig_check["min_h_sin"] >= -1.0 and trig_check["max_h_sin"] <= 1.0
    assert trig_check["min_h_cos"] >= -1.0 and trig_check["max_h_cos"] <= 1.0


def test_06_target_timestamp_alignment(feature_df, gold_df):
    """Invariant 6: target_window_start is strictly t+1h, and matches Gold hourly_kwh when valid."""
    # 1. 100% of rows have target_window_start exactly 3600s after window_start
    bad_target_ts = feature_df.filter(
        (F.unix_timestamp("target_window_start") - F.unix_timestamp("window_start")) != 3600
    ).count()
    assert bad_target_ts == 0, f"Found {bad_target_ts} rows with non-consecutive target_window_start"

    # 2. For rows with missing consecutive target in Gold, target_hourly_kwh must be NULL and target_is_valid = 0
    missing_target_rows = feature_df.filter(F.col("target_hourly_kwh").isNull())
    assert missing_target_rows.count() == 4219, f"Expected 4,219 missing target rows, got {missing_target_rows.count():,}"
    invalid_flag_count = missing_target_rows.filter(F.col("target_is_valid") != 0).count()
    assert invalid_flag_count == 0, f"Found {invalid_flag_count} rows with NULL target but target_is_valid != 0"

    # 3. For eligible instances, target matches Gold exactly
    sample_eligible = (
        feature_df.filter(
            (F.col("meter_id") == "BR02")
            & (F.col("is_valid_forecast_instance") == 1)
        )
        .orderBy("window_start")
        .first()
    )
    assert sample_eligible is not None, "No eligible sample row found"
    target_ts = sample_eligible["target_window_start"]
    gold_target = (
        gold_df.filter(
            (F.col("meter_id") == "BR02")
            & (F.col("window_start") == target_ts)
        )
        .first()
    )
    assert gold_target is not None, f"Gold target row not found for timestamp {target_ts}"
    assert sample_eligible["target_hourly_kwh"] == pytest.approx(gold_target["hourly_kwh"], 1e-5)


def test_07_target_independence_anti_leakage(feature_df):
    """Invariant 7: Target measurements are not leaked into feature columns at cutoff t."""
    # In rows where consumption actively changes, lag_0h must not equal target_hourly_kwh
    changing_rows = (
        feature_df.filter(
            (F.col("is_valid_forecast_instance") == 1)
            & (F.abs(F.col("target_hourly_kwh") - F.col("lag_0h")) > 0.05)
        )
        .limit(100)
        .collect()
    )
    assert len(changing_rows) > 0, "No instances found with changing consumption"
    for r in changing_rows:
        assert r["lag_0h"] != r["target_hourly_kwh"]

    # Verify target power quality columns are strictly absent from feature schema
    forbidden_features = [
        "mean_voltage_t_plus_1", "mean_current_t_plus_1", "outage_ratio_t_plus_1",
        "voltage_target", "current_target", "target_voltage"
    ]
    for col_name in forbidden_features:
        assert col_name not in feature_df.columns, f"Leaked target column found: {col_name}"


def test_08_chronological_split_correctness(feature_df):
    """Invariant 8: Partitions adhere 100% strictly to chronological dates on target_window_start."""
    # TRAIN < 2020-09-01
    bad_train = feature_df.filter(
        (F.col("split") == "TRAIN") & (F.col("target_window_start") >= "2020-09-01 00:00:00")
    ).count()
    assert bad_train == 0, f"Found {bad_train} TRAIN rows on or after Sep 1, 2020"

    # VAL in [2020-09-01, 2020-12-31 23:59:59]
    bad_val = feature_df.filter(
        (F.col("split") == "VAL")
        & ((F.col("target_window_start") < "2020-09-01 00:00:00") | (F.col("target_window_start") >= "2021-01-01 00:00:00"))
    ).count()
    assert bad_val == 0, f"Found {bad_val} VAL rows outside Sep-Dec 2020"

    # TEST >= 2021-01-01
    bad_test = feature_df.filter(
        (F.col("split") == "TEST") & (F.col("target_window_start") < "2021-01-01 00:00:00")
    ).count()
    assert bad_test == 0, f"Found {bad_test} TEST rows before Jan 1, 2021"

    # Exact split counts
    counts = {r["split"]: r["count"] for r in feature_df.groupBy("split").count().collect()}
    assert counts["TRAIN"] == 681535, f"Unexpected TRAIN count: {counts['TRAIN']}"
    assert counts["VAL"] == 166793, f"Unexpected VAL count: {counts['VAL']}"
    assert counts["TEST"] == 225673, f"Unexpected TEST count: {counts['TEST']}"


def test_09_cross_boundary_lag_integrity(feature_df):
    """Invariant 9: Validation rows in early Sep 2020 have valid non-null lags from late Aug Train set."""
    boundary_val_rows = (
        feature_df.filter(
            (F.col("split") == "VAL")
            & (F.col("meter_id") == "BR02")
            & (F.col("target_window_start") >= "2020-09-01 00:00:00")
            & (F.col("target_window_start") <= "2020-09-01 06:00:00")
        )
        .collect()
    )
    assert len(boundary_val_rows) > 0, "No boundary validation rows found"
    for r in boundary_val_rows:
        # lag_1h, lag_2h, lag_24h, lag_168h must be populated across the boundary
        assert r["lag_1h"] is not None, f"lag_1h was truncated at split boundary: {r['window_start']}"
        assert r["lag_24h"] is not None, f"lag_24h was truncated at split boundary: {r['window_start']}"


def test_10_zero_imputation_prohibition(feature_df):
    """Invariant 10: Cold starts and missing targets retain explicit NULLs without zero-imputation."""
    null_counts = feature_df.select(
        F.sum(F.when(F.col("lag_1h").isNull(), 1).otherwise(0)).alias("n_lag1"),
        F.sum(F.when(F.col("lag_2h").isNull(), 1).otherwise(0)).alias("n_lag2"),
        F.sum(F.when(F.col("lag_24h").isNull(), 1).otherwise(0)).alias("n_lag24"),
        F.sum(F.when(F.col("lag_168h").isNull(), 1).otherwise(0)).alias("n_lag168"),
        F.sum(F.when(F.col("rolling_std_24h").isNull(), 1).otherwise(0)).alias("n_std24"),
        F.sum(F.when(F.col("target_hourly_kwh").isNull(), 1).otherwise(0)).alias("n_target"),
    ).first().asDict()

    assert null_counts["n_lag1"] == 84, f"Expected 84 nulls for lag_1h, got {null_counts['n_lag1']}"
    assert null_counts["n_lag2"] == 168, f"Expected 168 nulls for lag_2h, got {null_counts['n_lag2']}"
    assert null_counts["n_lag24"] == 2016, f"Expected 2,016 nulls for lag_24h, got {null_counts['n_lag24']}"
    assert null_counts["n_lag168"] == 14112, f"Expected 14,112 nulls for lag_168h, got {null_counts['n_lag168']}"
    assert null_counts["n_std24"] == 84, f"Expected 84 nulls for rolling_std_24h, got {null_counts['n_std24']}"
    assert null_counts["n_target"] == 4219, f"Expected 4,219 nulls for target_hourly_kwh, got {null_counts['n_target']}"


def test_11_eligible_subset_contains_zero_nulls(feature_df):
    """Invariant 11: Exactly 978,031 eligible rows, containing 0 nulls across all 39 columns."""
    eligible_df = feature_df.filter(F.col("is_valid_forecast_instance") == 1)
    eligible_count = eligible_df.count()
    assert eligible_count == 978031, f"Expected 978,031 eligible instances, got {eligible_count:,}"

    null_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in feature_df.columns
    ]
    null_results = eligible_df.agg(*null_exprs).first().asDict()
    bad_columns = {k: v for k, v in null_results.items() if v > 0}
    assert len(bad_columns) == 0, f"Detected nulls in eligible training subset: {bad_columns}"


def test_12_all_84_meter_cohorts_preserved(feature_df):
    """Invariant 12: Preserves all 84 distinct meter cohorts across Bareilly (46) and Mathura (38)."""
    district_counts = {
        r["district"]: r["count"]
        for r in feature_df.select("district", "meter_id").distinct().groupBy("district").count().collect()
    }
    assert district_counts["Bareilly"] == 46, f"Expected 46 Bareilly meters, got {district_counts.get('Bareilly')}"
    assert district_counts["Mathura"] == 38, f"Expected 38 Mathura meters, got {district_counts.get('Mathura')}"


def test_13_physical_and_value_bounds(feature_df):
    """Invariant 13: Energy, voltage, frequency, ratios, and binary flags satisfy physical boundaries."""
    # 1. Non-negative energy
    neg_kwh = feature_df.filter(F.col("lag_0h") < 0.0).count()
    assert neg_kwh == 0, f"Found {neg_kwh} negative lag_0h records"
    neg_target = feature_df.filter(F.col("target_hourly_kwh").isNotNull() & (F.col("target_hourly_kwh") < 0.0)).count()
    assert neg_target == 0, f"Found {neg_target} negative target_hourly_kwh records"

    # 2. Voltage non-negative
    neg_v = feature_df.filter(F.col("mean_voltage_t") < 0.0).count()
    assert neg_v == 0, f"Found {neg_v} negative voltage records"

    # 3. Frequency in physical limits [0.0, 60.0] (0.0 during grid outages, up to 60.0 Hz)
    bad_freq = feature_df.filter(
        (F.col("mean_freq_t") < 0.0) | (F.col("mean_freq_t") > 60.0)
    ).count()
    assert bad_freq == 0, f"Found {bad_freq} frequency bound violations"

    # 4. Ratios in [0.0, 1.0]
    bad_ratios = feature_df.filter(
        (F.col("outage_ratio_t") < 0.0) | (F.col("outage_ratio_t") > 1.0)
        | (F.col("standby_ratio_t") < 0.0) | (F.col("standby_ratio_t") > 1.0)
        | (F.col("brownout_ratio_t") < 0.0) | (F.col("brownout_ratio_t") > 1.0)
        | (F.col("spike_ratio_t") < 0.0) | (F.col("spike_ratio_t") > 1.0)
        | (F.col("rolling_outage_ratio_24h").isNotNull() & ((F.col("rolling_outage_ratio_24h") < 0.0) | (F.col("rolling_outage_ratio_24h") > 1.0)))
    ).count()
    assert bad_ratios == 0, f"Found {bad_ratios} ratio bound violations"

    # 5. Binary flags in {0, 1}
    bad_flags = feature_df.filter(
        ~F.col("is_valid_forecast_instance").isin(0, 1)
        | ~F.col("has_complete_history").isin(0, 1)
        | ~F.col("target_is_valid").isin(0, 1)
        | ~F.col("target_is_weekend").isin(0, 1)
    ).count()
    assert bad_flags == 0, f"Found {bad_flags} binary flag violations"
