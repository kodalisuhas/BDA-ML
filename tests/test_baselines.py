"""
Automated Verification Test Suite for Phase 6.2: Domain Baseline Evaluation
===========================================================================
Verifies compliance of baseline evaluations against:
  - docs/SYSTEM_A_MODELING_CONTRACT.md (Phase 6.1)
  - docs/FORECASTING_FEATURE_STORE_CONTRACT.md (Phase 5.1)
  - Option C, Path C1 (Strict Common Intersection) specifications

Tests covered:
  1. test_01_b1_lag0_identity: B1 prediction is identical to lag_0h.
  2. test_02_b4_rolling24_identity: B4 prediction is identical to rolling_mean_24h.
  3. test_03_b2_temporal_alignment_real: B2 retrieves exact Y_{t-23} from t-24h origin on feature store.
  4. test_04_b3_temporal_alignment_real: B3 retrieves exact Y_{t-167} from t-168h origin on feature store.
  5. test_05_cross_meter_isolation_synthetic: Direct test verifying lookups cannot cross meter IDs.
  6. test_06_exact_timestamp_offsets_synthetic: Direct test verifying exact offset matching (+24h, +168h).
  7. test_07_missing_predecessors_synthetic: Absent historical rows evaluate to NULL and cohort exclusion.
  8. test_08_invalid_and_nonfinite_target_rejection_synthetic: Rejection of NaN, +/-inf, null, negative, outage.
  9. test_09_no_join_fan_out: 1-to-1 join guarantees zero fan-out across eligible rows.
 10. test_10_cross_boundary_history_retrieval: Early VAL instances consume late TRAIN history across split boundary.
 11. test_11_test_split_evaluation_isolation: TEST split is excluded from baseline evaluation and output JSON.
 12. test_12_common_cohort_uniformity_and_finite_safety: Identical row count, zero nulls/non-finites for all 4 baselines.
 13. test_13_metric_mathematical_invariants: Domain mathematical axioms (RMSE >= MAE >= 0, R2 <= 1.0, WAPE >= 0).
 14. test_14_primary_key_integrity_rejection_synthetic: Deliberate duplicate keys and null primary key fields are rejected with ValueError.
"""

import os
import sys
import math
import json
import datetime
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure Python executable and Windows Hadoop binaries are active
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
HADOOP_HOME = r"C:\Users\kodal\hadoop"
if os.path.exists(HADOOP_HOME):
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    os.environ["PATH"] = os.path.join(HADOOP_HOME, "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from src.evaluate_baselines import (
    align_baseline_predictions,
    build_historical_target_lookups,
    evaluate_split_baselines,
    is_valid_finite_energy,
    validate_historical_lookup_uniqueness
)


@pytest.fixture(scope="session")
def spark():
    """Shared SparkSession for test suite."""
    spark = SparkSession.builder \
        .appName("Test-SystemA-Baselines") \
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
def aligned_df(feature_df):
    """Produce aligned baseline predictions DataFrame for TRAIN and VAL."""
    return align_baseline_predictions(feature_df, target_splits=("TRAIN", "VAL")).cache()


def test_01_b1_lag0_identity(aligned_df):
    """Invariant 1: B1 predictions match lag_0h with zero divergence."""
    diff_count = aligned_df.filter(F.col("b1_pred") != F.col("lag_0h")).count()
    assert diff_count == 0, f"B1 prediction differs from lag_0h in {diff_count} rows"


def test_02_b4_rolling24_identity(aligned_df):
    """Invariant 2: B4 predictions match rolling_mean_24h with zero divergence."""
    diff_count = aligned_df.filter(F.col("b4_pred") != F.col("rolling_mean_24h")).count()
    assert diff_count == 0, f"B4 prediction differs from rolling_mean_24h in {diff_count} rows"


def test_03_b2_temporal_alignment_real(aligned_df, feature_df):
    """
    Invariant 3: B2 prediction equals the exact target_hourly_kwh of the row
    whose window_start is current.window_start - 24 hours for the same meter.
    """
    sample_rows = aligned_df.filter(
        (F.col("split") == "VAL") & F.col("b2_pred").isNotNull()
    ).limit(50).collect()

    assert len(sample_rows) == 50, "Could not sample 50 matched rows for B2"

    for row in sample_rows:
        dist = row["district"]
        m_id = row["meter_id"]
        ws = row["window_start"]
        b2_pred = row["b2_pred"]

        hist_rows = feature_df.filter(
            (F.col("district") == dist) &
            (F.col("meter_id") == m_id) &
            (F.col("window_start") == ws - F.expr("INTERVAL 24 HOUR"))
        ).collect()

        assert len(hist_rows) == 1, f"Expected 1 historical row at t-24h, found {len(hist_rows)}"
        hist_target = hist_rows[0]["target_hourly_kwh"]
        assert math.isclose(b2_pred, hist_target, rel_tol=1e-5), \
            f"B2 pred {b2_pred} != historical target {hist_target} for {dist}/{m_id} at {ws}"


def test_04_b3_temporal_alignment_real(aligned_df, feature_df):
    """
    Invariant 4: B3 prediction equals the exact target_hourly_kwh of the row
    whose window_start is current.window_start - 168 hours for the same meter.
    """
    sample_rows = aligned_df.filter(
        (F.col("split") == "VAL") & F.col("b3_pred").isNotNull()
    ).limit(50).collect()

    assert len(sample_rows) == 50, "Could not sample 50 matched rows for B3"

    for row in sample_rows:
        dist = row["district"]
        m_id = row["meter_id"]
        ws = row["window_start"]
        b3_pred = row["b3_pred"]

        hist_rows = feature_df.filter(
            (F.col("district") == dist) &
            (F.col("meter_id") == m_id) &
            (F.col("window_start") == ws - F.expr("INTERVAL 168 HOUR"))
        ).collect()

        assert len(hist_rows) == 1, f"Expected 1 historical row at t-168h, found {len(hist_rows)}"
        hist_target = hist_rows[0]["target_hourly_kwh"]
        assert math.isclose(b3_pred, hist_target, rel_tol=1e-5), \
            f"B3 pred {b3_pred} != historical target {hist_target} for {dist}/{m_id} at {ws}"


def test_05_cross_meter_isolation_synthetic(spark):
    """
    Invariant 5: Direct synthetic test verifying historical lookups cannot cross meter boundaries.
    Meter_A must never retrieve Meter_B's target, and Meter_B with no history must receive NULL.
    """
    t0 = datetime.datetime(2020, 1, 1, 12, 0, 0)
    t_24 = t0 + datetime.timedelta(hours=24)

    # Meter_A has history at t0 and target instance at t_24.
    # Meter_B has target instance at t_24 but NO history at t0.
    data = [
        # History at t0: Meter_A has target 1.25, Meter_C has target 9.99
        ("Bareilly", "Meter_A", t0, 1.25, 1, "TRAIN", 1, 1.25, 1.25),
        ("Bareilly", "Meter_C", t0, 9.99, 1, "TRAIN", 1, 9.99, 9.99),
        # Target instances at t_24: Meter_A and Meter_B
        ("Bareilly", "Meter_A", t_24, 1.50, 1, "VAL", 1, 1.30, 1.20),
        ("Bareilly", "Meter_B", t_24, 2.50, 1, "VAL", 1, 2.10, 2.00),
    ]
    schema = (
        "district STRING, meter_id STRING, window_start TIMESTAMP, "
        "target_hourly_kwh DOUBLE, target_is_valid INT, split STRING, "
        "is_valid_forecast_instance INT, lag_0h DOUBLE, rolling_mean_24h DOUBLE"
    )
    df_synth = spark.createDataFrame(data, schema=schema)
    aligned_synth = align_baseline_predictions(df_synth, target_splits=("VAL",))

    res = {r["meter_id"]: r["b2_pred"] for r in aligned_synth.collect()}

    # Meter_A should have retrieved Meter_A's target (1.25)
    assert res["Meter_A"] == 1.25, f"Meter_A retrieved {res['Meter_A']} instead of 1.25"
    # Meter_B had no t0 row; must receive None, not Meter_A or Meter_C
    assert res["Meter_B"] is None, f"Meter_B cross-contaminated with {res['Meter_B']}"


def test_06_exact_timestamp_offsets_synthetic(spark):
    """
    Invariant 6: Direct synthetic test verifying exact temporal matching:
    B2 matches ONLY the row at exactly t-24h origin; B3 matches ONLY the row at exactly t-168h origin.
    Neighboring offsets (t-23h, t-25h, t-167h, t-169h) must NOT match.
    """
    base = datetime.datetime(2020, 2, 1, 0, 0, 0)
    target_time = base + datetime.timedelta(hours=200)

    data = [
        # Candidate offsets for Meter_1:
        ("Bareilly", "M1", target_time - datetime.timedelta(hours=23), 0.23, 1, "TRAIN", 1, 0.23, 0.23),
        ("Bareilly", "M1", target_time - datetime.timedelta(hours=24), 0.24, 1, "TRAIN", 1, 0.24, 0.24), # True B2 origin
        ("Bareilly", "M1", target_time - datetime.timedelta(hours=25), 0.25, 1, "TRAIN", 1, 0.25, 0.25),
        ("Bareilly", "M1", target_time - datetime.timedelta(hours=167), 1.67, 1, "TRAIN", 1, 1.67, 1.67),
        ("Bareilly", "M1", target_time - datetime.timedelta(hours=168), 1.68, 1, "TRAIN", 1, 1.68, 1.68), # True B3 origin
        ("Bareilly", "M1", target_time - datetime.timedelta(hours=169), 1.69, 1, "TRAIN", 1, 1.69, 1.69),
        # Target row at target_time
        ("Bareilly", "M1", target_time, 0.50, 1, "VAL", 1, 0.45, 0.40),
    ]
    schema = (
        "district STRING, meter_id STRING, window_start TIMESTAMP, "
        "target_hourly_kwh DOUBLE, target_is_valid INT, split STRING, "
        "is_valid_forecast_instance INT, lag_0h DOUBLE, rolling_mean_24h DOUBLE"
    )
    df_synth = spark.createDataFrame(data, schema=schema)
    aligned = align_baseline_predictions(df_synth, target_splits=("VAL",)).collect()

    assert len(aligned) == 1
    row = aligned[0]
    assert row["b2_pred"] == 0.24, f"B2 expected 0.24, got {row['b2_pred']}"
    assert row["b3_pred"] == 1.68, f"B3 expected 1.68, got {row['b3_pred']}"


def test_07_missing_predecessors_synthetic(spark):
    """
    Invariant 7: Synthetic test verifying that missing historical predecessors (due to telemetry gaps)
    evaluate to NULL predictions and cause the row to be excluded from the common cohort.
    """
    t_target = datetime.datetime(2020, 5, 1, 12, 0, 0)
    # Row with NO historical rows at all
    data = [
        ("Bareilly", "M_Orphan", t_target, 0.40, 1, "VAL", 1, 0.35, 0.30)
    ]
    schema = (
        "district STRING, meter_id STRING, window_start TIMESTAMP, "
        "target_hourly_kwh DOUBLE, target_is_valid INT, split STRING, "
        "is_valid_forecast_instance INT, lag_0h DOUBLE, rolling_mean_24h DOUBLE"
    )
    df_synth = spark.createDataFrame(data, schema=schema)
    aligned = align_baseline_predictions(df_synth, target_splits=("VAL",))

    r = aligned.collect()[0]
    assert r["b2_pred"] is None, "Missing B2 predecessor should produce NULL"
    assert r["b3_pred"] is None, "Missing B3 predecessor should produce NULL"

    # Verify exclusion from common cohort
    common_rows = aligned.filter(
        is_valid_finite_energy(F.col("b2_pred")) & is_valid_finite_energy(F.col("b3_pred"))
    ).count()
    assert common_rows == 0, f"Expected 0 common rows for orphan instance, got {common_rows}"


def test_08_invalid_and_nonfinite_target_rejection_synthetic(spark):
    """
    Invariant 8: Rigorous synthetic regression test for all non-finite and invalid edge cases:
    NaN, +inf, -inf, null, negative, and target_is_valid == 0 must all be rejected and set to NULL.
    """
    schema = "district STRING, meter_id STRING, window_start TIMESTAMP, target_hourly_kwh DOUBLE, target_is_valid INT"
    data = [
        ("Bareilly", "M1", datetime.datetime(2020, 1, 1, 0, 0, 0), 0.5, 1),            # Valid: 0.5
        ("Bareilly", "M1", datetime.datetime(2020, 1, 1, 1, 0, 0), 0.0, 0),            # Invalid: outage flag
        ("Bareilly", "M1", datetime.datetime(2020, 1, 1, 2, 0, 0), -0.2, 1),           # Invalid: negative
        ("Bareilly", "M1", datetime.datetime(2020, 1, 1, 3, 0, 0), float("nan"), 1),    # Invalid: NaN
        ("Bareilly", "M1", datetime.datetime(2020, 1, 1, 4, 0, 0), float("inf"), 1),    # Invalid: +infinity
        ("Bareilly", "M1", datetime.datetime(2020, 1, 1, 5, 0, 0), float("-inf"), 1),   # Invalid: -infinity
        ("Bareilly", "M1", datetime.datetime(2020, 1, 1, 6, 0, 0), None, 1),            # Invalid: Null
    ]
    test_df = spark.createDataFrame(data, schema=schema)
    lut_b2, _ = build_historical_target_lookups(test_df)

    res = lut_b2.collect()
    pred_map = {r["b2_join_ws"].strftime("%Y-%m-%d %H:%M:%S"): r["b2_pred"] for r in res}

    # Only 00:00:00 should yield a valid b2_pred (at 2020-01-02 00:00:00)
    assert pred_map["2020-01-02 00:00:00"] == 0.5
    assert pred_map["2020-01-02 01:00:00"] is None, "Failed to reject target_is_valid == 0"
    assert pred_map["2020-01-02 02:00:00"] is None, "Failed to reject negative value"
    assert pred_map["2020-01-02 03:00:00"] is None, "Failed to reject NaN"
    assert pred_map["2020-01-02 04:00:00"] is None, "Failed to reject +infinity"
    assert pred_map["2020-01-02 05:00:00"] is None, "Failed to reject -infinity"
    assert pred_map["2020-01-02 06:00:00"] is None, "Failed to reject Null"


def test_09_no_join_fan_out(aligned_df, feature_df):
    """
    Invariant 9: 1-to-1 join guarantees zero fan-out across all eligible TRAIN + VAL instances.
    Aligned row count must exactly match eligible input count for those splits.
    """
    expected_count = feature_df.filter(
        (F.col("is_valid_forecast_instance") == 1) &
        F.col("split").isin(["TRAIN", "VAL"])
    ).count()
    aligned_count = aligned_df.count()
    assert aligned_count == expected_count, \
        f"Join fan-out detected: aligned count {aligned_count} != expected count {expected_count}"


def test_10_cross_boundary_history_retrieval(aligned_df, feature_df):
    """
    Invariant 10: Early VAL instances (in September 2020) successfully retrieve historical predecessors
    from late TRAIN (in August 2020) across the split boundary without artificial truncation.
    """
    # Sample VAL rows in early September 2020 (within 24h/168h of boundary)
    boundary_val_rows = aligned_df.filter(
        (F.col("split") == "VAL") &
        (F.col("window_start") >= "2020-09-01 00:00:00") &
        (F.col("window_start") <= "2020-09-02 00:00:00") &
        F.col("b2_pred").isNotNull()
    ).limit(10).collect()

    assert len(boundary_val_rows) > 0, "No boundary VAL rows found with valid B2"

    for r in boundary_val_rows:
        dist = r["district"]
        m_id = r["meter_id"]
        ws = r["window_start"]
        b2_pred = r["b2_pred"]

        # Historical predecessor was in August 2020 (TRAIN split)
        hist_row = feature_df.filter(
            (F.col("district") == dist) &
            (F.col("meter_id") == m_id) &
            (F.col("window_start") == ws - F.expr("INTERVAL 24 HOUR"))
        ).collect()[0]

        assert hist_row["split"] == "TRAIN", f"Historical row was not in TRAIN: {hist_row['split']}"
        assert math.isclose(b2_pred, hist_row["target_hourly_kwh"], rel_tol=1e-5)


def test_11_test_split_evaluation_isolation(aligned_df):
    """
    Invariant 11: TEST split evaluation isolation:
    1. Aligned DataFrame contains zero TEST split rows.
    2. Output metrics JSON contains only TRAIN and VAL splits; TEST is absent.
    """
    test_rows_in_aligned = aligned_df.filter(F.col("split") == "TEST").count()
    assert test_rows_in_aligned == 0, f"Found {test_rows_in_aligned} TEST rows in aligned DataFrame"

    report_json_path = os.path.abspath("reports/baselines/baseline_metrics.json")
    if os.path.exists(report_json_path):
        with open(report_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        splits = data["splits"]
        assert "TRAIN" in splits, "TRAIN split missing from metrics JSON"
        assert "VAL" in splits, "VAL split missing from metrics JSON"
        assert "TEST" not in splits, "TEST split was evaluated in violation of modeling contract"


def test_12_common_cohort_uniformity_and_finite_safety(aligned_df):
    """
    Invariant 12: In both TRAIN and VAL splits:
    - Common cohort row count is identical across all four baselines.
    - Zero nulls, zero NaNs, and zero infinities across target and all 4 baselines.
    """
    for sp in ["TRAIN", "VAL"]:
        sp_results = evaluate_split_baselines(aligned_df, sp)
        N = sp_results["cohort_audit"]["common_cohort_evaluated"]
        assert N > 0, f"Common cohort for {sp} has 0 rows"

        baselines = sp_results["baselines"]
        for b_id, b_metrics in baselines.items():
            assert b_metrics["evaluated_rows"] == N, \
                f"{b_id} on {sp} evaluated {b_metrics['evaluated_rows']} rows != common cohort {N}"
            assert math.isfinite(b_metrics["rmse_kwh"])
            assert math.isfinite(b_metrics["mae_kwh"])
            assert math.isfinite(b_metrics["r2"])
            assert math.isfinite(b_metrics["wape_pct"])
            assert math.isfinite(b_metrics["nrmse_pct"])


def test_13_metric_mathematical_invariants(aligned_df):
    """
    Invariant 13: Evaluated metrics satisfy domain mathematical axioms:
      - RMSE >= MAE >= 0
      - R2 <= 1.0
      - WAPE >= 0
      - NRMSE >= 0
    """
    val_results = evaluate_split_baselines(aligned_df, "VAL")
    for b_id, metrics in val_results["baselines"].items():
        rmse = metrics["rmse_kwh"]
        mae = metrics["mae_kwh"]
        r2 = metrics["r2"]
        wape = metrics["wape_pct"]
        nrmse = metrics["nrmse_pct"]

        assert mae >= 0.0, f"{b_id} MAE {mae} < 0"
        assert rmse >= mae, f"{b_id} RMSE {rmse} < MAE {mae}"
        assert r2 <= 1.0, f"{b_id} R2 {r2} > 1.0"
        assert wape >= 0.0, f"{b_id} WAPE {wape} < 0"
        assert nrmse >= 0.0, f"{b_id} NRMSE {nrmse} < 0"


def test_14_primary_key_integrity_rejection_synthetic(spark):
    """
    Invariant 14: Rigorous synthetic test verifying that:
      a) Duplicate (district, meter_id, window_start) keys
      b) Null primary key components (district, meter_id, or window_start)
    in the historical lookup source are detected and rejected with ValueError,
    preventing silent join fan-out, dropped matches, and downstream metric corruption.
    """
    t0 = datetime.datetime(2020, 1, 1, 0, 0, 0)
    schema = (
        "district STRING, meter_id STRING, window_start TIMESTAMP, "
        "target_hourly_kwh DOUBLE, target_is_valid INT, split STRING, "
        "is_valid_forecast_instance INT, lag_0h DOUBLE, rolling_mean_24h DOUBLE"
    )

    # 1. Duplicate keys check
    data_dup = [
        ("Bareilly", "M1", t0, 0.50, 1, "TRAIN", 1, 0.40, 0.40),
        ("Bareilly", "M1", t0, 0.60, 1, "TRAIN", 1, 0.40, 0.40),  # Deliberate duplicate key
    ]
    df_dup = spark.createDataFrame(data_dup, schema=schema)
    with pytest.raises(ValueError, match="Duplicate primary key detected"):
        validate_historical_lookup_uniqueness(df_dup)
    with pytest.raises(ValueError, match="Duplicate primary key detected"):
        build_historical_target_lookups(df_dup)
    with pytest.raises(ValueError, match="Duplicate primary key detected"):
        align_baseline_predictions(df_dup, target_splits=("TRAIN",))

    # 2. Null primary key component check (defensive hardening)
    data_null_pk = [
        ("Bareilly", None, t0, 0.50, 1, "TRAIN", 1, 0.40, 0.40),  # Null meter_id
    ]
    df_null_pk = spark.createDataFrame(data_null_pk, schema=schema)
    with pytest.raises(ValueError, match="Null primary key component detected"):
        validate_historical_lookup_uniqueness(df_null_pk)
    with pytest.raises(ValueError, match="Null primary key component detected"):
        build_historical_target_lookups(df_null_pk)
    with pytest.raises(ValueError, match="Null primary key component detected"):
        align_baseline_predictions(df_null_pk, target_splits=("TRAIN",))


