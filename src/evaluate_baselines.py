"""
System A Forecasting Domain Baseline Evaluator (Phase 6.2)
==========================================================
Evaluates the four benchmark baselines (Level 0) for next-hour smart-meter
electricity demand forecasting under Option C, Path C1 (Strict Common Intersection).

Baselines Evaluated:
--------------------
  B1: Naive Persistence (lag_0h = Y_t)
  B2: Diurnal Seasonal Naive (Y_{t-23}, retrieved via distributed join with t-24h origin target)
  B3: Weekly Seasonal Naive (Y_{t-167}, retrieved via distributed join with t-168h origin target)
  B4: Rolling 24h Mean (rolling_mean_24h = mu_24(t))

Methodology & Invariants (Option C, Path C1):
--------------------------------------------
1. Zero Data Leakage: All baseline inputs are strictly finalized at or before forecast origin t.
   - B2 target Y_{t-23} (interval [t-23, t-22)) concluded 23 hours prior to cutoff t+1.
   - B3 target Y_{t-167} (interval [t-167, t-166)) concluded 167 hours prior to cutoff t+1.
2. Temporal Exactness: B2 and B3 retrieve exact physical clock-hour targets (24h and 168h prior).
3. Historical Target Validity: Historical predecessors must satisfy target_is_valid == 1 and
   finite, non-negative bounds (not null, not NaN, not +/-inf, >= 0.0).
4. Strict Common Intersection: Evaluates the identical common cohort for B1-B4 within each split.
5. Evaluation Isolation: Evaluates TRAIN and VAL only (TEST held out exclusively for Phase 6.6).
6. Metrics Contract: RMSE, MAE, R2, WAPE, NRMSE computed in accordance with modeling contract.

Authoritative Contracts:
------------------------
  - docs/SYSTEM_A_MODELING_CONTRACT.md (Phase 6.1)
  - docs/FORECASTING_FEATURE_STORE_CONTRACT.md (Phase 5.1)
"""

import os
import sys
import time
import json
import math
import argparse
from typing import Dict, Any, Tuple

# Ensure Python executable and Windows Hadoop binaries are active
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
HADOOP_HOME = r"C:\Users\kodal\hadoop"
if os.path.exists(HADOOP_HOME):
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    os.environ["PATH"] = os.path.join(HADOOP_HOME, "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F


def build_spark_session(app_name: str = "SystemA-BaselineEvaluation") -> SparkSession:
    """Build or retrieve a local SparkSession configured for BDA-ML."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.session.timeZone", "Asia/Kolkata") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def is_valid_finite_energy(col: F.Column) -> F.Column:
    """
    Returns boolean Column ensuring an electrical energy value is non-null,
    strictly finite (not NaN, not +inf, not -inf), and non-negative (>= 0.0).
    """
    return (
        col.isNotNull() &
        (~F.isnan(col)) &
        (col < float("inf")) &
        (col > float("-inf")) &
        (col >= 0.0)
    )


def validate_historical_lookup_uniqueness(df_source: DataFrame) -> None:
    """
    Assert that (district, meter_id, window_start) forms a strictly unique and non-null
    primary key in the historical lookup source.

    Raises:
        ValueError: If null primary key components or duplicate keys are detected,
                    preventing silent join fan-out, dropped matches, and downstream metric corruption.
    """
    # 1. Non-null primary key component check
    null_pk_rows = df_source.filter(
        F.col("district").isNull() |
        F.col("meter_id").isNull() |
        F.col("window_start").isNull()
    ).head(1)
    if null_pk_rows:
        row = null_pk_rows[0]
        raise ValueError(
            f"Null primary key component detected in historical lookup source: "
            f"(district='{row['district']}', meter_id='{row['meter_id']}', window_start='{row['window_start']}'). "
            f"Historical lookup source requires strictly non-null primary keys."
        )

    # 2. Duplicate key check
    dup_rows = (
        df_source.groupBy("district", "meter_id", "window_start")
        .count()
        .filter(F.col("count") > 1)
        .head(1)
    )
    if dup_rows:
        dup = dup_rows[0]
        raise ValueError(
            f"Duplicate primary key detected in historical lookup source for "
            f"(district='{dup['district']}', meter_id='{dup['meter_id']}', window_start='{dup['window_start']}') "
            f"with count={dup['count']}. Historical lookup source must be strictly unique to prevent join fan-out."
        )


def build_historical_target_lookups(df_full: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Construct distributed lookup projections for B2 and B3 historical targets.

    Under Option C:
      - B2 (Diurnal): Forecast origin t needs target Y_{t-23}.
        The historical row with origin (t - 24h) has target (t - 24h) + 1h = t - 23h.
        Thus, joining where lut_window_start + 24h == current.window_start yields Y_{t-23}.
      - B3 (Weekly): Forecast origin t needs target Y_{t-167}.
        The historical row with origin (t - 168h) has target (t - 168h) + 1h = t - 167h.
        Thus, joining where lut_window_start + 168h == current.window_start yields Y_{t-167}.

    Data-Quality & Key Invariants:
      - Primary Key Uniqueness: Verifies (district, meter_id, window_start) uniqueness to prevent join fan-out.
      - Historical target is accepted ONLY IF:
        - target_is_valid == 1 (normal supply, complete interval, non-outage)
        - target_hourly_kwh is strictly finite (not NaN, not +/-inf) and >= 0.0
      Otherwise, the prediction evaluates to NULL (and is rejected from the common cohort).
    """
    validate_historical_lookup_uniqueness(df_full)

    lookup_base = df_full.select(
        F.col("district").alias("lut_district"),
        F.col("meter_id").alias("lut_meter_id"),
        F.col("window_start").alias("lut_window_start"),
        F.col("target_hourly_kwh").alias("lut_target_kwh"),
        F.col("target_is_valid").alias("lut_target_valid")
    )

    # Validated historical target expression
    valid_historical_target = F.when(
        (F.col("lut_target_valid") == 1) &
        is_valid_finite_energy(F.col("lut_target_kwh")),
        F.col("lut_target_kwh")
    ).otherwise(None)

    # B2 Projection: Offset by +24 hours
    df_lookup_b2 = lookup_base.withColumn("b2_pred", valid_historical_target).select(
        F.col("lut_district"),
        F.col("lut_meter_id"),
        (F.col("lut_window_start") + F.expr("INTERVAL 24 HOUR")).alias("b2_join_ws"),
        F.col("b2_pred")
    )

    # B3 Projection: Offset by +168 hours
    df_lookup_b3 = lookup_base.withColumn("b3_pred", valid_historical_target).select(
        F.col("lut_district"),
        F.col("lut_meter_id"),
        (F.col("lut_window_start") + F.expr("INTERVAL 168 HOUR")).alias("b3_join_ws"),
        F.col("b3_pred")
    )

    return df_lookup_b2, df_lookup_b3


def align_baseline_predictions(
    df_features: DataFrame,
    target_splits: Tuple[str, ...] = ("TRAIN", "VAL")
) -> DataFrame:
    """
    Join historical lookups with eligible forecast instances to create aligned predictions.

    Guarantees:
      - Evaluation Isolation: Target instances are filtered strictly to target_splits
        (TRAIN, VAL) prior to alignment. TEST instances are excluded from evaluation.
      - Historical Lookups: Historical lookups draw from the complete feature store
        so cross-split boundaries (e.g., early VAL consuming late TRAIN) are preserved.
      - Meter Isolation: Joins strictly on (district, meter_id).
      - No Join Fan-Out: Feature store primary key (district, meter_id, window_start) is unique.
      - Exact Temporal Alignment: B1 (lag_0h), B2 (Y_{t-23}), B3 (Y_{t-167}), B4 (rolling_mean_24h).
    """
    df_eligible = df_features.filter(
        (F.col("is_valid_forecast_instance") == 1) &
        F.col("split").isin(list(target_splits))
    )
    df_lookup_b2, df_lookup_b3 = build_historical_target_lookups(df_features)

    df_aligned = df_eligible \
        .join(
            df_lookup_b2,
            (df_eligible.district == df_lookup_b2.lut_district) &
            (df_eligible.meter_id == df_lookup_b2.lut_meter_id) &
            (df_eligible.window_start == df_lookup_b2.b2_join_ws),
            how="left"
        ) \
        .drop("lut_district", "lut_meter_id", "b2_join_ws") \
        .join(
            df_lookup_b3,
            (df_eligible.district == df_lookup_b3.lut_district) &
            (df_eligible.meter_id == df_lookup_b3.lut_meter_id) &
            (df_eligible.window_start == df_lookup_b3.b3_join_ws),
            how="left"
        ) \
        .drop("lut_district", "lut_meter_id", "b3_join_ws") \
        .withColumn("b1_pred", F.col("lag_0h")) \
        .withColumn("b4_pred", F.col("rolling_mean_24h"))

    return df_aligned


def evaluate_split_baselines(df_aligned: DataFrame, split_name: str) -> Dict[str, Any]:
    """
    Evaluate B1-B4 on the strict common intersection cohort of a specific split.

    Computes:
      - Retention audit (eligible, common cohort, exclusions, missing predecessor breakdown).
      - All five authoritative metrics: RMSE, MAE, R2, WAPE, NRMSE.
      - Cohort target summary statistics (mean, std, min, max, total energy).
    """
    split_df = df_aligned.filter(F.col("split") == split_name)
    n_eligible = split_df.count()

    # Match and exclusion audit counts
    b2_valid_matched = split_df.filter(F.col("b2_pred").isNotNull()).count()
    b3_valid_matched = split_df.filter(F.col("b3_pred").isNotNull()).count()

    missing_b2_only = split_df.filter(F.col("b2_pred").isNull() & F.col("b3_pred").isNotNull()).count()
    missing_b3_only = split_df.filter(F.col("b2_pred").isNotNull() & F.col("b3_pred").isNull()).count()
    missing_both = split_df.filter(F.col("b2_pred").isNull() & F.col("b3_pred").isNull()).count()

    # Strict Common Intersection Filter:
    # Target and all four baselines must be valid, finite (not NaN, not +/-inf), and non-negative.
    common_cohort_df = split_df.filter(
        is_valid_finite_energy(F.col("target_hourly_kwh")) &
        is_valid_finite_energy(F.col("b1_pred")) &
        is_valid_finite_energy(F.col("b2_pred")) &
        is_valid_finite_energy(F.col("b3_pred")) &
        is_valid_finite_energy(F.col("b4_pred"))
    )

    n_common = common_cohort_df.count()
    n_excluded = n_eligible - n_common
    retention_pct = (n_common / n_eligible * 100.0) if n_eligible > 0 else 0.0
    exclusion_pct = (n_excluded / n_eligible * 100.0) if n_eligible > 0 else 0.0

    cohort_audit = {
        "eligible_instances": n_eligible,
        "common_cohort_evaluated": n_common,
        "excluded_instances": n_excluded,
        "retention_pct": round(retention_pct, 4),
        "exclusion_pct": round(exclusion_pct, 4),
        "b2_valid_matches": b2_valid_matched,
        "b2_valid_match_pct": round((b2_valid_matched / n_eligible * 100.0), 4) if n_eligible > 0 else 0.0,
        "b3_valid_matches": b3_valid_matched,
        "b3_valid_match_pct": round((b3_valid_matched / n_eligible * 100.0), 4) if n_eligible > 0 else 0.0,
        "exclusion_breakdown": {
            "missing_b2_only": missing_b2_only,
            "missing_b2_only_pct": round((missing_b2_only / n_eligible * 100.0), 4) if n_eligible > 0 else 0.0,
            "missing_b3_only": missing_b3_only,
            "missing_b3_only_pct": round((missing_b3_only / n_eligible * 100.0), 4) if n_eligible > 0 else 0.0,
            "missing_both_b2_and_b3": missing_both,
            "missing_both_pct": round((missing_both / n_eligible * 100.0), 4) if n_eligible > 0 else 0.0
        }
    }

    if n_common == 0:
        return {
            "split": split_name,
            "cohort_audit": cohort_audit,
            "target_summary": {
                "count": 0, "sum_kwh": 0.0, "mean_kwh": 0.0, "std_kwh": 0.0, "min_kwh": 0.0, "max_kwh": 0.0
            },
            "baselines": {}
        }

    # Distributed metrics aggregation pass
    agg_exprs = [
        F.count(F.lit(1)).alias("N"),
        F.sum("target_hourly_kwh").alias("sum_y"),
        F.sum(F.pow(F.col("target_hourly_kwh"), 2)).alias("sum_y2"),
        F.min("target_hourly_kwh").alias("min_y"),
        F.max("target_hourly_kwh").alias("max_y"),
    ]

    baselines = [
        ("b1_pred", "B1_naive_persistence"),
        ("b2_pred", "B2_diurnal_seasonal"),
        ("b3_pred", "B3_weekly_seasonal"),
        ("b4_pred", "B4_rolling_24h_mean")
    ]

    for col_name, _ in baselines:
        agg_exprs.append(F.sum(F.pow(F.col("target_hourly_kwh") - F.col(col_name), 2)).alias(f"{col_name}_sse"))
        agg_exprs.append(F.sum(F.abs(F.col("target_hourly_kwh") - F.col(col_name))).alias(f"{col_name}_sae"))

    res = common_cohort_df.groupBy().agg(*agg_exprs).collect()[0]

    N = res["N"]
    sum_y = float(res["sum_y"])
    sum_y2 = float(res["sum_y2"])
    min_y = float(res["min_y"])
    max_y = float(res["max_y"])
    mean_y = sum_y / N
    ss_tot = sum_y2 - (sum_y ** 2) / N
    std_y = math.sqrt(ss_tot / (N - 1)) if N > 1 else 0.0

    baseline_metrics = {}
    for col_name, b_id in baselines:
        sse = float(res[f"{col_name}_sse"])
        sae = float(res[f"{col_name}_sae"])

        rmse = math.sqrt(sse / N)
        mae = sae / N
        r2 = 1.0 - (sse / ss_tot) if ss_tot > 0 else float("nan")
        wape = (sae / sum_y) * 100.0 if sum_y > 0 else float("nan")
        nrmse = (rmse / mean_y) * 100.0 if mean_y > 0 else float("nan")

        baseline_metrics[b_id] = {
            "evaluated_rows": N,
            "rmse_kwh": round(rmse, 6),
            "mae_kwh": round(mae, 6),
            "r2": round(r2, 6),
            "wape_pct": round(wape, 4),
            "nrmse_pct": round(nrmse, 4),
            "sse": round(sse, 6),
            "sae": round(sae, 6)
        }

    return {
        "split": split_name,
        "cohort_audit": {
            "eligible_instances": n_eligible,
            "common_cohort_evaluated": n_common,
            "excluded_instances": n_excluded,
            "retention_pct": round(retention_pct, 4),
            "exclusion_pct": round(exclusion_pct, 4),
            "b2_valid_matches": b2_valid_matched,
            "b2_valid_match_pct": round((b2_valid_matched / n_eligible * 100.0), 4),
            "b3_valid_matches": b3_valid_matched,
            "b3_valid_match_pct": round((b3_valid_matched / n_eligible * 100.0), 4),
            "exclusion_breakdown": {
                "missing_b2_only": missing_b2_only,
                "missing_b2_only_pct": round((missing_b2_only / n_eligible * 100.0), 4),
                "missing_b3_only": missing_b3_only,
                "missing_b3_only_pct": round((missing_b3_only / n_eligible * 100.0), 4),
                "missing_both_b2_and_b3": missing_both,
                "missing_both_pct": round((missing_both / n_eligible * 100.0), 4)
            }
        },
        "target_summary": {
            "count": N,
            "sum_kwh": round(sum_y, 4),
            "mean_kwh": round(mean_y, 6),
            "std_kwh": round(std_y, 6),
            "min_kwh": round(min_y, 6),
            "max_kwh": round(max_y, 6)
        },
        "baselines": baseline_metrics
    }


def run_baseline_evaluation(
    feature_store_path: str = "data/processed/features_forecasting",
    output_json_path: str = "reports/baselines/baseline_metrics.json",
    spark: SparkSession = None
) -> Dict[str, Any]:
    """
    Execute full baseline evaluation across TRAIN and VAL splits.
    Saves metrics JSON and returns execution summary dictionary.

    Determinism Guarantee:
      The 'splits' metrics payload is mathematically deterministic given the
      frozen feature store data. The 'run_diagnostics' section captures run-specific
      wall-clock time and runtime diagnostics.
    """
    start_time = time.time()
    created_spark = False
    if spark is None:
        spark = build_spark_session()
        created_spark = True

    try:
        print(f"[Phase 6.2] Loading Feature Store from {feature_store_path}...")
        df_features = spark.read.parquet(feature_store_path)
        total_rows = df_features.count()
        print(f"[Phase 6.2] Total Feature Store Rows: {total_rows:,}")

        print("[Phase 6.2] Aligning baselines B1-B4 for TRAIN and VAL splits...")
        df_aligned = align_baseline_predictions(df_features, target_splits=("TRAIN", "VAL"))

        print("[Phase 6.2] Evaluating TRAIN split baselines...")
        train_results = evaluate_split_baselines(df_aligned, "TRAIN")

        print("[Phase 6.2] Evaluating VAL split baselines...")
        val_results = evaluate_split_baselines(df_aligned, "VAL")

        total_runtime_sec = time.time() - start_time

        output_data = {
            "metadata": {
                "system": "System A (Next-Hour Electricity Demand Forecasting)",
                "milestone": "Phase 6.2 Baseline Evaluation",
                "methodology": "Option C Path C1 (Strict Common Intersection)",
                "feature_store_path": os.path.abspath(feature_store_path),
                "spark_version": spark.version,
                "test_split_policy": "TEST metrics were not evaluated and TEST was not used for model selection (held out exclusively for Phase 6.6)",
                "determinism_note": "Metrics under 'splits' are deterministic given the frozen dataset; 'run_diagnostics' captures run-specific environment execution details."
            },
            "run_diagnostics": {
                "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "total_runtime_seconds": round(total_runtime_sec, 2)
            },
            "splits": {
                "TRAIN": train_results,
                "VAL": val_results
            }
        }

        # Persist JSON output
        out_dir = os.path.dirname(os.path.abspath(output_json_path))
        os.makedirs(out_dir, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"[Phase 6.2] Metrics and diagnostics saved to: {output_json_path}")

        return output_data

    finally:
        if created_spark and spark is not None:
            spark.stop()


def print_formatted_results(results: Dict[str, Any]) -> None:
    """Print high-visibility terminal report summarizing results."""
    print("\n" + "=" * 90)
    print("BDA-ML SYSTEM A -- PHASE 6.2 DOMAIN BASELINE EVALUATION REPORT")
    print("Methodology: Option C, Path C1 (Strict Common Intersection)")
    print("=" * 90)

    for sp in ["TRAIN", "VAL"]:
        sp_data = results["splits"][sp]
        audit = sp_data["cohort_audit"]
        target = sp_data["target_summary"]
        baselines = sp_data["baselines"]

        print(f"\n--- Split: {sp} ---")
        print(f"Eligible Instances : {audit['eligible_instances']:,}")
        print(f"Common Cohort (N)  : {audit['common_cohort_evaluated']:,} ({audit['retention_pct']}%)")
        print(f"Excluded Instances : {audit['excluded_instances']:,} ({audit['exclusion_pct']}%)")
        print(f"  |-- Missing B2 Only      : {audit['exclusion_breakdown']['missing_b2_only']:,} ({audit['exclusion_breakdown']['missing_b2_only_pct']}%)")
        print(f"  |-- Missing B3 Only      : {audit['exclusion_breakdown']['missing_b3_only']:,} ({audit['exclusion_breakdown']['missing_b3_only_pct']}%)")
        print(f"  |-- Missing Both (B2 & B3): {audit['exclusion_breakdown']['missing_both_b2_and_b3']:,} ({audit['exclusion_breakdown']['missing_both_pct']}%)")
        print(f"Target Statistics  : Mean={target['mean_kwh']:.4f} kWh, Std={target['std_kwh']:.4f} kWh, Max={target['max_kwh']:.4f} kWh")
        print("-" * 90)
        print(f"{'Baseline ID & Name':<28} | {'RMSE (kWh)':<10} | {'MAE (kWh)':<10} | {'R2':<8} | {'WAPE (%)':<9} | {'NRMSE (%)':<9}")
        print("-" * 90)

        for b_id, metrics in baselines.items():
            print(f"{b_id:<28} | {metrics['rmse_kwh']:<10.4f} | {metrics['mae_kwh']:<10.4f} | {metrics['r2']:<8.4f} | {metrics['wape_pct']:<9.2f} | {metrics['nrmse_pct']:<9.2f}")
        print("-" * 90)

    print(f"\nTotal Runtime: {results['run_diagnostics']['total_runtime_seconds']} seconds")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate System A Baselines (Phase 6.2)")
    parser.add_argument(
        "--features",
        default="data/processed/features_forecasting",
        help="Path to feature store parquet directory"
    )
    parser.add_argument(
        "--output-json",
        default="reports/baselines/baseline_metrics.json",
        help="Path to output JSON file"
    )
    args = parser.parse_args()

    results = run_baseline_evaluation(
        feature_store_path=args.features,
        output_json_path=args.output_json
    )
    print_formatted_results(results)
