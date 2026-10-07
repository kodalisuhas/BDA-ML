"""
Phase 5.2: System A Supervised Demand Forecasting Feature Pipeline
Authoritative Specifications:
  - docs/FORECASTING_FEATURE_STORE_CONTRACT.md v1.0.0
  - docs/CANONICAL_DATA_CONTRACT.md v1.1
  - docs/DATA_QUALITY_POLICY.md v1.1

Input:
  data/processed/canonical_hourly.parquet (Gold Layer, 1,074,001 records)

Output:
  data/processed/features_forecasting/ (Feature Store)
  Partitioned by: split, district
  Exact 39-column schema conforming strictly to contract
"""
import os
import sys
import time
import math

# Ensure Windows Hadoop binaries are active
hadoop_home = r"C:\Users\kodal\hadoop"
os.environ["HADOOP_HOME"] = hadoop_home
os.environ["PATH"] = os.path.join(hadoop_home, "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType
import pyspark.sql.functions as F

CONTRACT_COLUMNS = [
    "district",
    "meter_id",
    "window_start",
    "target_window_start",
    "split",
    "district_idx",
    "lag_0h",
    "lag_1h",
    "lag_2h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_max_24h",
    "rolling_min_24h",
    "target_hour",
    "target_dow",
    "target_month",
    "target_is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "mean_voltage_t",
    "mean_current_t",
    "mean_freq_t",
    "outage_ratio_t",
    "standby_ratio_t",
    "brownout_ratio_t",
    "spike_ratio_t",
    "rolling_outage_ratio_24h",
    "target_hourly_kwh",
    "target_is_complete",
    "target_is_outage",
    "target_is_valid",
    "has_complete_history",
    "is_valid_forecast_instance",
]


def create_spark_session() -> SparkSession:
    """Initialize high-performance local SparkSession with deterministic timezone."""
    spark = (
        SparkSession.builder.appName("BDA-ML-Feature-Pipeline")
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.session.timeZone", "Asia/Kolkata")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_gold(spark: SparkSession, gold_path: str):
    """Read Gold layer Parquet, failing fast if absent."""
    if not os.path.exists(gold_path):
        raise FileNotFoundError(
            f"CRITICAL: Missing Gold layer input at {gold_path}. "
            "Pipeline cannot proceed without validated canonical hourly data."
        )
    return spark.read.parquet(gold_path)


def validate_gold_input(gold_df):
    """Validate Gold layer count and primary key invariants before feature creation."""
    gold_count = gold_df.count()
    expected_count = 1074001
    if gold_count != expected_count:
        raise ValueError(
            f"Gold row count mismatch! Expected {expected_count:,}, found {gold_count:,}."
        )
    print(f"  Verified Gold input records: {gold_count:,}")
    return gold_count


def build_meter_windows():
    """Establish strictly isolated meter partition windows for lags and rolling stats."""
    w_meter = Window.partitionBy("district", "meter_id").orderBy("window_start")
    w_rolling_24 = (
        Window.partitionBy("district", "meter_id")
        .orderBy("window_start")
        .rowsBetween(-23, 0)
    )
    return w_meter, w_rolling_24


def create_lag_features(df, w_meter):
    """Group A: Compute autoregressive consumption lags Y(t), Y(t-1), Y(t-2), Y(t-24), Y(t-168)."""
    return (
        df.withColumn("lag_0h", F.col("hourly_kwh"))
        .withColumn("lag_1h", F.lag("hourly_kwh", 1).over(w_meter))
        .withColumn("lag_2h", F.lag("hourly_kwh", 2).over(w_meter))
        .withColumn("lag_24h", F.lag("hourly_kwh", 24).over(w_meter))
        .withColumn("lag_168h", F.lag("hourly_kwh", 168).over(w_meter))
    )


def create_rolling_features(df, w_rolling_24):
    """Group B: Compute backward-looking 24-hour demand statistics [t-23, t]."""
    return (
        df.withColumn(
            "rolling_mean_24h", F.round(F.mean("hourly_kwh").over(w_rolling_24), 6)
        )
        .withColumn(
            "rolling_std_24h", F.round(F.stddev("hourly_kwh").over(w_rolling_24), 6)
        )
        .withColumn(
            "rolling_max_24h", F.round(F.max("hourly_kwh").over(w_rolling_24), 6)
        )
        .withColumn(
            "rolling_min_24h", F.round(F.min("hourly_kwh").over(w_rolling_24), 6)
        )
    )


def create_calendar_features(df):
    """Group C: Derive calendar and cyclical Fourier sine/cosine features for target window t+1."""
    two_pi = 2.0 * math.pi
    df = df.withColumn(
        "target_window_start", F.col("window_start") + F.expr("INTERVAL 1 HOUR")
    )
    df = df.withColumn("target_hour", F.hour("target_window_start"))
    df = df.withColumn("target_dow", F.dayofweek("target_window_start"))
    df = df.withColumn("target_month", F.month("target_window_start"))
    df = df.withColumn(
        "target_is_weekend",
        F.when(F.col("target_dow").isin(1, 7), F.lit(1)).otherwise(F.lit(0)),
    )

    df = df.withColumn(
        "hour_sin", F.sin(F.lit(two_pi) * F.col("target_hour") / 24.0)
    ).withColumn("hour_cos", F.cos(F.lit(two_pi) * F.col("target_hour") / 24.0))

    df = df.withColumn(
        "dow_sin", F.sin(F.lit(two_pi) * (F.col("target_dow") - 1) / 7.0)
    ).withColumn(
        "dow_cos", F.cos(F.lit(two_pi) * (F.col("target_dow") - 1) / 7.0)
    )

    df = df.withColumn(
        "month_sin", F.sin(F.lit(two_pi) * (F.col("target_month") - 1) / 12.0)
    ).withColumn(
        "month_cos", F.cos(F.lit(two_pi) * (F.col("target_month") - 1) / 12.0)
    )
    return df


def create_power_quality_features(df, w_rolling_24):
    """Group D & E: Attach cutoff-time (t) power-quality metrics and spatial encodings."""
    df = (
        df.withColumn("mean_voltage_t", F.col("mean_voltage"))
        .withColumn("mean_current_t", F.col("mean_current"))
        .withColumn("mean_freq_t", F.col("mean_freq"))
        .withColumn("outage_ratio_t", F.col("outage_ratio"))
        .withColumn("standby_ratio_t", F.col("standby_ratio"))
        .withColumn("brownout_ratio_t", F.col("brownout_ratio"))
        .withColumn("spike_ratio_t", F.col("spike_ratio"))
        .withColumn(
            "rolling_outage_ratio_24h",
            F.round(F.mean("outage_ratio").over(w_rolling_24), 4),
        )
    )

    # Spatial numerical index (Bareilly = 0.0, Mathura = 1.0)
    df = df.withColumn(
        "district_idx",
        F.when(F.col("district") == "Bareilly", F.lit(0.0))
        .otherwise(F.lit(1.0))
        .cast(DoubleType()),
    )
    return df


def attach_target(df, w_meter):
    """Group F: Look 1 row forward within meter to attach supervised target Y(t+1) and quality flags."""
    next_ws = F.lead("window_start", 1).over(w_meter)
    next_kwh = F.lead("hourly_kwh", 1).over(w_meter)
    next_complete = F.lead("is_complete_hour", 1).over(w_meter)
    next_outage = F.lead("is_outage_hour", 1).over(w_meter)
    next_valid = F.lead("is_valid_forecast_target", 1).over(w_meter)

    # Invariant: Target must strictly be the immediate subsequent hour [t+1, t+2)
    is_consecutive_lead = next_ws == F.col("target_window_start")

    df = (
        df.withColumn(
            "target_hourly_kwh",
            F.when(is_consecutive_lead, next_kwh).otherwise(F.lit(None).cast(DoubleType())),
        )
        .withColumn(
            "target_is_complete",
            F.when(is_consecutive_lead, next_complete).otherwise(F.lit(None).cast(IntegerType())),
        )
        .withColumn(
            "target_is_outage",
            F.when(is_consecutive_lead, next_outage).otherwise(F.lit(None).cast(IntegerType())),
        )
        .withColumn(
            "target_is_valid",
            F.when(is_consecutive_lead, next_valid).otherwise(F.lit(0)).cast(IntegerType()),
        )
    )
    return df


def calculate_eligibility(df):
    """Group F: Calculate cold-start history completeness and canonical supervised modeling eligibility."""
    # Complete history requires all historical lags and rolling stats to be non-null
    df = df.withColumn(
        "has_complete_history",
        F.when(
            F.col("lag_0h").isNotNull()
            & F.col("lag_1h").isNotNull()
            & F.col("lag_2h").isNotNull()
            & F.col("lag_24h").isNotNull()
            & F.col("lag_168h").isNotNull()
            & F.col("rolling_mean_24h").isNotNull()
            & F.col("rolling_std_24h").isNotNull(),
            F.lit(1),
        )
        .otherwise(F.lit(0))
        .cast(IntegerType()),
    )

    # Supervised instance is valid only when target is valid normal-supply, target energy is non-null, and full lag history exists
    df = df.withColumn(
        "is_valid_forecast_instance",
        F.when(
            (F.col("has_complete_history") == 1)
            & (F.col("target_is_valid") == 1)
            & F.col("target_hourly_kwh").isNotNull(),
            F.lit(1),
        )
        .otherwise(F.lit(0))
        .cast(IntegerType()),
    )
    return df


def assign_temporal_split(df):
    """Assign chronological Train, Validation, and Test splits based on target_window_start."""
    return df.withColumn(
        "split",
        F.when(
            F.col("target_window_start") < "2020-09-01 00:00:00",
            F.lit("TRAIN"),
        )
        .when(
            F.col("target_window_start") < "2021-01-01 00:00:00", F.lit("VAL")
        )
        .otherwise(F.lit("TEST")),
    )


def select_contract_schema(df):
    """Enforce exact 39-column contract schema and column ordering."""
    return df.select(*CONTRACT_COLUMNS)


def validate_pre_write_invariants(df, gold_count):
    """Run pre-write invariant checks on the assembled feature store."""
    print("\n  Validating pre-write feature store invariants...")
    feature_count = df.count()
    if feature_count != gold_count:
        raise ValueError(
            f"Row count invariant violated: Expected {gold_count:,}, got {feature_count:,}!"
        )

    # Check column count
    col_count = len(df.columns)
    if col_count != 39:
        raise ValueError(
            f"Schema column count violated: Expected 39, got {col_count}!"
        )

    # Check split distribution
    split_counts = df.groupBy("split").count().collect()
    print("  Split Partition Counts:")
    for row in split_counts:
        print(f"    - {row['split']}: {row['count']:,} rows")

    # Check eligibility counts
    eligible_count = df.filter(
        F.col("is_valid_forecast_instance") == 1
    ).count()
    complete_hist_count = df.filter(
        F.col("has_complete_history") == 1
    ).count()
    print(
        f"  Complete Lag History Rows: {complete_hist_count:,} ({complete_hist_count/feature_count*100:.2f}%)"
    )
    print(
        f"  Eligible Supervised Modeling Rows (Target Valid & Complete History): {eligible_count:,} ({eligible_count/feature_count*100:.2f}%)"
    )

    # Verify zero nulls in eligible training subset
    eligible_df = df.filter(F.col("is_valid_forecast_instance") == 1)
    null_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in CONTRACT_COLUMNS
    ]
    null_row = eligible_df.agg(*null_exprs).collect()[0].asDict()
    total_nulls_in_eligible = sum(null_row.values())
    if total_nulls_in_eligible > 0:
        bad_cols = {k: v for k, v in null_row.items() if v > 0}
        raise ValueError(
            f"Anti-null invariant violated in eligible modeling subset! Nulls found: {bad_cols}"
        )
    print("  Eligible Modeling Subset Null Audit: Exactly 0 nulls detected.")


def build_feature_store():
    """Main execution entry point for Phase 5.2."""
    print("=" * 90)
    print(
        "PHASE 5.2: SYSTEM A SUPERVISED DEMAND FORECASTING FEATURE PIPELINE"
    )
    print("=" * 90)

    start_total = time.time()
    spark = create_spark_session()

    gold_path = os.path.abspath("data/processed/canonical_hourly.parquet")
    output_path = os.path.abspath("data/processed/features_forecasting")

    # 1. Read Gold Layer
    print(f"\n[1/7] Reading Gold Layer from: {gold_path}...")
    gold_df = read_gold(spark, gold_path)

    # 2. Validate Input Invariants
    print("\n[2/7] Validating Gold Input Invariants...")
    gold_count = validate_gold_input(gold_df)

    # 3. Establish Meter Windows & Temporal Lags
    print(
        "\n[3/7] Building Meter Windows, Lag Features, and Rolling Demand Statistics..."
    )
    w_meter, w_rolling_24 = build_meter_windows()
    df_lags = create_lag_features(gold_df, w_meter)
    df_rolling = create_rolling_features(df_lags, w_rolling_24)

    # 4. Calendar Features & Power Quality Context
    print(
        "\n[4/7] Generating Calendar Fourier Vector and Power-Quality Context Features..."
    )
    df_calendar = create_calendar_features(df_rolling)
    df_pq = create_power_quality_features(df_calendar, w_rolling_24)

    # 5. Target Attachment & Eligibility Logic
    print(
        "\n[5/7] Attaching Target Y(t+1), Cold-Start Flags, and Eligibility Logic..."
    )
    df_target = attach_target(df_pq, w_meter)
    df_eligible = calculate_eligibility(df_target)

    # 6. Temporal Split Assignment & Schema Enforcement
    print(
        "\n[6/7] Assigning Chronological Splits and Enforcing 39-Column Contract Schema..."
    )
    df_split = assign_temporal_split(df_eligible)
    df_final = select_contract_schema(df_split)

    # Pre-write validation
    validate_pre_write_invariants(df_final, gold_count)

    # 7. Materialize Feature Store to Parquet
    print(
        f"\n[7/7] Materializing Feature Store to: {output_path} (Partitioned by split, district)..."
    )
    df_final.write.mode("overwrite").partitionBy("split", "district").parquet(
        output_path
    )

    elapsed = time.time() - start_total
    print("\n" + "=" * 90)
    print(
        f"PHASE 5.2 COMPLETE! Successfully persisted {gold_count:,} records in {elapsed:.2f}s"
    )
    print("=" * 90)

    spark.stop()


if __name__ == "__main__":
    build_feature_store()
