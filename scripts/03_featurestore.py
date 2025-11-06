from pyspark.sql import SparkSession, functions as F, Window

# Initialize Spark
spark = SparkSession.builder.appName("gold_feature_store_full").getOrCreate()

# Snapshot date
SNAPSHOT_DATE = "2017-03-01"

# ---------------------------------------------------------------------
# 1. Load Silver tables
# ---------------------------------------------------------------------
members = spark.read.parquet("/app/datamart/silver/members/")
transactions = spark.read.parquet("/app/datamart/silver/transactions/")
userlogs = spark.read.parquet("/app/datamart/silver/user_logs/")
max_expiry_txn = spark.read.parquet("/app/datamart/silver/max_expiry_transactions/")

# ---------------------------------------------------------------------
# 2. Registered users up to snapshot
# ---------------------------------------------------------------------
snapshot_users = (
    members.filter(F.col("registration_date") <= F.lit(SNAPSHOT_DATE))
    .withColumn("snapshot_date", F.lit(SNAPSHOT_DATE))
    .select("msno", "registration_date", "registered_via", "city", "snapshot_date")
)

# ---------------------------------------------------------------------
# 3. Filter user logs: 30-day & 7-day windows
# ---------------------------------------------------------------------
ref_date = F.to_date(F.lit(SNAPSHOT_DATE))
userlogs = userlogs.withColumn("date", F.to_date("date"))

logs_30d = userlogs.filter(
    (F.col("date") <= ref_date) & (F.col("date") >= F.date_sub(ref_date, 29))
)
logs_7d = userlogs.filter(
    (F.col("date") <= ref_date) & (F.col("date") >= F.date_sub(ref_date, 6))
)

# ---------------------------------------------------------------------
# 4. User log features
# ---------------------------------------------------------------------
agg_30d = (
    logs_30d.groupBy("msno")
    .agg(
        F.sum("total_secs").alias("sum_secs_w30"),
        F.countDistinct("date").alias("active_days_w30"),
        (F.sum("num_100") / F.sum("num_unq")).alias("complete_rate_w30"),
    )
)

agg_7d = logs_7d.groupBy("msno").agg(F.sum("total_secs").alias("sum_secs_w7"))

features = (
    agg_30d.join(agg_7d, "msno", "left")
    .withColumn(
        "engagement_ratio_7_30",
        F.when(F.col("sum_secs_w30") > 0, F.col("sum_secs_w7") / F.col("sum_secs_w30")).otherwise(0),
    )
)

# ---------------------------------------------------------------------
# 5. Days since last play
# ---------------------------------------------------------------------
last_play = (
    userlogs.filter(F.col("date") <= ref_date)
    .groupBy("msno")
    .agg(F.max("date").alias("last_play_date"))
    .withColumn("days_since_last_play", F.datediff(F.lit(SNAPSHOT_DATE), F.col("last_play_date")))
)

features = features.join(last_play, "msno", "left")

# ---------------------------------------------------------------------
# 6. Trend in total_secs over 30d using slope ≈ cov(day_idx, daily_secs)/var(day_idx)
# ---------------------------------------------------------------------
daily_secs_30d = (
    logs_30d.groupBy("msno", "date").agg(F.sum("total_secs").alias("daily_secs"))
)

w = Window.partitionBy("msno").orderBy("date")
trend_30d = (
    daily_secs_30d.withColumn("day_idx", F.row_number().over(w))
    .groupBy("msno")
    .agg((F.covar_pop("day_idx", "daily_secs") / F.var_pop("day_idx")).alias("trend_secs_w30"))
)

features = features.join(trend_30d, "msno", "left")

# ---------------------------------------------------------------------
# 7. Transaction features (AS OF snapshot)
# ---------------------------------------------------------------------
transactions = transactions.withColumn("transaction_date", F.to_date("transaction_date"))

t = transactions.alias("t")
r = snapshot_users.select("msno", "snapshot_date", "registration_date").alias("r")

# Transactions up to snapshot date
tx_asof_snap = (
    t.join(r, F.col("t.msno") == F.col("r.msno"), "inner")
    .where(F.col("t.transaction_date") <= F.col("r.snapshot_date"))
)

# Latest transaction per msno, snapshot
w_latest = Window.partitionBy("r.msno").orderBy(F.col("t.transaction_date").desc())
latest_tx = (
    tx_asof_snap.withColumn("rn", F.row_number().over(w_latest))
    .where(F.col("rn") == 1)
    .select(
        F.col("r.msno").alias("msno"),
        F.col("r.snapshot_date").alias("snapshot_date"),
        F.col("t.transaction_date").alias("latest_transaction_date"),
        F.col("t.is_auto_renew").alias("last_is_auto_renew"),
        F.col("t.plan_list_price").alias("last_plan_list_price"),
    )
)

# Tenure days
tenure_asof = latest_tx.join(
    r.select("msno", "snapshot_date", "registration_date"),
    ["msno", "snapshot_date"],
    "left",
).withColumn("tenure_days", F.datediff(F.col("latest_transaction_date"), F.col("registration_date")))

# Auto-renew stats
auto_renew_stats = (
    tx_asof_snap.groupBy(F.col("r.msno").alias("msno"), F.col("r.snapshot_date").alias("snapshot_date"))
    .agg(
        F.sum(F.when(F.col("t.is_auto_renew") == 1, 1).otherwise(0)).alias("auto_renew_count"),
        F.count("*").alias("total_tx_before_expire"),
    )
    .withColumn(
        "auto_renew_share",
        F.col("auto_renew_count") / F.when(F.col("total_tx_before_expire") > 0, F.col("total_tx_before_expire")).otherwise(1),
    )
    .select("msno", "snapshot_date", "auto_renew_share")
)

# ---------------------------------------------------------------------
# 8. Merge all features (avoid duplicate columns)
# ---------------------------------------------------------------------
fill_map = {
    "sum_secs_w30": 0.0,
    "active_days_w30": 0,
    "complete_rate_w30": 0.0,
    "sum_secs_w7": 0.0,
    "engagement_ratio_7_30": 0.0,
    "days_since_last_play": 0.0,
    "trend_secs_w30": 0.0,
    "tenure_days": 0.0,
    "auto_renew_share": 0.0,
}

# Drop duplicate columns before joining
snapshot_users = snapshot_users.drop("city", "registered_via")
features = features.drop("snapshot_date")
latest_tx = latest_tx.drop("snapshot_date")
tenure_asof = tenure_asof.drop("snapshot_date")
auto_renew_stats = auto_renew_stats.drop("snapshot_date")

final = (
    snapshot_users
    .join(members.select("msno", "registered_via", "city"), "msno", "left")
    .join(features, "msno", "left")
    .join(latest_tx, "msno", "left")
    .join(tenure_asof.select("msno", "tenure_days"), "msno", "left")
    .join(auto_renew_stats, "msno", "left")
    .na.fill(fill_map)
)

# Write to Gold Feature Store
output_path = "/app/datamart/gold/feature_store/2017_03_01/"
final.write.mode("overwrite").parquet(output_path)

spark.stop()