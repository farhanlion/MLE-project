import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col, to_date, lit

def create_gold_features(inference_date: str, spark, output_path: str = "datamart/gold/feature_store"):
    """
    Build GOLD layer features for churn/engagement modeling as of `inference_date`.

    Parameters
    ----------
    inference_date : str
        Snapshot/cutoff date in 'YYYY-MM-DD' format. Only data <= this date is used.
    spark : SparkSession
        Active Spark session.
    output_path : str
        Parquet destination for the GOLD feature store.
    """

    # -------------------------
    # 1) Load SILVER tables
    # -------------------------
    df_userlogs = (spark.read
        .option("header", True).option("inferSchema", True)
        .parquet("datamart/silver/user_logs"))

    df_transactions = (spark.read
        .option("header", True).option("inferSchema", True)
        .parquet("datamart/silver/transactions"))

    df_members = (spark.read
        .option("header", True).option("inferSchema", True)
        .parquet("datamart/silver/members"))

    # -------------------------
    # 2) Registered users as of inference_date
    # -------------------------
    registered_users = (
        df_members
        .withColumn("registration_date", to_date(col("registration_date")))
        .filter(col("registration_date") <= to_date(lit(inference_date)))
        .withColumn(
            "tenure_days_at_snapshot",
            F.datediff(to_date(lit(inference_date)), col("registration_date"))
        )
        .select(
            "msno",
            "registration_date",
            "tenure_days_at_snapshot",
            "registered_via",
            "city_clean",
            "via_oh",
            "city_oh",
        )
    )

    # -------------------------
    # 3) User logs windows (7d / 30d ending at inference_date)
    # -------------------------
    ref_today = to_date(lit(inference_date))
    lower30   = F.date_sub(ref_today, 29)   # inclusive range [today-29, today]
    lower7    = F.date_sub(ref_today, 6)    # inclusive range [today-6,  today]

    df_userlogs = df_userlogs.withColumn("date", to_date(col("date")))

    userlogs_30d = df_userlogs.filter((col("date") >= lower30) & (col("date") <= ref_today))
    userlogs_7d  = df_userlogs.filter((col("date") >= lower7)  & (col("date") <= ref_today))

    # Engagement aggregates (30d)
    user_sum_30d = (
        userlogs_30d.groupBy("msno")
        .agg(F.sum("total_secs").alias("sum_secs_w30"))
    )
    registered_users = (
        registered_users.join(user_sum_30d, on="msno", how="left")
        .na.fill({"sum_secs_w30": 0.0})
    )

    user_active_days_30d = (
        userlogs_30d.groupBy("msno")
        .agg(F.countDistinct("date").alias("active_days_w30"))
    )
    registered_users = (
        registered_users.join(user_active_days_30d, on="msno", how="left")
        .na.fill({"active_days_w30": 0})
    )

    user_complete_rate_30d = (
        userlogs_30d.groupBy("msno")
        .agg((F.sum("num_100") / F.sum("num_unq")).alias("complete_rate_w30"))
    )
    registered_users = (
        registered_users.join(user_complete_rate_30d, on="msno", how="left")
        .na.fill({"complete_rate_w30": 0.0})
    )

    # Engagement aggregates (7d) + ratio
    user_sum_7d = (
        userlogs_7d.groupBy("msno")
        .agg(F.sum("total_secs").alias("sum_secs_w7"))
    )
    registered_users = (
        registered_users.join(user_sum_7d, on="msno", how="left")
        .na.fill({"sum_secs_w7": 0.0})
        .withColumn(
            "engagement_ratio_7_30",
            col("sum_secs_w7") / F.when(col("sum_secs_w30") > 0, col("sum_secs_w30")).otherwise(F.lit(1.0))
        )
    )

    # Days since last play (as of inference_date)
    last_play = (
        df_userlogs.filter(col("date") <= ref_today)
        .groupBy("msno")
        .agg(F.max("date").alias("last_play_date"))
    )
    registered_users = (
        registered_users.join(last_play, on="msno", how="left")
        .withColumn("days_since_last_play", F.datediff(ref_today, col("last_play_date")))
    )

    # Trend (slope) of daily secs over 30d window
    daily_secs = (
        userlogs_30d.groupBy("msno", "date")
        .agg(F.sum("total_secs").alias("daily_secs"))
    )
    w_day = Window.partitionBy("msno").orderBy("date")
    trend = (
        daily_secs
        .withColumn("day_idx", F.row_number().over(w_day))
        .groupBy("msno")
        .agg((F.covar_pop("day_idx", "daily_secs") / F.var_pop("day_idx")).alias("trend_secs_w30"))
    )
    registered_users = (
        registered_users.join(trend, on="msno", how="left")
        .na.fill({"trend_secs_w30": 0.0})
    )

    # -------------------------
    # 4) Transactions as of inference_date
    # -------------------------
    df_transactions_filtered = df_transactions.filter(
        to_date(col("transaction_date")) <= to_date(lit(inference_date))
    )

    # Latest transaction date and tenure_days (registration -> latest_tx)
    latest_tx = (
        df_transactions_filtered.groupBy("msno")
        .agg(F.max("transaction_date").alias("latest_transaction_date"))
    )
    registered_users = (
        registered_users.join(latest_tx, on="msno", how="left")
        .withColumn(
            "tenure_days",
            F.datediff(col("latest_transaction_date"), col("registration_date"))
        )
        .na.fill({"tenure_days": 0})
    )

    # Auto-renew share (global up to inference_date)
    auto_renew_stats = (
        df_transactions_filtered.groupBy("msno")
        .agg(
            F.sum(F.when(col("is_auto_renew") == 1, 1).otherwise(0)).alias("auto_renew_count"),
            F.count(F.lit(1)).alias("total_tx_before_expire"),
        )
        .withColumn(
            "auto_renew_share",
            col("auto_renew_count") / F.when(col("total_tx_before_expire") > 0, col("total_tx_before_expire")).otherwise(F.lit(1.0))
        )
        .select("msno", "auto_renew_share")
    )
    registered_users = (
        registered_users.join(auto_renew_stats, on="msno", how="left")
        .na.fill({"auto_renew_share": 0.0})
    )

    # Last is_auto_renew flag
    w_latest = Window.partitionBy("msno").orderBy(col("transaction_date").desc())
    latest_tx_flag = (
        df_transactions_filtered
        .withColumn("rn", F.row_number().over(w_latest))
        .filter(col("rn") == 1)
        .select("msno", col("is_auto_renew").alias("last_is_auto_renew"))
    )
    registered_users = (
        registered_users.join(latest_tx_flag, on="msno", how="left")
        .na.fill({"last_is_auto_renew": 0})
    )

    # Ensure no nulls in numeric aggregates (defensive)
    registered_users = registered_users.na.fill({
        "sum_secs_w30": 0.0,
        "sum_secs_w7": 0.0,
        "active_days_w30": 0,
        "complete_rate_w30": 0.0,
        "engagement_ratio_7_30": 0.0,
        "days_since_last_play": 0,
        "trend_secs_w30": 0.0,
        "tenure_days": 0,
        "auto_renew_share": 0.0,
        "last_is_auto_renew": 0,
    })

    # -------------------------
    # 5) Write GOLD feature store
    # -------------------------
    dated_output_path = f"{output_path}/{inference_date}"

    (
        registered_users
        .write
        .mode("overwrite")
        .parquet(dated_output_path)
    )
    
    print(f"✅ Feature store snapshot saved to: {dated_output_path}")

    return registered_users
