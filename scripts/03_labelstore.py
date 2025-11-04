from pyspark.sql import SparkSession, functions as F

# =========================================================
# Initialize Spark session
# =========================================================
spark = SparkSession.builder \
    .appName("gold_label_store") \
    .getOrCreate()

# =========================================================
# Parameters
# =========================================================
SNAPSHOT_DATE = "2017-03-01"
LOOKAHEAD_DAYS = 30

# =========================================================
# Load Gold Feature Store and Silver Max-Expiry Transactions
# =========================================================
features = spark.read.parquet("/app/datamart/gold/feature_store/2017_03_01/")
max_expiry_txn = spark.read.parquet("/app/datamart/silver/max_expiry_transactions/")

# 🔧 Ensure correct data type
max_expiry_txn = max_expiry_txn.withColumn(
    "transaction_date", F.to_date("transaction_date")
)

# =========================================================
# Label Definition
# User is retained (1) if they have a renewal (expiry extended)
# within 30 days after the snapshot date; else churned (0).
# =========================================================
cutoff_date = F.date_add(F.lit(SNAPSHOT_DATE), LOOKAHEAD_DAYS)

renewed_users = (
    max_expiry_txn
    .filter(
        (F.col("transaction_date") > F.lit(SNAPSHOT_DATE)) &
        (F.col("transaction_date") <= cutoff_date)
    )
    .select("msno")
    .distinct()
)

# =========================================================
# Assign churn label per user
# =========================================================
label_store = (
    features
    .join(renewed_users.withColumn("label", F.lit(1)), on="msno", how="left")
    .withColumn("label", F.when(F.col("label").isNull(), 0).otherwise(1))
)

# =========================================================
# Save Gold Label Store
# =========================================================
output_path = "/app/datamart/gold/label_store/2017_03_01/"
label_store.write.mode("overwrite").parquet(output_path)

# =========================================================
# Stop Spark
# =========================================================
spark.stop()