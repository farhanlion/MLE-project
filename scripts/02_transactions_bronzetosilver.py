from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder \
    .appName("silver_transactions") \
    .getOrCreate()

# Read transactions data from bronze
transactions = (
    spark.read
    .option("header", True)
    .parquet("/app/datamart/bronze/transactions/")
)

# Transform and clean columns
transactions = (
    transactions
    .withColumn("transaction_date", F.to_date(F.col("transaction_date").cast("string"), "yyyyMMdd"))
    .withColumn("membership_expire_date", F.to_date(F.col("membership_expire_date").cast("string"), "yyyyMMdd"))
    .withColumn("is_auto_renew", F.col("is_auto_renew").cast("int"))
    .withColumn("is_cancel", F.col("is_cancel").cast("int"))
    .withColumn("payment_method_id", F.col("payment_method_id").cast("string"))
    .withColumn("payment_plan_days", F.col("payment_plan_days").cast("int"))
    .withColumn("plan_list_price", F.col("plan_list_price").cast("int"))
    .withColumn("actual_amount_paid", F.col("actual_amount_paid").cast("int"))
    .dropna(subset=["msno", "transaction_date", "membership_expire_date"])
)

# Write clean transactions to silver
transactions.write.mode("overwrite").parquet("/app/datamart/silver/transactions/")

# Stop Spark session
spark.stop()