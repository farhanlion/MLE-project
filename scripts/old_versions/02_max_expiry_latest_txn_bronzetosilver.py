from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize Spark session
spark = SparkSession.builder \
    .appName("silver_max_expirydate") \
    .getOrCreate()

# Read transactions data from bronze layer
transactions = (
    spark.read
    .option("header", True)
    .parquet("/app/datamart/bronze/transactions/")
)

# Select record with maximum membership_expire_date for each msno
window_spec = Window.partitionBy("msno").orderBy(F.col("membership_expire_date").desc())

max_expiry = (
    transactions
    .withColumn("rn", F.row_number().over(window_spec))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

# Write result to silver layer
max_expiry.write.mode("overwrite").parquet("/app/datamart/silver/max_expiry_transactions/")

# Stop Spark session
spark.stop()