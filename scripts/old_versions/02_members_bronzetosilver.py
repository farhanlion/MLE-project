from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder \
    .appName("silver_members") \
    .getOrCreate()

# Read bronze members data
members = (
    spark.read
    .option("header", True)
    .parquet("/app/datamart/bronze/members/")
)

# Transform columns
members = (
    members
    .withColumn("registration_date", F.to_date(F.col("registration_init_time").cast("string"), "yyyyMMdd"))
    .withColumn("year", F.year("registration_date"))
    .withColumn("month", F.format_string("%02d", F.month("registration_date")))
    .withColumn("city", F.col("city").cast("string"))
    .withColumn("registered_via", F.col("registered_via").cast("string"))
    .drop("registration_init_time")
)

# Write to silver layer
members.write.mode("overwrite").parquet("/app/datamart/silver/members/")

# Stop Spark session
spark.stop()