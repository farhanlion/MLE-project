import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import to_date, col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .config("spark.driver.memory", "4g") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

def clean_bronze_table(monthdir):
    df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .parquet(monthdir))

    # Remove rows with negative total_secs
    df = df.filter(col("total_secs") >= 0)

    # Remove outliers
    cols = ["num_25","num_50","num_75","num_985","num_100","num_unq","total_secs"]
    def fences_for(colname):
        q1, q3 = df.approxQuantile(colname, [0.25, 0.75], 0.001)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return lower, upper

    condition = None
    for c in cols:
        lower, upper = fences_for(c)
        cond = (col(c) >= lower) & (col(c) <= upper)
        condition = cond if condition is None else (condition & cond)

    df_clean = df.filter(condition)

    # Change date column to date type
    df_clean = df_clean.withColumn("date", to_date(col("date").cast("string"), "yyyyMMdd"))

    return df_clean  # <-- make sure we return this



# === Process all months from bronze and write to silver ===
bronze_root = "/app/datamart/bronze/user_logs"
silver_root = "/app/datamart/silver/user_logs"

# Find month directories like: datamart/bronze/user_logs/year=YYYY/month=MM
month_dirs = sorted(glob.glob(os.path.join(bronze_root, "year=*/month=*")))

for monthdir in month_dirs:
    print(f"\n=== Processing {monthdir} ===")
    df_silver = clean_bronze_table(monthdir)

    # Extract year and month from the path
    parts = monthdir.replace("\\", "/").split("/")
    year = parts[-2].split("=")[1]
    month = parts[-1].split("=")[1]

    # Build output path and write
    outdir = os.path.join(silver_root, f"year={year}", f"month={month}")
    (df_silver.write
        .mode("overwrite")
        .parquet(outdir))
    print(f"Wrote silver data to: {outdir}")

