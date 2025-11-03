import argparse
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# to call this script: python model_inference.py --snapshotdate "2016-05-01" --modelname "credit_model_2024_09_01.pkl"  

def main(snapshotdate, modelname):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    
    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname
    config["model_bank_directory"] = "/app/notebooks/06_modelling/model_bank/"
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    
    pprint.pprint(config)
    

    # --- load model artefact from model bank ---
    # Load the model from the pickle file
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    
    print("Model loaded successfully! " + config["model_artefact_filepath"])


    # --- load feature store ---
    feature_location = "/app/datamart/gold/inference_feature_store"
    
    # Option 1: Check if specific partition exists and load it
    specific_feature_file = f"{feature_location}snapshot_date={config['snapshot_date_str']}/*.parquet"
    specific_feature_dir = f"{feature_location}snapshot_date={config['snapshot_date_str']}/"
    
    print(f"Looking for feature files in: {specific_feature_dir}")
    
    # Check if the specific partition directory exists and has parquet files
    parquet_files_exist = False
    features_sdf = None
    
    try:
        # Check if any parquet files exist in the specific partition
        parquet_files = glob.glob(specific_feature_file)
        if parquet_files:
            print(f"Found {len(parquet_files)} Parquet file(s) for the specific date")
            features_sdf = spark.read.parquet(specific_feature_dir)
            parquet_files_exist = True
        else:
            print(f"No Parquet files found in specific partition: {specific_feature_dir}")
    except Exception as e:
        print(f"Error checking specific partition: {e}")
    
    # Option 2: If specific partition not found, check entire directory
    if not parquet_files_exist or features_sdf is None or features_sdf.count() == 0:
        print("Trying to load from entire feature store directory...")
        try:
            # Check if any parquet files exist in the entire feature store
            all_parquet_files = glob.glob(f"{feature_location}*.parquet") or \
                               glob.glob(f"{feature_location}*/*.parquet") or \
                               glob.glob(f"{feature_location}*/*/*.parquet")
            
            if all_parquet_files:
                print(f"Found {len(all_parquet_files)} Parquet file(s) in feature store")
                # Load all data and filter by snapshot_date
                features_store_sdf = spark.read.parquet(feature_location)
                features_sdf = features_store_sdf.filter((col("snapshot_date") == config["snapshot_date_str"]))
            else:
                print("No Parquet files found in feature store directory")
                raise ValueError(f"No Parquet files found in {feature_location}")
                
        except Exception as e:
            print(f"Error loading from feature store: {e}")
            # Final fallback: check if it's a schema issue and try with explicit schema
            try:
                print("Attempting to load with explicit schema...")
                # Define a basic schema based on your expected columns
                from pyspark.sql.types import StructType, StructField
                # Add your expected schema here based on your feature store
                # Example:
                # schema = StructType([
                #     StructField("msno", StringType(), True),
                #     StructField("snapshot_date", StringType(), True),
                #     StructField("fe_1", FloatType(), True),
                #     # ... add all your feature columns
                # ])
                # features_store_sdf = spark.read.schema(schema).parquet(feature_location)
                
                # For now, try without schema but with different options
                features_store_sdf = spark.read.option("mergeSchema", "true").parquet(feature_location)
                features_sdf = features_store_sdf.filter((col("snapshot_date") == config["snapshot_date_str"]))
            except Exception as final_error:
                raise ValueError(f"Failed to load features: {final_error}")

    # Check if we finally have data
    if features_sdf is None or features_sdf.count() == 0:
        raise ValueError(f"No features found for snapshot date: {config['snapshot_date_str']}")
    
    print(f"extracted features_sdf: {features_sdf.count()} rows for snapshot_date: {config['snapshot_date_str']}")
    
    # Show schema for debugging
    print("Feature schema:")
    features_sdf.printSchema()
    
    # Show a sample of the data
    print("Sample data:")
    features_sdf.show(5)

    # Convert to Pandas for sklearn processing
    features_pdf = features_sdf.toPandas()
    print(f"Converted to Pandas DataFrame with shape: {features_pdf.shape}")

    # --- preprocess data for modeling ---
    # prepare X_inference
    feature_cols = [fe_col for fe_col in features_pdf.columns if fe_col.startswith('fe_')]
    X_inference = features_pdf[feature_cols]
    
    # apply transformer - standard scaler
    transformer_stdscaler = model_artefact["preprocessing_transformers"]["stdscaler"]
    X_inference = transformer_stdscaler.transform(X_inference)
    
    print('X_inference', X_inference.shape[0])


    # --- model prediction inference ---
    # load model
    model = model_artefact["model"]
    
    # predict model
    y_inference = model.predict_proba(X_inference)[:, 1]
    
    # prepare output
    y_inference_pdf = features_pdf[["msno","snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference
    

    # --- save model inference to datamart gold table ---
    # create bronze datalake
    gold_directory = f"datamart/gold/model_predictions/{config["model_name"][:-4]}/"
    print(gold_directory)
    
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    # save gold table - IRL connect to database to write
    partition_name = config["model_name"][:-4] + "_predictions_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

    
    # --- end spark session --- 
    spark.stop()
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
