# Section 1: Directory Tree 📂
```
MLE-project/
│
├── data/                                  # Raw CSV input files from KKBox competition
│   ├── members_v3.csv                     # User demographic data
│   ├── transactions.csv                   # Subscription transactions until Feb 2017
│   ├── transactions_v2.csv                # Extended transaction data until Mar 2017
│   ├── user_logs.csv                      # User activity logs until Feb 2017
│   └── user_logs_v2.csv                   # Extended user activity until Mar 2017
│
├── datamart/                              # Multi-layer data warehouse for end-to-end processing
│   ├── bronze/                            # Raw ingested data, stored in Parquet format
│   │   ├── members/                       # Directly loaded member demographic data
│   │   ├── transactions/                  # Directly loaded transaction data (partitioned by year: 2015–2017)
│   │   └── user_logs/                     # Directly loaded user activity logs (partitioned by year: 2015–2017)
│   │       ├── year=2015/
│   │       ├── year=2016/
│   │       └── year=2017/
│   │
│   ├── silver/                            # Business logic–validated & feature-ready data
│   │   ├── members/                       
│   │   ├── transactions/                  
│   │   ├── user_logs/                     
│   │   ├── max_expiry_transactions/       # Latest transaction of each user, based on farthest expiry date
│   │   └── latest_transactions/           # # Latest transaction of each user, based on latest transaction date
│   │
│   └── gold/                              # Feature Store and Label Store
│       ├── feature_store/                 # Aggregated features per (msno, snapshot_date)   
│       └── label_store/                   # Labels (0 = churn, 1 = non-churn) per (msno, snapshot_date)
│
├── airflow/                               # Apache Airflow setup for automated data pipeline
│   ├── Dockerfile
│   ├── airflow-webserver.pid
│   ├── airflow.cfg
│   ├── airflow.db
│   ├── requirements.txt
│   └── webserver_config.py
│
├── dags/                                  # Airflow DAGs orchestrating data & ML pipelines
│   ├── daily_inference_monitoring_dag.py
│   ├── data_pipeline_dag.py
│   └── scheduled_training_dag.py
│
├── documentations/                        # Project documentation and visual diagrams
│   ├── [MLE] Project Proposal_Group1.pdf
│   ├── final_pipeline.png
│   └── silver_member docu.rtf
│
├── mlflow/                                # MLflow tracking and experiment management
│   └── .env
│
├── notebooks/                             # Jupyter notebooks
│   ├── 01_eda/                            # Exploratory Data Analysis
│   │   ├── data_exploration.ipynb
│   │   └── Detailed EDA on 4 columns.ipynb
│   │
│   ├── 02_bronze_processing/              # Raw ingestion into Bronze layer
│   │   ├── bronze_members.ipynb
│   │   ├── bronze_transactions.ipynb
│   │   └── bronze_user_logs.ipynb
│   │
│   ├── 03_silver_processing/              # Cleaning and validation into Silver layer
│   │   ├── silver_latest_transactions.ipynb
│   │   ├── silver_max_expirydate.ipynb
│   │   ├── silver_members.ipynb
│   │   ├── transactions_bronzetosilver.ipynb
│   │   └── userlogs_bronzetosilver.ipynb
│   │
│   ├── 04_gold_processing/                # Feature and label engineering into Gold layer
│   │   ├── gold_feature_store2.ipynb
│   │   ├── gold_inference_feature_store.ipynb
│   │   ├── gold_label.ipynb
│   │   └── old_notebooks/
│   │       ├── Generating feature store for daily snapshots (draft).ipynb
│   │       ├── gold_feature_store.ipynb
│   │       ├── gold_label_store.ipynb
│   │       ├── label_store_2_(one_snapshot_date).ipynb
│   │       ├── old_gold_feature_store.ipynb
│   │       └── test_gold_label.ipynb
│   │
│   ├── 05_model_training/                 # Model training and MLflow logging
│   │   ├── Model Training.ipynb
│   │   ├── fake_mlflow_model_training.ipynb
│   │   ├── mlflow_model_training_v2.ipynb
│   │   ├── xgb_churn_model.pkl
│   │   ├── mlflow_artifacts/
│   │   │   ├── X_test_head.parquet
│   │   │   ├── classification_report.txt
│   │   │   ├── feature_importance.json
│   │   │   └── xgb_bundle.pkl
│   │   ├── mlruns/                          
│   │   │   ├── 0/
│   │   │   │   └── meta.yaml                
│   │   │   └── 769348050294425984/          
│   │   │       └── 1f07ece2986b488f871d1fe14b3868b3/
│   │   │           ├── params/              
│   │   │           │   ├── class_weight
│   │   │           │   ├── cv_folds
│   │   │           │   ├── model_type
│   │   │           │   ├── n_iter
│   │   │           │   └── tuning_method
│   │   │           ├── tags/                
│   │   │           │   ├── mlflow.runName
│   │   │           │   ├── mlflow.source.name
│   │   │           │   ├── mlflow.source.type
│   │   │           │   └── mlflow.user
│   │   │           ├── meta.yaml
│   │   │           └── meta.yaml
│   │   ├── model_bank/
│   │   │   ├── credit_model_2017_03_01.pkl
│   │   │   └── credit_model_label22017_03_01.pkl
│   │   └── models/
│   │       ├── lr_churn_model_latest.pkl
│   │       ├── lr_churn_model_20251103_122821.pkl
│   │       └── xgb_model_20251103_122821.pkl
│   │
│   ├── 06_model_inferencing/               # Model inference 
│   │   ├── model_inference.ipynb
│   │   └── model_inference_mlflow.ipynb
│   │
│   ├── Dockerfile
│   └── requirements.txt
│
├── scripts/                                # Automated PySpark + MLflow scripts
│   ├── 01_bronze_members.py
│   ├── 01_bronze_transactions.py
│   ├── 01_bronze_userlogs.py
│   ├── 02_latest_transactions_bronzetosilver.py
│   ├── 02_silver_latest_transactions.py
│   ├── 02_silver_max_expirydate.py
│   ├── 02_silver_members.py
│   ├── 02_silver_transactions.py
│   ├── 02_silver_userlogs.py
│   ├── 03_gold_feature_processing.py
│   ├── 03_gold_label_processing.py
│   ├── 04_model_training_LR.py
│   ├── 04_model_training_RF.py
│   ├── 04_model_training_XGB.py
│   ├── 05_model_inference_mlflow.py
│   ├── 06_model_monitoring.py
│   ├── helloForAirflow.py
│   ├── second_hello_to_airflow.py
│   ├── MLFlow Inference Assumptions.md
│   └── old_versions/
│       ├── 02_max_expiry_latest_txn_bronzetosilver.py
│       ├── 02_members_bronzetosilver.py
│       ├── 02_transactions_bronzetosilver.py
│       ├── 03_gold_feature_processing_v2.py
│       ├── 03_gold_label_processing_v2.py
│       ├── 03_model_inference_mlflow.py
│       └── 03_model_training_v2.py            
│
├── utils/                                 
│   └── model_preprocessor.py                                         
│
├── .DS_Store                              
├── .gitignore                             # Files and folders excluded from Git tracking
├── .jupyter_ystore.db                     # JupyterLab session state
├── Readme.md                              # Project folder & file documentation (this file)
├── docker-compose.yaml                    # Container orchestration for Airflow + MLflow
├── main.py                                # Entry point for testing integrated pipeline
└── mlflow_test.ipynb                      # Notebook for validating MLflow integration
```
# Section 2: How to Run 
## 1️⃣ Start Environment

Make sure you have Docker + Docker Compose installed.
Build and start all services (Airflow, MLflow, JupyterLab):
`docker-compose up --build`
Once started:
	•	Airflow Web UI: http://localhost:8080
	•	MLflow Tracking UI: http://localhost:5000
	•	JupyterLab: http://localhost:8888

## 2️⃣ Run Data Pipeline

### Option A – via Airflow (Recommended)
Airflow DAGs are located in /dags:
	•	data_pipeline_dag.py – ETL from Bronze → Silver → Gold
	•	scheduled_training_dag.py – Automated model training
	•	daily_inference_monitoring_dag.py – Daily inference and monitoring

Steps:
	1.	Open Airflow UI (http://localhost:8080)
	2.	Trigger the DAG manually or let it run on schedule

### Option B – via Python Scripts
Run specific stages manually:
`# Bronze Layer
python scripts/01_bronze_members.py
python scripts/01_bronze_transactions.py
python scripts/01_bronze_userlogs.py

# Silver Layer
python scripts/02_silver_members.py
python scripts/02_silver_transactions.py
python scripts/02_silver_userlogs.py
python scripts/02_silver_latest_transactions.py
python scripts/02_silver_max_expirydate.py

# Gold Layer
python scripts/03_gold_feature_processing.py
python scripts/03_gold_label_processing.py

# Model Training & Inference
python scripts/04_model_training_LR.py
python scripts/04_model_training_RF.py
python scripts/04_model_training_XGB.py
python scripts/05_model_inference_mlflow.py`
