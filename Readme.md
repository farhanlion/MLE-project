# Section 1: Directory Tree 📂
```
MLE-project/
│
├── data/                       # Raw KKBox CSVs
│   ├── members_v3.csv
│   ├── transactions*.csv
│   └── user_logs*.csv
│
├── datamart/                   # Medallion data warehouse
│   ├── bronze/ (raw parquet, partitioned by year)
│   ├── silver/ (cleaned + validated)
│   └── gold/   (feature_store/, label_store/)
│
├── airflow/                    # Airflow setup + config
├── dags/                       # DAGs: data pipeline, training, monitoring
│
├── docs/                       # Project docs + diagrams
│
├── notebooks/                  # Dev notebooks
│   ├── 01_eda/
│   ├── 02_bronze_processing/
│   ├── 03_silver_processing/
│   ├── 04_gold_processing/
│   ├── 05_model_training/
│   └── 06_model_inferencing/
│
├── scripts/                    # PySpark + ML training scripts
│   ├── 01_bronze_*.py
│   ├── 02_silver_*.py
│   ├── 03_gold_*.py
│   ├── 04_model_training_*.py
│   └── 05_model_inference_*.py
│
├── mlflow/                     # MLflow tracking/experiments
│
├── utils/
│   └── model_preprocessor.py
│
├── docker-compose.yaml         # Airflow + MLflow orchestration
│
├── 01_generate_medallion_tables.py     # Bronze/Silver/Gold pipeline (ETL)
├── 02_main_training_pipeline.py        # Training + MLflow registration
├── 03_inference_and_monitoring.py      # Batch/online inference + monitoring
│
└── README.md

```
# Section 2: How to Run 
## 1️⃣ Start Environment

Make sure you have Docker + Docker Compose installed.  
Build and start all services (Airflow, MLflow, JupyterLab):
```bash
docker-compose up --build
```
Once started:  
| Service                | URL                                            |
| ---------------------- | ---------------------------------------------- |
| **Airflow Web UI**     | [http://localhost:8080](http://localhost:8080) |
| **MLflow Tracking UI** | [http://localhost:5000](http://localhost:5000) |
| **JupyterLab**         | [http://localhost:8888](http://localhost:8888) |


## 2️⃣ Run Data Pipeline

### Option A – via Airflow (Recommended)
Airflow DAGs are located in /dags:  
| DAG                                 | Purpose                                   |
| ----------------------------------- | ----------------------------------------- |
| `data_pipeline_dag.py`              | ETL pipeline (Bronze → Silver → Gold)     |
| `scheduled_training_dag.py`         | Scheduled model training & MLflow logging |
| `daily_inference_monitoring_dag.py` | Daily inference + model monitoring        |
  

Steps:  
	1.	Open Airflow UI (http://localhost:8080)  
	2.	Trigger the DAG manually or let it run on schedule  

### Option B – via Python Scripts
Run specific stages manually:

Or run each script: 


```bash
python 01_generate_medallion_tables.py
```
Creates Bronze → Silver → Gold tables (full ETL)
```bash
python 02_main_training_pipeline.py 2016-04-02
```
Trains the model for a chosen date and logs results to MLflow.
```bash
python 03_inference_and_monitoring.py
``` 
Runs inference and performs drift + performance monitoring.
For simplicity and ease of debugging, we have hardcoded some of the dates and some of the dates are in variables. Which ofcourse will not be the case in a real deployment :)


