# 📂 Project Structure
```
📁 kkbox-churn-prediction/
│
├── 📄 README.md                       # Project overview and setup guide
├── 🚫 .gitignore                      # Git ignore (data/, models/, logs/)
├── 📦 requirements.txt                # Python dependencies
├── ⚙️  config.yaml                     # Configuration settings
│
├── 🐳 Dockerfile                      # Docker container setup
├── 🐳 docker-compose.yaml             # Docker orchestration
│
├── 📁 data/                           # Raw data from Kaggle (gitignored)
│   ├── 📄 members_v3.csv              # User demographic data
│   ├── 📄 train.csv                   # Training labels v1
│   ├── 📄 train_v2.csv                # Training labels v2 (main)
│   ├── 📄 transactions.csv            # Transaction records 2015-2016
│   ├── 📄 transactions_v2.csv         # Transaction records 2017
│   ├── 📄 user_logs.csv               # User listening logs 2015-2017
│   └── 📄 user_logs_v2.csv            # User listening logs March 2017
│
├── 📁 datamart/                       # Processed data (gitignored)
│   │
│   ├── 🥉 bronze/                     # Raw ingested data (parquet)
│   │   ├── 📁 transactions/           # Partitioned by year/month
│   │   │   ├── 📁 year=2015/
│   │   │   │   ├── 📁 month=01/
│   │   │   │   ├── 📁 month=02/
│   │   │   │   └── ...
│   │   │   ├── 📁 year=2016/
│   │   │   └── 📁 year=2017/
│   │   │
│   │   ├── 📁 members/                # Not partitioned (spans 14 years)
│   │   │   └── 📊 *.parquet
│   │   │
│   │   ├── 📁 user_logs/              # Partitioned by year/month
│   │   │   ├── 📁 year=2015/
│   │   │   │   ├── 📁 month=01/
│   │   │   │   └── ...
│   │   │   ├── 📁 year=2016/
│   │   │   └── 📁 year=2017/
│   │   │
│   │   └── 📁 user_logs_v2/           # March 2017 only
│   │       └── 📁 year=2017/
│   │           └── 📁 month=03/
│   │
│   ├── 🥈 silver/                     # Cleaned and joined data
│   │   ├── 📁 users_cleaned/          # Cleaned member data
│   │   ├── 📁 transactions_agg/       # Aggregated transactions
│   │   ├── 📁 listening_agg/          # Aggregated listening behavior
│   │   └── 📁 master_dataset/         # All data joined
│   │
│   └── 🥇 gold/                       # ML-ready feature datasets
│       ├── 📁 training/
│       │   ├── 📊 train.parquet
│       │   ├── 📊 val.parquet
│       │   └── 📊 test.parquet
│       ├── 📁 inference/
│       │   └── 📊 latest.parquet
│       └── 📋 feature_metadata.json   # Feature descriptions
│
├── 📓 notebooks/                      # Jupyter notebooks
│   │
│   ├── 📁 00_existing/                # Your original notebooks
│   │   ├── 📓 bronze_user_logs.ipynb
│   │   ├── 📓 bronze_transactions_members.ipynb
│   │   └── 📓 data_exploration.ipynb
│   │
│   ├── 📁 01_eda/                     # Exploratory Data Analysis
│   │   ├── 📓 01_members_analysis.ipynb
│   │   ├── 📓 02_transactions_analysis.ipynb
│   │   ├── 📓 03_user_logs_analysis.ipynb
│   │   └── 📓 04_churn_patterns.ipynb
│   │
│   ├── 📁 02_feature_engineering/     # Feature experiments
│   │   ├── 📓 01_user_features.ipynb
│   │   ├── 📓 02_transaction_features.ipynb
│   │   ├── 📓 03_listening_features.ipynb
│   │   └── 📓 04_feature_validation.ipynb
│   │
│   └── 📁 03_modeling/                # Model experiments
│       ├── 📓 01_baseline_models.ipynb
│       ├── 📓 02_advanced_models.ipynb
│       ├── 📓 03_model_tuning.ipynb
│       └── 📓 04_final_evaluation.ipynb
│
├── 🔧 utils/                          # Utility functions
│   ├── 🐍 __init__.py
│   └── 🐍 bronze_processing.py        # Bronze layer utilities
│
├── 💻 src/                            # Core source code
│   ├── 🐍 __init__.py
│   ├── 🐍 data_processing.py          # Data transformation logic
│   ├── 🐍 features.py                 # Feature engineering
│   └── 🐍 models.py                   # ML training/evaluation
│
├── ⚡ scripts/                        # Executable pipeline scripts
│   ├── 🐍 process_bronze.py           # CSV → Bronze (parquet)
│   ├── 🐍 process_silver.py           # Bronze → Silver (cleaned)
│   ├── 🐍 process_gold.py             # Silver → Gold (features)
│   ├── 🐍 train_model.py              # Train and save model
│   └── 🐍 run_full_pipeline.py        # Execute complete pipeline
│
├── 🤖 models/                         # Saved models (gitignored)
│   ├── 📁 experiments/                # Experimental models
│   │   ├── 📁 exp_001/
│   │   │   ├── 💾 model.pkl
│   │   │   ├── 📊 metrics.json
│   │   │   └── ⚙️  config.yaml
│   │   └── 📁 exp_002/
│   │
│   └── 📁 production/                 # Production model
│       └── 📁 latest/
│           ├── 💾 model.pkl
│           ├── 📋 feature_names.json
│           └── 📋 metadata.json
│
├── 📝 logs/                           # Log files (gitignored)
│   ├── 📄 bronze_processing.log
│   ├── 📄 silver_processing.log
│   ├── 📄 gold_processing.log
│   └── 📄 training.log
│
└── 📊 reports/                        # Generated reports (optional)
    ├── 📁 data_quality/
    │   └── 📋 bronze_summary.json
    └── 📁 model_performance/
        └── 📄 evaluation_report.html
```
📋 Icon Legend	
    
| Icon | Meaning              |
|:----:|-----------------------|
| 📁   | Directory/Folder      |
| 📄   | Text/Document file    |
| 📓   | Jupyter Notebook      |
| 🐍   | Python file           |
| 📊   | Data file (parquet/chart) |
| 📋   | JSON/Config file      |
| 💾   | Model/Binary file     |
| ⚙️   | Configuration         |
| ⚡   | Executable script     |
| 🔧   | Utilities             |
| 💻   | Source code           |
| 🤖   | ML Models             |
| 📝   | Logs                  |
| 🚫   | Git ignore            |
| 📦   | Dependencies          |
| 🐳   | Docker                |
| 🥉   | Bronze layer          |
| 🥈   | Silver layer          |
| 🥇   | Gold layer            |


