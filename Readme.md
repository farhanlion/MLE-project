# 📂 Project Structure
```
📁 project_root/
├── 📁 data/
│   ├── 📄 members_v3.csv
│   ├── 📄 train.csv
│   ├── 📄 train_v2.csv
│   ├── 📄 transactions.csv
│   ├── 📄 transactions_v2.csv
│   ├── 📄 user_logs.csv
│   └── 📄 user_logs_v2.csv
│
├── 📁 datamart/
│   └── 📁 bronze/ 
│       └── 📁 user_logs/ (zipped file from google drive)
│           ├── 📁 year=2015/
│           ├── 📁 year=2016/
│           └── 📁 year=2017/
│
├── 📁 utils/
│
├── 📄 .gitignore
├── 📄 bronze_user_logs.ipynb
├── 📄 data_exploration.ipynb
├── 📄 docker-compose.yaml
├── 📄 Dockerfile
├── 📄 Readme.md
└── 📄 requirements.txt
```

