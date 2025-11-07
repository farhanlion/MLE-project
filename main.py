import subprocess

# Define the scripts in the order you want them to run
scripts = [
    "/app/scripts/01_bronze_members.py",
    "/app/scripts/01_bronze_transactions.py",
    "/app/scripts/01_bronze_userlogs.py",
    "/app/scripts/02_silver_max_expirydate.py",
    "/app/scripts/02_silver_members.py",
    "/app/scripts/02_silver_transactions.py",
    "/app/scripts/02_silver_latest_transactions.py",
    "/app/scripts/02_silver_userlogs.py",
    "/app/scripts/03_gold_feature_processing.py --mode full",
    "/app/scripts/03_gold_label_processing.py --mode full"
]

def run_script(script_path):
    print(f"\n🚀 Running: {script_path}")
    result = subprocess.run(f"python {script_path}", shell=True)
    if result.returncode == 0:
        print(f"✅ Completed: {script_path}")
    else:
        print(f"❌ Failed: {script_path}")

if __name__ == "__main__":
    for script in scripts:
        run_script(script)
