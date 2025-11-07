""""
Call this script: python main_training_pipeline.py YYYY-MM-DD
e.g. python main_training_pipeline.py 2016-04-02
"""


import subprocess
import sys

scripts = [
    "/app/scripts/04_model_training_LR.py",
    "/app/scripts/04_model_training_RF.py",
    "/app/scripts/04_model_training_XGB.py",
]

def run_script(script_path, train_date):
    print(f"\n🚀 Running: {script_path}")
    result = subprocess.run(
        f"python {script_path} --train_date {train_date}",
        shell=True
    )
    if result.returncode == 0:
        print(f"✅ Completed: {script_path}")
    else:
        print(f"❌ Failed: {script_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <train_date>")
        sys.exit(1)
    
    train_date = sys.argv[1]
    
    for script in scripts:
        run_script(script, train_date)