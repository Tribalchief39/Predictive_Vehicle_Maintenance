"""run_all.py - One-click training for viva demo"""
from pathlib import Path
import subprocess, sys

steps = [
    [sys.executable, "src/train_classifier.py"],
    [sys.executable, "src/train_rul.py"],
    [sys.executable, "notebooks/01_EDA.py"],
]
for cmd in steps:
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)
print("\n✅ All steps done. Now run: streamlit run app.py")
