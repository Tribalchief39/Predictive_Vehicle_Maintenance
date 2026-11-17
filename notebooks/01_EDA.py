"""
01_EDA.py - Can be run as notebook or script
Exploratory analysis for viva: do `python notebooks/01_EDA.py` to generate plots in reports/
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))
from preprocessing import prepare_datasets

DATA = Path(__file__).parent.parent / "data" / "synthetic_telemetry_data.csv"
OUT = Path(__file__).parent.parent / "reports"
OUT.mkdir(exist_ok=True, parents=True)

df, feats = prepare_datasets(DATA if DATA.exists() else Path(r"C:\Users\ansht\Desktop\AIML\PROJECT\data\synthetic_telemetry_data.csv"))
print(df["failure_type"].value_counts())
print(df["failure_imminent"].value_counts(normalize=True))

# 1. Class balance
plt.figure(figsize=(6,4))
sns.countplot(x="failure_type", data=df, order=df["failure_type"].value_counts().index[:6])
plt.xticks(rotation=20); plt.title("Failure Type Distribution (heavily imbalanced)"); plt.tight_layout()
plt.savefig(OUT / "eda_class_balance.png", dpi=150); plt.close()

# 2. Sensor hist healthy vs failure
fig, axes = plt.subplots(2,2, figsize=(12,8))
for ax, col in zip(axes.flat, ["engine_temp_c","coolant_temp_c","vibration_level","battery_health_percent"]):
    sns.histplot(data=df, x=col, hue="failure_imminent", kde=True, ax=ax, palette="Set2")
    ax.set_title(col)
plt.tight_layout(); plt.savefig(OUT / "eda_sensor_hist.png", dpi=150); plt.close()

# 3. Correlation
plt.figure(figsize=(10,8))
num = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(num, cmap="coolwarm", center=0)
plt.title("Correlation heatmap"); plt.tight_layout()
plt.savefig(OUT / "eda_corr.png", dpi=150); plt.close()

print(f"Plots saved to {OUT}")
