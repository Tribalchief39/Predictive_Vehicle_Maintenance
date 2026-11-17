"""
preprocessing.py - Data loading, cleaning, leakage prevention, target creation
for Predictive Vehicle Maintenance (OBD-II / Telematics)
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Columns that MUST NOT be used as features (leakage)
LEAKAGE_COLS = ["failure_date", "failure_type", "engine_failure_imminent",
                "brake_issue_imminent", "battery_issue_imminent"]

# Failure imminent columns in dataset -> we combine into one target
IMMINENT_COLS = ["engine_failure_imminent", "brake_issue_imminent", "battery_issue_imminent"]

TARGET_CLASS = "failure_imminent"
TARGET_RUL = "rul_days"
TARGET_FAULT = "fault_type_label"

def load_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["failure_date"] = pd.to_datetime(df["failure_date"], errors="coerce")
    df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)
    return df

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. Binary classification target: any imminent flag == 1 OR failure_type != No Failure
    if all(c in df.columns for c in IMMINENT_COLS):
        df[TARGET_CLASS] = (df[IMMINENT_COLS].sum(axis=1) > 0).astype(int)
        # Also mark actual failure rows as 1 (for safety)
        df.loc[df["failure_type"] != "No Failure", TARGET_CLASS] = 1
    else:
        df[TARGET_CLASS] = (df["failure_type"] != "No Failure").astype(int)

    # 2. Fault type label for diagnosis (multi-class)
    # Map failure_type to simpler categories, keep No Failure as 0
    df[TARGET_FAULT] = df["failure_type"]

    # 3. RUL regression target in DAYS (and hours) : failure_date - timestamp
    # For "No Failure" rows failure_date is 2050-01-01 -> RUL will be huge (~10k days)
    # We keep it but downstream training filters to meaningful RUL window
    df[TARGET_RUL] = (df["failure_date"] - df["timestamp"]).dt.total_seconds() / (3600*24)
    # Clip negative (if timestamp after failure_date due to data noise)
    df[TARGET_RUL] = df[TARGET_RUL].clip(lower=0)

    # For rows where RUL > 365*5, cap or set NaN for regression? Keep but mark
    # For presentation we keep raw value, model training will filter.

    # Additional time feature
    df["days_since_start"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / (3600*24)
    return df

def get_feature_columns(df: pd.DataFrame, target: str = TARGET_CLASS) -> list[str]:
    exclude = set(LEAKAGE_COLS + [TARGET_CLASS, TARGET_RUL, TARGET_FAULT,
                                  "timestamp", "failure_date", "vehicle_id", "brand",
                                  "gps_latitude", "gps_longitude", "failure_type",
                                  "days_since_start"])
    # But we WILL use vehicle_id/brand as encoded features later via one-hot or grouping,
    # for base model we exclude vehicle_id to force vehicle-agnostic learning
    # Numeric sensor columns
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric if c not in exclude]
    # Also add encoded brand if exists
    return feats

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleaning: fill NaNs, fix impossible values
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    # Categorical
    for c in ["brand", "failure_type"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")
    # Drop duplicates
    df = df.drop_duplicates()
    return df

def prepare_datasets(csv_path: str | Path):
    df = load_data(csv_path)
    df = create_targets(df)
    df = clean_dataframe(df)
    feature_cols = get_feature_columns(df)
    return df, feature_cols

if __name__ == "__main__":
    csv = Path(__file__).parent.parent / "data" / "synthetic_telemetry_data.csv"
    # fallback to old location
    if not csv.exists():
        csv = Path(r"C:\Users\ansht\Desktop\AIML\PROJECT\synthetic_telemetry_data.csv")
    df, feats = prepare_datasets(csv)
    print(f"Shape: {df.shape}")
    print(f"Features ({len(feats)}): {feats[:10]}...")
    print(df[TARGET_CLASS].value_counts())
    print(df[TARGET_RUL].describe())
    print(df["failure_type"].value_counts().head(10))
