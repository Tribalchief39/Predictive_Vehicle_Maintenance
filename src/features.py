"""
features.py - Feature engineering for telematics time-series

Generates:
 - rolling means / stds per vehicle
 - rate of change (diff)
 - deviation from vehicle baseline
 - interaction features
 - lag features

Important: Features are computed per vehicle_id sorted by timestamp to avoid leakage across vehicles.
"""
import pandas as pd
import numpy as np

ROLLING_WINDOW = 5
SENSOR_COLS = [
    "engine_temp_c", "engine_rpm", "oil_pressure_psi", "coolant_temp_c",
    "fuel_level_percent", "engine_load_percent", "throttle_pos_percent",
    "vibration_level", "brake_temp_c", "brake_pad_wear_mm",
    "battery_voltage_v", "battery_health_percent", "vehicle_speed_kph"
]

def add_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["vehicle_id", "timestamp"])
    for col in SENSOR_COLS:
        if col not in df.columns:
            continue
        # rolling mean per vehicle
        df[f"{col}_roll_mean_{window}"] = df.groupby("vehicle_id")[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{col}_roll_std_{window}"] = df.groupby("vehicle_id")[col].transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
        # rate of change
        df[f"{col}_diff"] = df.groupby("vehicle_id")[col].diff().fillna(0)
        # deviation from vehicle median baseline
        median_baseline = df.groupby("vehicle_id")[col].transform("median")
        df[f"{col}_dev_baseline"] = df[col] - median_baseline
    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "engine_load_percent" in df.columns and "engine_temp_c" in df.columns:
        df["engine_stress"] = df["engine_load_percent"] * df["engine_temp_c"] / 100
    if "vibration_level" in df.columns and "engine_rpm" in df.columns:
        df["vibration_x_rpm"] = df["vibration_level"] * df["engine_rpm"] / 1000
    if "battery_voltage_v" in df.columns and "battery_health_percent" in df.columns:
        df["battery_power_index"] = df["battery_voltage_v"] * df["battery_health_percent"] / 100
    if "brake_pad_wear_mm" in df.columns and "brake_temp_c" in df.columns:
        df["brake_wear_temp"] = df["brake_pad_wear_mm"] * df["brake_temp_c"] / 100
    if "coolant_temp_c" in df.columns and "engine_temp_c" in df.columns:
        df["temp_diff"] = df["engine_temp_c"] - df["coolant_temp_c"]
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    # hours since last reading per vehicle
    df["hours_since_last"] = df.groupby("vehicle_id")["timestamp"].diff().dt.total_seconds().fillna(0) / 3600
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rolling_features(df)
    df = add_interaction_features(df)
    df = add_time_features(df)
    # fill any new NaNs
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df

if __name__ == "__main__":
    from preprocessing import prepare_datasets
    from pathlib import Path
    csv = Path(__file__).parent.parent / "data" / "synthetic_telemetry_data.csv"
    df, feats = prepare_datasets(csv)
    df = engineer_features(df)
    print(f"After engineering: {df.shape}")
    print([c for c in df.columns if "roll" in c][:5])
