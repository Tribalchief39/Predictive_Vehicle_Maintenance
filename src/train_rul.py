"""
train_rul.py - Remaining Useful Life (RUL) Regression
RUL = failure_date - timestamp (in days)
Trains on vehicles that have eventual failure OR on filtered window.
Saves regressor to models/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from preprocessing import prepare_datasets, TARGET_RUL
from features import engineer_features

DATA_PATH = Path(__file__).parent.parent / "data" / "synthetic_telemetry_data.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_rul(verbose=True):
    df, _ = prepare_datasets(DATA_PATH if DATA_PATH.exists() else Path(r"C:\Users\ansht\Desktop\AIML\PROJECT\synthetic_telemetry_data.csv"))
    df = engineer_features(df)

    # For RUL, we need meaningful targets.
    # The dataset has ~35 failure rows, but many rows belong to vehicles that eventually fail
    # within the observation window. Use RUL clipping 0-60 days for training; exclude huge 10k values.
    # Strategy: train on rows where rul_days < 365 (i.e., failure within a year) OR failure_type != No Failure
    # But to have enough data, we include all; models will learn that healthy vehicles have large RUL.
    # Better: filter to RUL <= 60 days for focused model, and also include healthy samples with capped RUL=60
    df_rul = df.copy()
    # Cap RUL at 60 days for regression stability (common in RUL literature)
    df_rul["rul_capped"] = df_rul[TARGET_RUL].clip(upper=60)

    # If we have very few failures, we still train on capped regression (demonstrates pipeline)
    # Filter: keep all rows, but we will report metrics on both overall and on failure-proximal subset
    exclude = {"failure_imminent", "rul_days", "rul_capped", "fault_type_label", "timestamp",
               "failure_date", "vehicle_id", "brand", "failure_type", "gps_latitude", "gps_longitude",
               "engine_failure_imminent","brake_issue_imminent","battery_issue_imminent",
               "days_since_start"}
    feature_cols = [c for c in df_rul.select_dtypes(include=[np.number]).columns if c not in exclude]

    X = df_rul[feature_cols]
    y = df_rul["rul_capped"]
    groups = df_rul["vehicle_id"]

    # pandas 3 returns an Arrow-backed string array here; sklearn can only
    # index NumPy/pandas objects, so convert before splitting.
    unique_veh = np.asarray(groups.unique(), dtype=object)
    train_veh, test_veh = train_test_split(unique_veh, test_size=0.2, random_state=42)
    train_mask = groups.isin(train_veh)
    test_mask = groups.isin(test_veh)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if verbose:
        print(f"RUL train rows {len(X_train)} test {len(X_test)} | features {len(feature_cols)}")
        print(f"RUL stats train: {y_train.describe().to_dict()}")
        print(f"RUL stats test: {y_test.describe().to_dict()}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, MODEL_DIR / "scaler_rul.joblib")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols_rul.joblib")

    results = {}

    # Linear Regression baseline
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    pred_lr = lr.predict(X_test_s)
    results["LinearRegression"] = {
        "mae": mean_absolute_error(y_test, pred_lr),
        "rmse": np.sqrt(mean_squared_error(y_test, pred_lr)),
        "r2": r2_score(y_test, pred_lr)
    }
    joblib.dump(lr, MODEL_DIR / "rul_lr.joblib")
    if verbose:
        print(f"\nLinearRegression: MAE={results['LinearRegression']['mae']:.3f} RMSE={results['LinearRegression']['rmse']:.3f} R2={results['LinearRegression']['r2']:.3f}")

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results["RandomForest"] = {
        "mae": mean_absolute_error(y_test, pred_rf),
        "rmse": np.sqrt(mean_squared_error(y_test, pred_rf)),
        "r2": r2_score(y_test, pred_rf)
    }
    joblib.dump(rf, MODEL_DIR / "rul_rf.joblib")
    if verbose:
        print(f"RandomForest: MAE={results['RandomForest']['mae']:.3f} RMSE={results['RandomForest']['rmse']:.3f} R2={results['RandomForest']['r2']:.3f}")
        imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
        print("Top 10 RUL features:\n", imp)

    if HAS_XGB:
        xgr = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        xgr.fit(X_train, y_train)
        pred_x = xgr.predict(X_test)
        results["XGBoost"] = {
            "mae": mean_absolute_error(y_test, pred_x),
            "rmse": np.sqrt(mean_squared_error(y_test, pred_x)),
            "r2": r2_score(y_test, pred_x)
        }
        joblib.dump(xgr, MODEL_DIR / "rul_xgb.joblib")
        joblib.dump(xgr, MODEL_DIR / "rul_model.joblib")
        if verbose:
            print(f"XGBoost: MAE={results['XGBoost']['mae']:.3f} RMSE={results['XGBoost']['rmse']:.3f} R2={results['XGBoost']['r2']:.3f}")
    else:
        joblib.dump(rf, MODEL_DIR / "rul_model.joblib")

    summary = pd.DataFrame(results).T
    summary.to_csv(MODEL_DIR / "rul_metrics.csv")
    print("\nRUL metrics saved:\n", summary)
    return results

if __name__ == "__main__":
    train_rul()
