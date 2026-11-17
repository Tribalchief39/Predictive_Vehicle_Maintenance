"""
train_classifier.py - Fault/Maintenance Classification
Tasks:
 - Binary: failure_imminent (0/1)
 - Multi-class: fault type diagnosis
Uses Group split by vehicle_id to avoid leakage.
Saves models to models/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from preprocessing import prepare_datasets, TARGET_CLASS, TARGET_FAULT
from features import engineer_features

DATA_PATH = Path(__file__).parent.parent / "data" / "synthetic_telemetry_data.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def get_X_y(df, feature_cols):
    X = df[feature_cols].copy()
    y = df[TARGET_CLASS].copy()
    return X, y

def train_evaluate(verbose=True):
    df, base_feats = prepare_datasets(DATA_PATH if DATA_PATH.exists() else Path(r"C:\Users\ansht\Desktop\AIML\PROJECT\synthetic_telemetry_data.csv"))
    df = engineer_features(df)

    # extended feature list includes engineered features
    exclude = {"failure_imminent", "rul_days", "fault_type_label", "timestamp", "failure_date",
               "vehicle_id", "brand", "failure_type", "gps_latitude", "gps_longitude",
               "engine_failure_imminent","brake_issue_imminent","battery_issue_imminent"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    X = df[feature_cols]
    y = df[TARGET_CLASS]
    groups = df["vehicle_id"]

    # Vehicle-grouped split: 80% vehicles train, 20% test
    # pandas 3 returns an Arrow-backed string array here; sklearn can only
    # index NumPy/pandas objects, so convert before splitting.
    unique_vehicles = np.asarray(groups.unique(), dtype=object)
    train_veh, test_veh = train_test_split(unique_vehicles, test_size=0.2, random_state=42)
    train_mask = groups.isin(train_veh)
    test_mask = groups.isin(test_veh)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if verbose:
        print(f"Vehicles: total {len(unique_vehicles)} | train {len(train_veh)} | test {len(test_veh)}")
        print(f"Rows: train {X_train.shape[0]} | test {X_test.shape[0]}")
        print(f"Features: {len(feature_cols)}")
        print(f"Class balance train:\n{y_train.value_counts(normalize=True)}")
        print(f"Class balance test:\n{y_test.value_counts(normalize=True)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Keep scaler and feature list for inference
    joblib.dump(scaler, MODEL_DIR / "scaler_cls.joblib")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols_cls.joblib")

    results = {}

    # --- Logistic Regression Baseline ---
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    y_proba_lr = lr.predict_proba(X_test_s)[:,1]
    results["LogisticRegression"] = {
        "acc": accuracy_score(y_test, y_pred_lr),
        "roc_auc": roc_auc_score(y_test, y_proba_lr) if y_test.nunique()>1 else 0.5,
        "report": classification_report(y_test, y_pred_lr, zero_division=0)
    }
    joblib.dump(lr, MODEL_DIR / "classifier_lr.joblib")
    if verbose:
        print("\n=== Logistic Regression ===")
        print(results["LogisticRegression"]["report"])
        print(f"ROC-AUC: {results['LogisticRegression']['roc_auc']:.4f}")

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:,1]
    results["RandomForest"] = {
        "acc": accuracy_score(y_test, y_pred_rf),
        "roc_auc": roc_auc_score(y_test, y_proba_rf) if y_test.nunique()>1 else 0.5,
        "report": classification_report(y_test, y_pred_rf, zero_division=0)
    }
    joblib.dump(rf, MODEL_DIR / "classifier_rf.joblib")
    if verbose:
        print("\n=== Random Forest ===")
        print(results["RandomForest"]["report"])
        print(f"ROC-AUC: {results['RandomForest']['roc_auc']:.4f}")
        # feature importance
        imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
        print("Top 10 features:\n", imp)

    # --- XGBoost (if available) ---
    if HAS_XGB:
        # scale_pos_weight for imbalance
        neg, pos = np.bincount(y_train)
        scale = neg / max(pos, 1)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
        xgb_clf.fit(X_train, y_train)
        y_pred_x = xgb_clf.predict(X_test)
        y_proba_x = xgb_clf.predict_proba(X_test)[:,1]
        results["XGBoost"] = {
            "acc": accuracy_score(y_test, y_pred_x),
            "roc_auc": roc_auc_score(y_test, y_proba_x) if y_test.nunique()>1 else 0.5,
            "report": classification_report(y_test, y_pred_x, zero_division=0)
        }
        joblib.dump(xgb_clf, MODEL_DIR / "classifier_xgb.joblib")
        # also save as main classifier (best)
        joblib.dump(xgb_clf, MODEL_DIR / "classifier.joblib")
        if verbose:
            print("\n=== XGBoost ===")
            print(results["XGBoost"]["report"])
            print(f"ROC-AUC: {results['XGBoost']['roc_auc']:.4f}")
    else:
        joblib.dump(rf, MODEL_DIR / "classifier.joblib")
        print("XGBoost not installed, using RF as main classifier")

    # Save test set for dashboard demo
    df_test = df[test_mask].copy()
    df_test.to_csv(MODEL_DIR / "test_sample.csv", index=False)

    # Save metrics summary
    summary = pd.DataFrame({k: {"accuracy": v["acc"], "roc_auc": v["roc_auc"]} for k,v in results.items()}).T
    summary.to_csv(MODEL_DIR / "classification_metrics.csv")
    print("\nSaved models to", MODEL_DIR)
    print(summary)
    return results, feature_cols

if __name__ == "__main__":
    train_evaluate()
