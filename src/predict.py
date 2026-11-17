"""predict.py - Single-row inference for classifier + RUL"""
import pandas as pd
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

def load_models():
    clf = joblib.load(MODEL_DIR / "classifier.joblib")
    rul = joblib.load(MODEL_DIR / "rul_model.joblib")
    scaler_cls = joblib.load(MODEL_DIR / "scaler_cls.joblib") if (MODEL_DIR / "scaler_cls.joblib").exists() else None
    scaler_rul = joblib.load(MODEL_DIR / "scaler_rul.joblib") if (MODEL_DIR / "scaler_rul.joblib").exists() else None
    feats_cls = joblib.load(MODEL_DIR / "feature_cols_cls.joblib")
    feats_rul = joblib.load(MODEL_DIR / "feature_cols_rul.joblib")
    return clf, rul, scaler_cls, scaler_rul, feats_cls, feats_rul

def predict_single(row: dict):
    """row: dict of sensor values matching feature names"""
    import numpy as np
    clf, rul, scaler_cls, scaler_rul, feats_cls, feats_rul = load_models()
    # build DataFrames
    x_cls = pd.DataFrame([row])[feats_cls].fillna(0)
    x_rul = pd.DataFrame([row])[feats_rul].fillna(0)
    # classifier uses RF/XGB directly (no scaler needed for tree models)
    # but if LR is chosen, would need scaler; we handle both
    try:
        proba = clf.predict_proba(x_cls)[0,1]
        pred = int(proba > 0.5)
    except:
        pred = int(clf.predict(x_cls)[0])
        proba = float(pred)
    rul_pred = float(rul.predict(x_rul)[0])
    rul_pred = max(0, min(rul_pred, 60))
    return {"failure_imminent": pred, "failure_probability": round(float(proba),4), "rul_days": round(rul_pred,2)}

if __name__ == "__main__":
    # quick demo with test sample
    import pandas as pd
    df = pd.read_csv(MODEL_DIR / "test_sample.csv", nrows=1)
    print(df.head(1).to_dict(orient="records")[0])
    print(predict_single(df.iloc[0].to_dict()))
