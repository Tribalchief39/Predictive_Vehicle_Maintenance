"""
app.py - Streamlit Dashboard for Predictive Vehicle Maintenance
Mimics an automotive mechanic diagnostic dashboard.

Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Predictive Vehicle Maintenance", layout="wide", page_icon="🚗")

DATA_PATH = Path("data/synthetic_telemetry_data.csv")
MODEL_DIR = Path("models")

# --- Load data & models ---
@st.cache_data
def load_data():
    for p in [DATA_PATH, Path(r"C:\Users\ansht\Desktop\AIML\PROJECT\data\synthetic_telemetry_data.csv"), Path("synthetic_telemetry_data.csv")]:
        if p.exists():
            df = pd.read_csv(p, parse_dates=["timestamp", "failure_date"])
            return df
    return pd.DataFrame()

@st.cache_resource
def load_models():
    try:
        clf = joblib.load(MODEL_DIR / "classifier.joblib")
        rul = joblib.load(MODEL_DIR / "rul_model.joblib")
        feats_cls = joblib.load(MODEL_DIR / "feature_cols_cls.joblib")
        feats_rul = joblib.load(MODEL_DIR / "feature_cols_rul.joblib")
        return clf, rul, feats_cls, feats_rul
    except Exception as e:
        return None, None, None, None

df = load_data()
clf, rul_model, feats_cls, feats_rul = load_models()

# --- Header ---
st.title("🚗 AI-Based Predictive Vehicle Maintenance & Fault Diagnosis")
st.caption("Prototype using synthetic OBD-II / Telematics sensor data — predicts imminent faults & Remaining Useful Life (RUL)")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Controls")
    if df.empty:
        st.error("No data found")
        st.stop()
    vehicles = sorted(df["vehicle_id"].unique())
    selected = st.selectbox("Select Vehicle", vehicles, index=0)
    df_v = df[df["vehicle_id"] == selected].sort_values("timestamp")
    # show latest timestamp selector
    idx = st.slider("Timeline (record)", 0, len(df_v)-1, len(df_v)-1)
    row = df_v.iloc[idx]
    st.divider()
    st.subheader("Simulate Live Sensors")
    st.caption("Drag to simulate what-if sensor drift")
    sim_engine_temp = st.slider("Engine Temp (°C)", 60, 130, int(row["engine_temp_c"]))
    sim_coolant = st.slider("Coolant Temp (°C)", 60, 120, int(row["coolant_temp_c"]))
    sim_vib = st.slider("Vibration", 0.0, 5.0, float(row["vibration_level"]), step=0.1)
    sim_batt_health = st.slider("Battery Health %", 20, 100, int(row["battery_health_percent"]))
    sim_brake_wear = st.slider("Brake Pad Wear (mm)", 0.0, 12.0, float(row["brake_pad_wear_mm"]), step=0.1)

# --- Prepare engineered row for prediction ---
from src.preprocessing import create_targets
from src.features import engineer_features

# Build a mini dataframe for engineering (need history)
df_mini = df_v.copy()
# ensure days_since_start exists (training used it) - compute like preprocessing.py
if "days_since_start" not in df_mini.columns:
    # use global min from full dataset for consistency with training
    global_min = df["timestamp"].min()
    df_mini["days_since_start"] = (df_mini["timestamp"] - global_min).dt.total_seconds() / (3600*24)
# override simulated values for current row
df_mini.iloc[idx, df_mini.columns.get_loc("engine_temp_c")] = sim_engine_temp
df_mini.iloc[idx, df_mini.columns.get_loc("coolant_temp_c")] = sim_coolant
df_mini.iloc[idx, df_mini.columns.get_loc("vibration_level")] = sim_vib
df_mini.iloc[idx, df_mini.columns.get_loc("battery_health_percent")] = sim_batt_health
df_mini.iloc[idx, df_mini.columns.get_loc("brake_pad_wear_mm")] = sim_brake_wear

# engineer (adds rolling, interactions, time features)
df_mini = engineer_features(df_mini)
cur = df_mini.iloc[idx]

# Predict - robust to missing columns (reindex)
if clf is not None and feats_cls is not None:
    Xc = pd.DataFrame([cur]).reindex(columns=feats_cls, fill_value=0).fillna(0)
    try:
        proba = float(clf.predict_proba(Xc)[0,1])
    except:
        proba = float(clf.predict(Xc)[0])
        proba = proba  # binary
    pred_label = int(proba > 0.5)
    Xr = pd.DataFrame([cur]).reindex(columns=feats_rul, fill_value=0).fillna(0)
    rul_pred = float(rul_model.predict(Xr)[0])
    rul_pred = max(0, min(rul_pred, 60))
else:
    proba, pred_label, rul_pred = 0.0, 0, 60.0
    st.warning("Models not trained yet. Run `python src/train_classifier.py` and `python src/train_rul.py`")

# --- Top metrics ---
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Vehicle", selected, delta=row["brand"])
with c2:
    risk = "🔴 HIGH RISK" if proba > 0.6 else "🟡 Medium" if proba > 0.3 else "🟢 Healthy"
    st.metric("Failure Probability", f"{proba*100:.1f}%", delta=risk)
with c3:
    st.metric("Predicted RUL (days)", f"{rul_pred:.1f}", delta="capped at 60d")
with c4:
    st.metric("Odometer (km)", f"{row['odometer_reading']:.0f}", delta=f"{row['vehicle_speed_kph']:.0f} kph")

# --- Risk gauge ---
col_g1, col_g2 = st.columns([1,2])
with col_g1:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=proba*100, title={'text': "Risk Score"},
        gauge={'axis': {'range': [0,100]},
               'bar': {'color': "red" if proba>0.6 else "orange" if proba>0.3 else "green"},
               'steps': [{'range':[0,30], 'color':'#d4edda'},{'range':[30,60], 'color':'#fff3cd'},{'range':[60,100], 'color':'#f8d7da'}]}))
    fig.update_layout(height=250, margin=dict(t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)
    if pred_label==1:
        st.error(f"⚠️ **Recommendation:** Schedule maintenance within {rul_pred:.1f} days — inspect engine/battery/brakes")
    else:
        st.success("✅ No imminent failure predicted — continue monitoring")
    st.caption(f"True label: {row['failure_type']} | failure_date: {row['failure_date'].date() if pd.notna(row['failure_date']) else 'N/A'}")
    # SHAP-like explanation (fallback to feature importance)
    st.subheader("Why this prediction?")
    if feats_cls is not None:
        try:
            importances = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0]) if hasattr(clf,"coef_") else None
            if importances is not None:
                top_idx = np.argsort(importances)[-5:][::-1]
                for i in top_idx:
                    st.write(f"- **{feats_cls[i]}**: weight {importances[i]:.3f} | value {Xc.iloc[0, i]:.2f}")
        except Exception as e:
            st.write("Enable SHAP after training for detailed explanations")

with col_g2:
    # Sensor trend for vehicle
    st.subheader(f"Sensor Timeline — {selected}")
    sensors = ["engine_temp_c","coolant_temp_c","vibration_level","battery_health_percent","brake_pad_wear_mm","engine_rpm"]
    sel_sensors = st.multiselect("Sensors to plot", sensors, default=["engine_temp_c","coolant_temp_c","vibration_level"])
    fig2 = px.line(df_v, x="timestamp", y=sel_sensors, markers=True)
    fig2.update_layout(height=380, legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)

# --- Current telemetry cards ---
st.subheader("📊 Current Telemetry Snapshot")
t1, t2, t3, t4, t5, t6 = st.columns(6)
t1.metric("Engine Temp", f"{sim_engine_temp} °C", delta=f"{sim_engine_temp - 95:.1f} vs norm 95")
t2.metric("Coolant", f"{sim_coolant} °C")
t3.metric("Oil Pressure", f"{row['oil_pressure_psi']:.1f} psi")
t4.metric("Vibration", f"{sim_vib:.2f}")
t5.metric("Battery V", f"{row['battery_voltage_v']:.2f} V")
t6.metric("Battery Health", f"{sim_batt_health}%")
t1b, t2b, t3b, t4b, t5b, t6b = st.columns(6)
t1b.metric("Brake Temp", f"{row['brake_temp_c']:.0f} °C")
t2b.metric("Brake Wear", f"{sim_brake_wear:.1f} mm")
t3b.metric("Engine RPM", f"{row['engine_rpm']:.0f}")
t4b.metric("Engine Load", f"{row['engine_load_percent']:.0f}%")
t5b.metric("Speed", f"{row['vehicle_speed_kph']:.0f} kph")
t6b.metric("Fuel", f"{row['fuel_level_percent']:.0f}%")

# --- Fault Diagnosis table ---
st.subheader("🔧 Fault Diagnosis (Multi-class)")
fault_dist = df_v["failure_type"].value_counts().reset_index()
fault_dist.columns = ["failure_type", "count"]
st.dataframe(fault_dist, use_container_width=True)
st.caption("Note: Dataset is synthetic, with ~1.7% failure rows — this is a prototype. Real deployment needs OEM-validated data.")

# --- Metrics if available ---
if (MODEL_DIR / "classification_metrics.csv").exists():
    st.divider()
    st.subheader("📈 Model Performance (held-out vehicles)")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Classification**")
        st.dataframe(pd.read_csv(MODEL_DIR / "classification_metrics.csv"), use_container_width=True)
    with c2:
        if (MODEL_DIR / "rul_metrics.csv").exists():
            st.write("**RUL Regression (capped 60d)**")
            st.dataframe(pd.read_csv(MODEL_DIR / "rul_metrics.csv"), use_container_width=True)

st.markdown("---")
st.caption("Pipeline: OBD-II Telematics → Cleaning → Feature Engineering (rolling, diff, interactions) → XGBoost/RandomForest → SHAP → Streamlit | Leakage-safe: failure_date not used as feature | Group split by vehicle_id")
