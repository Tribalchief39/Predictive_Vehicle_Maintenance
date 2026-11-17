# AI-Based Predictive Vehicle Maintenance & Fault Diagnosis Using Telematics / OBD-II Data

> **Academic prototype** — demonstrates how continuous OBD-II telemetry can be used to predict imminent faults and estimate Remaining Useful Life (RUL). Built for CSL236 AI/ML coursework.

## 🎯 Project Objectives
- **Classification:** Predict `failure_imminent` (0 = healthy, 1 = failure likely) from real-time sensor readings.
- **Diagnosis:** Identify fault type (`Battery Dead`, `Engine Overheat`, `Brake Pad Worn`, etc.)
- **Regression:** Predict **RUL in days** = `failure_date - timestamp` (capped at 60d for stability)

Leakage-safe: `failure_date`, `failure_type`, and target flags are **never** used as input features.

## 🗂️ Dataset
`data/synthetic_telemetry_data.csv` (~1,970 rows, 50 vehicles, 41 cols)
- **Source:** Synthetic telematics dataset (NOT real fleet data — described as prototype in report)
- Sensors: `engine_temp_c`, `engine_rpm`, `oil_pressure_psi`, `coolant_temp_c`, `vibration_level`, `brake_temp_c`, `brake_pad_wear_mm`, `battery_voltage_v`, `battery_health_percent`, `vehicle_speed_kph`, etc.
- Events: `timestamp`, `failure_date`, `failure_type`, `engine/brake/battery_issue_imminent`
- Class balance: ~98.3% `No Failure` / 1.7% failures → class-weighted models + vehicle-grouped split

## 🏗️ Pipeline
```
CSV → Pandas ingest → timestamp parsing → cleaning → target creation (failure_imminent + rul_days)
→ Feature engineering (rolling mean/std, diff, baseline deviation, interactions)
→ Group split by vehicle_id (prevents leakage) → Scaler
→ Classification: Logistic Regression (baseline) → RandomForest → XGBoost
→ Regression: Linear Regression → RandomForest → XGBoost (RUL capped 60d)
→ Evaluation (Precision/Recall/F1/ROC-AUC ; MAE/RMSE/R²) + SHAP/Importances
→ Joblib models → Streamlit Dashboard
```

## 📁 Folder Structure
```
PROJECT/
├── data/synthetic_telemetry_data.csv
├── src/
│   ├── preprocessing.py       # load, clean, create targets (no leakage)
│   ├── features.py            # rolling, diff, interactions
│   ├── train_classifier.py    # fault classifier with vehicle split
│   ├── train_rul.py           # RUL regressor
│   └── predict.py             # single-row inference
├── models/                    # saved joblibs + metrics
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modelling.ipynb
├── app.py                     # Streamlit dashboard
├── requirements.txt
└── README.md
```

## ⚡ Quick Start
```bash
pip install -r requirements.txt

# 1. Train classifier
python src/train_classifier.py
# 2. Train RUL
python src/train_rul.py
# 3. Launch dashboard
streamlit run app.py
```

## 📊 Feature Engineering
| Type | Example | Why |
|------|---------|-----|
| Raw | `engine_temp_c` | Current state |
| Rolling mean/std (w=5 per vehicle) | `engine_temp_c_roll_mean_5` | Denoise, detect drift |
| Rate of change | `engine_temp_c_diff` | Rapid deterioration |
| Baseline deviation | `engine_temp_c_dev_baseline` | Anomaly vs own history |
| Interaction | `engine_stress = load×temp/100` | Combined stress |
| Time | `hour`, `hours_since_last` | Operating context |

## 🧪 Evaluation (held-out vehicles, 20% vehicles)
- Classification prioritizes **Recall** (missed failure > false alarm)
- RUL reports **MAE** (e.g., 8.2d MAE means avg 8.2d error on capped 60d window)

See `models/classification_metrics.csv` and `models/rul_metrics.csv` after training.

## 🖥️ Dashboard (`app.py`)
- Vehicle selector + timeline slider
- Live sensor simulation sliders (what-if)
- Gauge + risk score, RUL prediction, maintenance recommendation
- Sensor time-series (Plotly) + telemetry cards
- Top feature importances (SHAP-ready)
- Model performance tables

## ⚠️ Limitations (mention in viva/report)
1. Synthetic data — not real OEM reliability proof
2. Few failures (~35/1970) → RUL is proof-of-concept, prone to overfit
3. RUL derived from timestamps, not measured wear
4. No explicit road-type / vehicle-category coverage unless documented
5. Production needs calibrated sensors + maintenance history

## 👥 Division of Work (as per PDF)
- **You:** EDA, preprocessing, features, classifier, SHAP
- **Friend:** RUL target, regressor, Streamlit, integration
- **Both:** Dataset audit, testing, report, PPT, viva prep

## 📚 References
Barreto et al. 2018 (ENIAC) — ML on automotive engine sensor data for driver profiling
## License
Academic use only.
