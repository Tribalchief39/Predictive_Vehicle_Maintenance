# Predictive Vehicle Maintenance & Fault Diagnosis — Project Documentation

**Course:** CSL236 – Introduction to AI and Machine Learning
**Topic:** Predictive Vehicle Maintenance and Fault Diagnosis using Telematics / Engine Sensor Data

---

## 1. What the project does (in one paragraph)

Modern cars continuously send sensor readings (engine temperature, RPM, battery voltage, brake wear, etc.) through the OBD-II port. This project takes that kind of telemetry data and uses machine learning to answer two practical questions:

1. **Will this vehicle fail soon?** → binary classification (`failure_imminent` = 0 or 1)
2. **How many days are left before failure?** → regression, called **RUL (Remaining Useful Life)**, measured in days and capped at 60.

A Streamlit dashboard shows both answers for any selected vehicle, with a risk score, a maintenance recommendation and the sensor trends that led to it.

---

## 2. Dataset

| Item | Value |
|---|---|
| File | `data/synthetic_telemetry_data.csv` |
| Rows | 1,970 |
| Columns | 41 |
| Vehicles | 50 |
| Failure rows | 35 (~1.8% → highly imbalanced) |
| Source | Synthetic telematics dataset generated for this project (prototype, not real fleet data) |

Each row is one timestamped reading from one vehicle. Columns fall into three groups:

- **Sensors:** `engine_temp_c`, `engine_rpm`, `oil_pressure_psi`, `coolant_temp_c`, `fuel_level_percent`, `engine_load_percent`, `vibration_level`, `brake_temp_c`, `brake_pad_wear_mm`, `battery_voltage_v`, `battery_health_percent`, `vehicle_speed_kph`, …
- **Vehicle info:** `vehicle_id`, `brand`, `odometer_reading`, `engine_hours`, `timestamp`
- **Labels (never used as inputs):** `failure_date`, `failure_type`, and the three `*_imminent` flags

### Data contract (input ranges seen in the dataset)

| Feature | Type | Allowed range | Why it matters |
|---|---|---|---|
| engine_temp_c | float | 77.8 – 119.3 | Engine overheating is a direct failure signal |
| engine_rpm | float | 700 – 4,168 | High sustained RPM means more wear |
| oil_pressure_psi | float | 5.9 – 80.3 | Low oil pressure damages the engine |
| coolant_temp_c | float | 78.8 – 100.7 | Cooling system health |
| fuel_level_percent | float | 0 – 100 | Very low fuel can trigger faults |
| engine_load_percent | float | 0 – 100 | Load combined with heat = stress |
| vibration_level | float | 0 – 7.4 | Abnormal vibration → mechanical fault |
| brake_temp_c | float | −85 – 399 | Brake overheating indicates brake failure |
| brake_pad_wear_mm | float | 5.9 – 9.8 | Thinner pads = pads worn out |
| battery_voltage_v | float | 9.1 – 14.5 | Low voltage → dead battery |
| battery_health_percent | float | 90.2 – 99.4 | Battery condition |
| vehicle_speed_kph | float | 0 – 180 | Operating context |
| odometer_reading | float | 13,579 – 100,410 | Overall vehicle age/wear |
| **failure_imminent** | int (target) | {0, 1} | 1 = failure expected soon |

---

## 3. How the data flows (pipeline)

```
CSV
 │
 ├── 1. Load & clean          preprocessing.py  → parse timestamps, fill NaNs, drop duplicates
 ├── 2. Create targets        preprocessing.py  → failure_imminent (0/1) and rul_days
 ├── 3. Feature engineering   features.py       → rolling mean/std, diff, deviation from baseline, interactions
 ├── 4. Vehicle-grouped split                   → 80% vehicles train / 20% vehicles test
 ├── 5. Train & compare                          Logistic Regression / Linear Regression, RandomForest, XGBoost
 ├── 6. Evaluate                                 Accuracy, ROC-AUC, MAE, RMSE, R²
 ├── 7. Save models                             models/*.joblib
 └── 8. Dashboard                               app.py (Streamlit + Plotly)
```

**Why the split is grouped by `vehicle_id`:** readings from the same vehicle are very similar. A random row split would put the same vehicle in both train and test, so the model would look better than it really is (data leakage). Splitting by vehicle gives an honest "new car" test.

---

## 4. Features used by the model

| Type | Example | Why we add it |
|---|---|---|
| Raw sensor | `engine_temp_c` | Current state |
| Rolling mean / std (window = 5 readings per vehicle) | `engine_temp_c_roll_mean_5` | Smooths noise, shows drift |
| Rate of change | `engine_temp_c_diff` | Sudden deterioration |
| Deviation from own baseline | `engine_temp_c_dev_baseline` | "Is this car worse than its usual self?" |
| Interaction | `engine_stress = load × temp / 100` | Combined stress is worse than either alone |
| Time | `hour`, `hours_since_last` | Operating context |

Result: **91 numeric features** after engineering. Any label column (`failure_date`, `failure_type`, `*_imminent`, `rul_days`) is explicitly excluded so the model cannot cheat.

---

## 5. Models compared and results

Split: 40 vehicles for training (1,577 rows) and 10 vehicles for testing (393 rows).

### Classification — `failure_imminent`

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression (baseline, scaled) | 0.987 | 0.989 |
| RandomForest (class_weight = balanced) | 0.997 | 0.998 |
| **XGBoost** (scale_pos_weight for imbalance) | 0.995 | **0.9996** |

XGBoost's report on the test vehicles: precision 0.86, recall 0.86, F1 0.86 for the failure class — it caught 6 of 7 real failure rows.

### Regression — RUL in days (capped at 60)

| Model | MAE (days) | RMSE | R² |
|---|---|---|---|
| Linear Regression | 3.79 | 6.87 | 0.15 |
| **RandomForest** | **0.36** | 3.29 | 0.80 |
| XGBoost | 0.51 | **3.21** | **0.81** |

**Chosen models:** XGBoost for classification (best ROC-AUC) and XGBoost for RUL (best R²/RMSE); both are saved as `models/classifier.joblib` and `models/rul_model.joblib`, which is what the dashboard loads.

**Why Accuracy alone is not enough:** 98% of rows are healthy, so a model that always says "no failure" gets 98% accuracy but is useless. That is why we report ROC-AUC and per-class recall, and why class weights / `scale_pos_weight` are used.

---

## 6. Dashboard (`app.py`)

Run with `streamlit run app.py`. It shows:

- Vehicle selector + timeline slider (replay any reading in time)
- What-if sliders for engine temp, coolant, vibration, battery health, brake wear
- Failure probability gauge + RUL in days + maintenance recommendation
- Sensor timeline chart (Plotly)
- "Why this prediction?" — top 5 model feature importances with the current values
- Live telemetry cards (temps, RPM, load, fuel, etc.)

---

## 7. How to run (environment used: Python 3.14)

```bash
pip install -r requirements.txt

python src/train_classifier.py     # trains the failure classifier
python src/train_rul.py            # trains the RUL regressor
python notebooks/01_EDA.py         # optional: saves EDA plots to reports/
streamlit run app.py               # opens the dashboard

python run_all.py                  # runs all three scripts in one go
```

Outputs: models in `models/`, metrics CSVs (`classification_metrics.csv`, `rul_metrics.csv`), plots in `reports/`.

**Required library versions** (pinned in `requirements.txt`): pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1, xgboost 3.4.1, matplotlib 3.11.2, seaborn 0.13.2, streamlit 1.67.0, plotly 7.1.0, joblib 1.6.0.

---

## 8. File map

| Path | Purpose |
|---|---|
| `data/synthetic_telemetry_data.csv` | The dataset |
| `src/preprocessing.py` | Load, clean, create targets, list leakage-safe features |
| `src/features.py` | Rolling / diff / baseline / interaction / time features |
| `src/train_classifier.py` | Trains LR, RF, XGBoost for `failure_imminent` |
| `src/train_rul.py` | Trains LR, RF, XGBoost for RUL (capped 60 days) |
| `src/predict.py` | Single-row inference helper |
| `app.py` | Streamlit dashboard |
| `notebooks/01_EDA.py` | Class balance, sensor histograms, correlation heatmap |
| `models/` | Saved `.joblib` models, feature lists, scalers, metrics |
| `reports/` | EDA plots |

---

## 9. Design decisions worth explaining in the viva

1. **Leakage prevention** — `failure_date` and `failure_type` are removed from the feature list; otherwise the model would just read the answer.
2. **Vehicle-grouped split** — same vehicle never appears in both train and test.
3. **RUL capped at 60 days** — most rows are healthy with RUL in the thousands of days; without capping, the regressor would be dominated by "healthy" values and ignore the near-failure window we actually care about.
4. **Class imbalance handling** — `class_weight="balanced"` (LR, RF) and `scale_pos_weight` (XGBoost), plus recall-focused reporting.
5. **Tree models need no scaling for the main path** — the classifier/RUL models used by the dashboard are tree-based, so raw engineered features are fine; scalers are still saved for the linear baselines.
6. **Separate offline training and online dashboard** — models are trained by scripts and loaded once by the app, not retrained on every interaction.

---

## 10. Limitations

1. Data is **synthetic** — it demonstrates the method, it does not prove real-world reliability.
2. Only 35 failure rows out of 1,970, so results are optimistic and the RUL model is a proof of concept.
3. RUL is derived from timestamps, not from physically measured component wear.
4. No road-type / driving-condition categories.
5. The dashboard is a local Streamlit app, not a deployed service.

---

## 11. Short answers to likely teacher questions

| Question | Answer |
|---|---|
| Why two targets? | Fault detection (will it fail?) and fault timing (how long left?) are different business questions; one is classification, the other regression. |
| Why compare three models? | To show a baseline (linear), a robust ensemble (RF) and a strong boosted model (XGBoost), and to justify the final choice with numbers. |
| Why is R² low for Linear Regression? | The RUL relationship is non-linear and mostly saturated at the 60-day cap; trees capture that, a straight line cannot. |
| What happens if an input is missing/NaN? | `preprocessing.clean_dataframe` and `features.engineer_features` fill numeric NaNs (median / 0), and `app.py` reindexes features with `fill_value=0`, so the model always receives a complete row. |
| How do you know the model needs retraining? | If live sensor distributions drift from the training data, or recall on recent failures drops — that is what a drift check on logged inputs would monitor. |
| What would you add next? | A FastAPI endpoint with a validated request schema, MLflow experiment tracking, a drift-monitoring script and a Docker deployment. |
