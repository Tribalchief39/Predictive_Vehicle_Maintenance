# Viva Preparation — Predictive Vehicle Maintenance

## Q1: What problem does this solve?
Instead of scheduled or breakdown maintenance, we predict failures from OBD-II telemetry before they occur. Two outputs: is failure imminent? (classification) and how many days left? (RUL regression).

## Q2: What data did you use?
Synthetic telematics ~1970 rows, 50 vehicles. Sensors: engine_temp, rpm, oil_pressure, coolant_temp, vibration, brake_temp, brake_pad_wear, battery_voltage/health, speed, etc. Labels: failure_type, failure_date, imminent flags. IMPORTANT: Call it synthetic/prototype, not real fleet.

## Q3: How did you avoid data leakage?
Never use failure_date/failure_type/imminent flags as features. Split by vehicle_id (Group split), not random rows, so same vehicle not in train+test. Feature engineering per vehicle.

## Q4: How did you create targets?
failure_imminent = 1 if any imminent flag or failure_type != No Failure else 0. RUL = failure_date - timestamp (days), capped at 60d for stable regression. Fault diagnosis = original failure_type.

## Q5: Why XGBoost/RandomForest over Logistic/Linear?
Baseline LR gives decent ROC but misses non-linear sensor interactions (e.g., high temp + high load). Trees capture thresholds and interactions, handle imbalance via class_weight/scale_pos_weight.

## Q6: Metrics?
Classification: Accuracy, Precision/Recall/F1, ROC-AUC — recall most important (missed fault costly). Regression: MAE/RMSE/R² — MAE interpretable as avg days error.

## Q7: Class imbalance?
~98% No Failure. Used class_weight=balanced, scale_pos_weight=neg/pos for XGB, and focus on recall. Report that few failures limit RUL generalizability.

## Q8: Feature engineering?
Rolling mean/std (window 5 per vehicle) to denoise, diff for rate of change, deviation from median baseline, interactions like engine_stress=load*temp/100, battery_power_index, vibration×rpm, time features.

## Q9: Dashboard?
Streamlit: vehicle selector, timeline slider, what-if sliders, gauge for risk, RUL, recommendation, sensor trends (Plotly), telemetry cards, feature importances (SHAP-ready).

## Q10: Limitations?
Synthetic data, few failures → RUL overfits, RUL derived not measured, no claim of covering all vehicle types, needs real OEM data for production.

## Q11: Show overfitting check?
We used held-out vehicles (20% vehicles never seen). If train vs test metrics diverge a lot, mention regularization (max_depth, learning_rate) and that more data would help.

Be ready to run: `python src/train_classifier.py`, `streamlit run app.py`, and explain app.py:line 40-120.
