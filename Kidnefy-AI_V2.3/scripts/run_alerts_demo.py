import json
import shutil, os
from src.monitoring.longitudinal_monitor import LongitudinalMonitor
from src.monitoring.smart_alerts import SmartAlertEngine

# Clear any previous runs
if os.path.exists("data/demo_alerts"):
    shutil.rmtree("data/demo_alerts")

monitor = LongitudinalMonitor(data_dir="data/demo_alerts")
engine = SmartAlertEngine(monitor)
patient_id = "demo_patient_99"

# Add Historical Data (Patient starts stable, then deteriorates)
monitor.add_measurement(patient_id, "2025-01-01", egfr=65, creatinine=1.2, uacr=25, bp_systolic=125)
monitor.add_measurement(patient_id, "2025-03-01", egfr=63, creatinine=1.25, uacr=30, bp_systolic=130)
monitor.add_measurement(patient_id, "2025-06-01", egfr=60, creatinine=1.3, uacr=45, bp_systolic=135)
monitor.add_measurement(patient_id, "2025-09-01", egfr=50, creatinine=1.6, uacr=150, bp_systolic=150) # Sudden drop!
monitor.add_measurement(patient_id, "2025-12-01", egfr=42, creatinine=1.9, uacr=320, bp_systolic=165) # Worse!

print("=== 1. Longitudinal Trend Report ===")
print(monitor.format_trend_report(patient_id))

print("\n=== 2. Smart Alert Engine: Anomaly Detection (Isolation Forest) ===")
anomaly = engine.detect_anomalies(patient_id)
print(f"Is Anomaly: {anomaly.is_anomaly}")
print(f"Severity: {anomaly.severity}")
print("Anomalous Features:")
for f in anomaly.anomalous_features:
    feat = f["feature"]
    latest = f["latest_value"]
    mean = f["historical_mean"]
    z = f["z_score"]
    print(f"  - {feat}: latest {latest}, mean {mean:.2f} (Z={z:.1f})")

print("\n=== 3. Smart Alert Engine: Predictive Analytics ===")
pred = engine.predict_future_risk(patient_id)
print(f"Risk Score: {pred.overall_risk_score}/100 ({pred.risk_classification})")
print(f"Message:\n{pred.alert_message}")

print("\n=== 4. NLP Symptom Analysis ===")
symptom = engine.analyze_symptoms("عندي تورم شديد في رجلي ومش قادر اتنفس", patient_id)
print(f"Urgency: {symptom.urgency}")
print("Recommendations:")
for r in symptom.recommendations:
    print(f"  - {r}")
if symptom.correlation_with_labs:
    print(f"Lab Correlation:\n{symptom.correlation_with_labs}")
