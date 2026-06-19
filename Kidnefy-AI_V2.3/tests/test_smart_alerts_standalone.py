"""
Standalone test for Smart Alerts Engine.
Tests all 3 AI components: Anomaly Detection, Predictive Analytics, NLP Symptom Analysis.
"""

import os
os.environ["PYTHONUTF8"] = "1"

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import shutil
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

PASS = 0
FAIL = 0

def assert_test(condition, test_name):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {test_name}")
    else:
        FAIL += 1
        print(f"  ❌ {test_name}")


def test_smart_alerts():
    """Test all Smart Alerts components."""
    
    print("\n" + "=" * 60)
    print(" Testing Smart Alerts Engine")
    print("=" * 60)
    
    # --- Step 1: Setup with temp directory ---
    print("\n Step 1: Setting up LongitudinalMonitor + SmartAlertEngine...")
    try:
        from src.monitoring.longitudinal_monitor import LongitudinalMonitor
        from src.monitoring.smart_alerts import (
            SmartAlertEngine, AnomalyResult, PredictiveResult, 
            SymptomAnalysis, SmartAlert, AlertPriority, AlertType
        )
        print("  ✅ Imports successful")
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return
    
    temp_dir = tempfile.mkdtemp(prefix="smart_alerts_test_")
    print(f"   Temp dir: {temp_dir}")
    
    try:
        monitor = LongitudinalMonitor(data_dir=temp_dir)
        engine = SmartAlertEngine(monitor=monitor, gemini_rag=None)
        assert_test(engine is not None, "SmartAlertEngine created")
        
        # --- Step 2: Test Anomaly Detection with insufficient data ---
        print("\n Step 2: Anomaly Detection — insufficient data...")
        result = engine.detect_anomalies("test_patient_001")
        assert_test(isinstance(result, AnomalyResult), "Returns AnomalyResult")
        assert_test(result.is_anomaly == False, "No anomaly with no data")
        assert_test(result.severity == "low", "Low severity with no data")
        
        # --- Step 3: Add enough measurements for anomaly detection ---
        print("\n Step 3: Adding patient measurements...")
        
        # Normal measurements (baseline)
        measurements = [
            ("2025-01-01", 85.0, 1.2, 30.0, 6.5, 125),
            ("2025-02-01", 83.0, 1.25, 32.0, 6.6, 128),
            ("2025-03-01", 82.0, 1.3, 34.0, 6.7, 126),
            ("2025-04-01", 80.0, 1.35, 36.0, 6.8, 130),
            ("2025-05-01", 78.0, 1.4, 38.0, 6.9, 132),
        ]
        
        for date, egfr, creat, uacr, hba1c, bp in measurements:
            monitor.add_measurement(
                patient_id="test_patient_001",
                date=date,
                egfr=egfr,
                creatinine=creat,
                uacr=uacr,
                hba1c=hba1c,
                bp_systolic=bp,
            )
        
        history = monitor.get_patient_history("test_patient_001")
        assert_test(len(history) == 5, f"5 measurements added (got {len(history)})")
        
        # --- Step 4: Anomaly detection on normal data ---
        print("\n Step 4: Anomaly Detection — normal data (should be no anomaly)...")
        result = engine.detect_anomalies("test_patient_001")
        assert_test(isinstance(result, AnomalyResult), "Returns AnomalyResult")
        print(f"     is_anomaly={result.is_anomaly}, score={result.anomaly_score:.3f}, severity={result.severity}")
        
        # --- Step 5: Add ANOMALOUS measurement and detect ---
        print("\n Step 5: Adding anomalous measurement...")
        monitor.add_measurement(
            patient_id="test_patient_001",
            date="2025-06-01",
            egfr=40.0,     # Sudden drop from ~78 → 40!
            creatinine=3.5, # Sudden spike from ~1.4 → 3.5!
            uacr=200.0,     # Huge jump from ~38 → 200!
            hba1c=9.5,      # Spike from ~6.9 → 9.5!
            bp_systolic=180, # Spike from ~132 → 180!
        )
        
        result = engine.detect_anomalies("test_patient_001")
        assert_test(isinstance(result, AnomalyResult), "Returns AnomalyResult after anomaly")
        assert_test(result.is_anomaly == True, f"Anomaly detected (is_anomaly={result.is_anomaly})")
        assert_test(len(result.anomalous_features) > 0, f"Anomalous features found: {len(result.anomalous_features)}")
        assert_test(result.severity in ["high", "medium"], f"Severity is {result.severity}")
        
        if result.anomalous_features:
            print(f"     Top anomalous feature: {result.anomalous_features[0]['feature']} "
                  f"(z={result.anomalous_features[0]['z_score']:.2f})")
        
        # --- Step 6: Predictive Analytics ---
        print("\n Step 6: Predictive Analytics...")
        prediction = engine.predict_future_risk("test_patient_001")
        assert_test(isinstance(prediction, PredictiveResult), "Returns PredictiveResult")
        assert_test(prediction.overall_risk_score >= 0, f"Risk score: {prediction.overall_risk_score}")
        assert_test(prediction.risk_classification in ["critical", "high", "moderate", "low", "stable", "insufficient_data"],
                    f"Classification: {prediction.risk_classification}")
        assert_test(len(prediction.alert_message) > 0, "Alert message generated")
        
        print(f"     Risk Score: {prediction.overall_risk_score}/100")
        print(f"     Classification: {prediction.risk_classification}")
        print(f"     Timeline: {prediction.predicted_timeline}")
        print(f"     Alert: {prediction.alert_message[:100]}...")
        
        # Check biomarker trends
        for bm, trend in prediction.biomarker_trends.items():
            if trend.get('available'):
                print(f"     {bm}: trend={trend['trend']}, slope={trend.get('slope_per_year', 'N/A')}/yr")
        
        # --- Step 7: NLP Symptom Analysis (keyword fallback) ---
        print("\n Step 7: NLP Symptom Analysis (keyword fallback)...")
        
        # Emergency symptoms
        analysis = engine.analyze_symptoms("لا أستطيع التنفس وعندي ألم صدر")
        assert_test(isinstance(analysis, SymptomAnalysis), "Returns SymptomAnalysis")
        assert_test(analysis.urgency == "emergency", f"Emergency urgency detected (got {analysis.urgency})")
        assert_test(len(analysis.recommendations) > 0, "Recommendations generated")
        print(f"     Urgency: {analysis.urgency}")
        print(f"     Conditions: {analysis.matched_conditions}")
        print(f"     Recommendations: {analysis.recommendations}")
        
        # Urgent symptoms
        analysis2 = engine.analyze_symptoms("عندي تورم في رجلي ودوخة")
        assert_test(analysis2.urgency == "urgent", f"Urgent detected (got {analysis2.urgency})")
        
        # Routine symptoms
        analysis3 = engine.analyze_symptoms("عندي تعب خفيف وعطش")
        assert_test(analysis3.urgency == "routine", f"Routine detected (got {analysis3.urgency})")
        
        # English symptoms
        analysis4 = engine.analyze_symptoms("I have severe swelling in my legs and dark urine")
        assert_test(analysis4.urgency == "urgent", f"English urgent detected (got {analysis4.urgency})")
        
        # With patient context
        analysis5 = engine.analyze_symptoms("عندي تورم في رجلي", patient_id="test_patient_001")
        assert_test(isinstance(analysis5, SymptomAnalysis), "Symptoms with patient context")
        print(f"     Lab correlation: {analysis5.correlation_with_labs}")
        
        # --- Step 8: Full Alert Aggregation ---
        print("\n Step 8: Generate Smart Alerts (all combined)...")
        
        rule_based = [
            " eGFR أقل من 60: المريض في مرحلة 3 أو أعلى",
            "⚠️ ارتفاع الكرياتينين عن المعدل الطبيعي"
        ]
        
        all_alerts = engine.generate_smart_alerts(
            "test_patient_001",
            include_rule_based=True,
            rule_based_alerts=rule_based
        )
        
        assert_test(isinstance(all_alerts, list), "Returns list of alerts")
        assert_test(len(all_alerts) > 0, f"Got {len(all_alerts)} alerts")
        assert_test(all(isinstance(a, SmartAlert) for a in all_alerts), "All are SmartAlert instances")
        
        # Check sorting (CRITICAL should come first)
        if len(all_alerts) >= 2:
            priorities = [a.priority for a in all_alerts]
            print(f"     Alert priorities (sorted): {priorities}")
            
        # Print all alerts
        for i, alert in enumerate(all_alerts):
            print(f"     [{i+1}] [{alert.priority}] [{alert.alert_type}] {alert.title}")
            print(f"         {alert.message[:80]}...")
        
        # --- Step 9: Convert to dict (serialization test) ---
        print("\n Step 9: Serialization test...")
        alerts_dict = engine.alerts_to_dict(all_alerts)
        assert_test(isinstance(alerts_dict, list), "alerts_to_dict returns list")
        assert_test(all(isinstance(d, dict) for d in alerts_dict), "All items are dicts")
        assert_test('alert_type' in alerts_dict[0], "Dict has alert_type key")
        assert_test('priority' in alerts_dict[0], "Dict has priority key")
        assert_test('message' in alerts_dict[0], "Dict has message key")
        
        # --- Step 10: Test with second patient (fast progressor) ---
        print("\n Step 10: Fast progressor patient...")
        
        fast_measurements = [
            ("2025-01-01", 70.0, 1.5, 50.0, 7.0, 140),
            ("2025-04-01", 60.0, 1.8, 80.0, 7.5, 145),
            ("2025-07-01", 48.0, 2.2, 120.0, 8.0, 150),
            ("2025-10-01", 35.0, 2.8, 180.0, 8.5, 158),
            ("2026-01-01", 22.0, 3.5, 280.0, 9.2, 170),
        ]
        
        for date, egfr, creat, uacr, hba1c, bp in fast_measurements:
            monitor.add_measurement(
                patient_id="fast_progressor",
                date=date,
                egfr=egfr,
                creatinine=creat,
                uacr=uacr,
                hba1c=hba1c,
                bp_systolic=bp,
            )
        
        prediction2 = engine.predict_future_risk("fast_progressor")
        assert_test(prediction2.risk_classification in ["critical", "high"],
                    f"Fast progressor classified as {prediction2.risk_classification}")
        assert_test(prediction2.overall_risk_score >= 50,
                    f"High risk score: {prediction2.overall_risk_score}")
        assert_test(prediction2.predicted_timeline is not None,
                    f"Timeline predicted: {prediction2.predicted_timeline}")
        
        print(f"     Risk Score: {prediction2.overall_risk_score}/100")
        print(f"     Classification: {prediction2.risk_classification}")
        print(f"     Timeline: {prediction2.predicted_timeline}")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n   Cleaned up temp dir")


if __name__ == "__main__":
    print(" Smart Alerts Engine — Comprehensive Test")
    print("=" * 60)
    
    test_smart_alerts()
    
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f" Results: {PASS}/{total} passed, {FAIL} failed")
    if FAIL == 0:
        print(" ALL TESTS PASSED!")
    else:
        print(f"⚠️ {FAIL} test(s) failed")
    print("=" * 60)
