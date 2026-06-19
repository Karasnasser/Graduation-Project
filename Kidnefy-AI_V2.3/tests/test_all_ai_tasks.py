"""
Comprehensive Test Suite - All 7 AI Tasks
Tests every component WITHOUT requiring trained models (unit tests).

يختبر كل المكونات الـ 7:
1. Multi-Biomarker Integration
2. Early-Stage Detection
3. CKD Staging
4. Risk Stratification
5. XAI (SHAP)
6. Longitudinal Monitoring
7. Fast Progressor Detection
"""

import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

PASS = 0
FAIL = 0

def assert_test(condition, test_name):
    global PASS, FAIL
    if condition:
        print(f"   ✅ {test_name}")
        PASS += 1
    else:
        print(f"   ❌ {test_name}")
        FAIL += 1


def test_gfr_calculator():
    """Test 2: Early-Stage Detection + Test 3: CKD Staging"""
    print("\n" + "=" * 60)
    print(" Test: GFR Calculator / CKD Staging / Early Detection")
    print("=" * 60)
    
    from src.staging import GFRCalculator, GFRStage
    
    calc = GFRCalculator()
    
    # Stage G1: eGFR >= 90
    result = calc.calculate_stage(egfr=105, age=30, is_female=False)
    assert_test(result.gfr_stage == GFRStage.G1, "Stage G1 (eGFR=105)")
    
    # Stage G2: eGFR 60-89
    result = calc.calculate_stage(egfr=75, age=50, is_female=False)
    assert_test(result.gfr_stage == GFRStage.G2, "Stage G2 (eGFR=75)")
    
    # Stage G3a: eGFR 45-59
    result = calc.calculate_stage(egfr=50, age=60, is_female=True)
    assert_test(result.gfr_stage == GFRStage.G3a, "Stage G3a (eGFR=50)")
    
    # Stage G3b: eGFR 30-44
    result = calc.calculate_stage(egfr=35, age=65, is_female=False)
    assert_test(result.gfr_stage == GFRStage.G3b, "Stage G3b (eGFR=35)")
    
    # Stage G4: eGFR 15-29
    result = calc.calculate_stage(egfr=22, age=70, is_female=True)
    assert_test(result.gfr_stage == GFRStage.G4, "Stage G4 (eGFR=22)")
    
    # Stage G5: eGFR < 15
    result = calc.calculate_stage(egfr=10, age=75, is_female=False)
    assert_test(result.gfr_stage == GFRStage.G5, "Stage G5 (eGFR=10)")
    
    # CKD-EPI eGFR calculation
    egfr = calc.calculate_egfr_ckdepi(creatinine=0.9, age=30, is_female=False)
    assert_test(egfr is not None and egfr > 90, f"CKD-EPI eGFR normal (got {egfr:.1f})")
    
    egfr_high_cr = calc.calculate_egfr_ckdepi(creatinine=3.0, age=60, is_female=False)
    assert_test(egfr_high_cr is not None and egfr_high_cr < 30, f"CKD-EPI eGFR low (got {egfr_high_cr:.1f})")


def test_risk_assessor():
    """Test 4: Risk Stratification + Enhanced Risk Score"""
    print("\n" + "=" * 60)
    print(" Test: Risk Stratification + Enhanced Risk Score")
    print("=" * 60)
    
    from src.staging import RiskAssessor, RiskLevel
    
    assessor = RiskAssessor()
    
    # Healthy person
    result = assessor.complete_assessment(
        ckd_probability=0.05,
        creatinine=0.9,
        acr=10.0,
        age=30,
        is_female=False
    )
    assert_test(result.risk_level == RiskLevel.LOW, f"Healthy: Low Risk (got {result.risk_level.value})")
    assert_test(result.enhanced_risk_score is not None, f"Enhanced risk score exists: {result.enhanced_risk_score}")
    assert_test(result.enhanced_risk_score < 20, f"Healthy: Low enhanced score ({result.enhanced_risk_score})")
    
    # Advanced CKD patient with diabetes
    result2 = assessor.complete_assessment(
        ckd_probability=0.95,
        creatinine=2.5,
        acr=350.0,
        age=70,
        is_female=True,
        other_values={
            'hba1c': 9.5,
            'uric_acid': 9.0,
            'bmi': 32,
            'smoking': 1,
            'diabetes_duration': 15
        }
    )
    assert_test(
        result2.risk_level in [RiskLevel.VERY_HIGH, RiskLevel.CRITICAL],
        f"Advanced CKD: High Risk (got {result2.risk_level.value})"
    )
    assert_test(result2.enhanced_risk_score > 50, f"Advanced CKD: High enhanced score ({result2.enhanced_risk_score})")
    assert_test(len(result2.alerts) >= 3, f"Multiple alerts generated ({len(result2.alerts)} alerts)")
    
    # Check specific alerts for HbA1c and uric acid
    alert_text = " ".join(result2.alerts)
    assert_test("HbA1c" in alert_text, "HbA1c alert generated")
    assert_test("اليوريك" in alert_text, "Uric acid alert generated")
    
    # Test enhanced risk score calculation
    score_low = assessor.calculate_enhanced_risk_score(
        egfr=100, acr=10, hba1c=5.0, creatinine=0.8
    )
    score_high = assessor.calculate_enhanced_risk_score(
        egfr=20, acr=400, hba1c=10.0, creatinine=3.5,
        uric_acid=9.0, bmi=35, smoking=True, diabetes_duration=15, age=70
    )
    assert_test(score_low < score_high, f"Risk scores ordered correctly ({score_low} < {score_high})")


def test_longitudinal_monitoring():
    """Test 6: Longitudinal Monitoring + Test 7: Fast Progressor"""
    print("\n" + "=" * 60)
    print(" Test: Longitudinal Monitoring + Fast Progressor Detection")
    print("=" * 60)
    
    from src.monitoring import LongitudinalMonitor
    
    test_dir = "data/_test_monitoring"
    monitor = LongitudinalMonitor(data_dir=test_dir)
    
    try:
        # === Fast Progressor (declining rapidly) ===
        patient_fast = "fast_001"
        monitor.add_measurement(patient_fast, "2025-01-01", egfr=80, creatinine=1.2, hba1c=8.0)
        monitor.add_measurement(patient_fast, "2025-04-01", egfr=72, creatinine=1.4)
        monitor.add_measurement(patient_fast, "2025-07-01", egfr=63, creatinine=1.7)
        monitor.add_measurement(patient_fast, "2025-10-01", egfr=54, creatinine=2.0)
        result = monitor.add_measurement(patient_fast, "2026-01-01", egfr=45, creatinine=2.4)
        
        assert_test(result['total_measurements'] == 5, f"5 measurements stored")
        
        slope = monitor.calculate_egfr_slope(patient_fast)
        assert_test(slope is not None and slope < -5, f"Slope is fast decline ({slope:.1f} mL/min/year)")
        
        assert_test(monitor.is_fast_progressor(patient_fast), "Fast progressor detected ✓")
        
        trend = monitor.calculate_trend(patient_fast)
        assert_test(trend.trend == "rapid_decline", f"Trend = rapid_decline (got {trend.trend})")
        assert_test(trend.is_fast_progressor, "Trend confirms fast progressor")
        assert_test(trend.predicted_years_to_esrd is not None, 
                   f"ESRD prediction: {trend.predicted_years_to_esrd:.1f} years")
        assert_test(trend.alert_message is not None, "Alert message generated")
        
        # === Stable Patient ===
        patient_stable = "stable_001"
        monitor.add_measurement(patient_stable, "2025-01-01", egfr=92)
        monitor.add_measurement(patient_stable, "2025-06-01", egfr=90)
        monitor.add_measurement(patient_stable, "2026-01-01", egfr=91)
        
        assert_test(not monitor.is_fast_progressor(patient_stable), "Stable patient NOT fast progressor")
        
        trend_stable = monitor.calculate_trend(patient_stable)
        assert_test(trend_stable.trend == "stable", f"Trend = stable (got {trend_stable.trend})")
        
        # === Insufficient Data ===
        patient_new = "new_001"
        monitor.add_measurement(patient_new, "2025-01-01", egfr=85)
        trend_new = monitor.calculate_trend(patient_new)
        assert_test(trend_new.trend == "insufficient_data", "Single measurement = insufficient data")
        
        # === Get All Fast Progressors ===
        fast_list = monitor.get_fast_progressors()
        assert_test(len(fast_list) == 1, f"Found 1 fast progressor in system")
        assert_test(fast_list[0]['patient_id'] == patient_fast, "Correct patient identified")
        
        # === Trend Report ===
        report = monitor.format_trend_report(patient_fast)
        assert_test("Rapid Decline" in report or "rapid_decline" in report.lower(), "Report shows decline")
        assert_test("ESRD" in report or "eGFR" in report, "Report mentions ESRD/eGFR")
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)


def test_shap_explainer():
    """Test 5: XAI - SHAP Explainer (module instantiation only, no trained model needed)"""
    print("\n" + "=" * 60)
    print(" Test: SHAP Explainer (XAI)")
    print("=" * 60)
    
    from src.explainability import SHAPExplainer
    
    explainer = SHAPExplainer()
    assert_test(explainer is not None, "SHAPExplainer created")
    assert_test(explainer.explainer is None, "Explainer not fitted (expected)")
    
    # Test without fitting - should return error
    import numpy as np
    result = explainer.explain_prediction(
        np.array([[1.0, 2.0, 3.0]]),
        feature_names=["f1", "f2", "f3"]
    )
    assert_test("error" in result, "Returns error when not fitted")
    
    # Test with a simple model (if sklearn available)
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # Create a tiny model for testing
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Fit SHAP
        explainer.fit(model, X, model_type="tree")
        assert_test(explainer.explainer is not None, "SHAP explainer fitted with RF")
        
        # Local explanation
        feature_names = ["feat_A", "feat_B", "feat_C", "feat_D", "feat_E"]
        explanation = explainer.explain_prediction(
            X[0:1], feature_names, top_k=3
        )
        assert_test("top_risk_factors" in explanation, "Local explanation has risk factors")
        assert_test("top_protective_factors" in explanation, "Local explanation has protective factors")
        assert_test("explanation_text" in explanation, "Explanation text generated")
        assert_test(len(explanation.get('top_features', [])) > 0, "Top features returned")
        
        # Global importance
        global_imp = explainer.global_feature_importance(X, feature_names)
        assert_test("feature_importance" in global_imp, "Global importance computed")
        assert_test(len(global_imp['feature_importance']) == 5, "All 5 features ranked")
        assert_test(global_imp['feature_importance'][0]['rank'] == 1, "Rank 1 exists")
        
        # Report generation
        report = explainer.get_explanation_report(X[0:1], feature_names)
        assert_test("Risk Factors" in report or "الخطر" in report, "Report contains risk info")
        
    except ImportError:
        print("   ⚠️ sklearn not available, skipping model-based SHAP tests")


def test_multi_biomarker_integration():
    """Test 1: Multi-Biomarker Integration"""
    print("\n" + "=" * 60)
    print(" Test: Multi-Biomarker Integration")
    print("=" * 60)
    
    from config import CKD_FEATURE_ORDER, CKD_FEATURE_DEFAULTS
    
    # Check that all required biomarkers from the project document are present
    required_biomarkers = ['sc', 'egfr', 'hba1c', 'bu', 'uric_acid', 'bmi', 'age', 'bp']
    
    available = set(CKD_FEATURE_ORDER)
    for biomarker in required_biomarkers:
        assert_test(biomarker in available, f"Biomarker '{biomarker}' in feature order")
    
    # Check defaults exist
    for biomarker in required_biomarkers:
        assert_test(
            biomarker in CKD_FEATURE_DEFAULTS,
            f"Default value for '{biomarker}': {CKD_FEATURE_DEFAULTS.get(biomarker, 'MISSING')}"
        )
    
    # Check new features
    new_features = ['hba1c', 'egfr', 'uacr', 'uric_acid', 'serum_albumin', 'bmi',
                    'bp_dia', 'smoking', 'dyslipidemia', 'diabetes_type', 'diabetes_duration']
    for feat in new_features:
        assert_test(feat in available, f"New feature '{feat}' integrated")


def test_main_system_import():
    """Test that main system can be imported and initialized."""
    print("\n" + "=" * 60)
    print(" Test: Main System Import & Components")
    print("=" * 60)
    
    try:
        from main import KidneyDiseasePredictionSystem
        system = KidneyDiseasePredictionSystem()
        
        assert_test(system.shap_explainer is not None, "SHAP explainer initialized")
        assert_test(system.longitudinal_monitor is not None, "Longitudinal monitor initialized")
        assert_test(system.risk_assessor is not None, "Risk assessor initialized")
        assert_test(system.gfr_calculator is not None, "GFR calculator initialized")
        assert_test(system.ensemble_model is not None, "Ensemble model initialized")
        assert_test(hasattr(system, 'add_patient_measurement'), "add_patient_measurement method exists")
        assert_test(hasattr(system, 'get_patient_trend'), "get_patient_trend method exists")
        
    except Exception as e:
        print(f"   ❌ System import failed: {e}")


def test_smart_alerts():
    """Test 8/9/10: Smart Alerts — Anomaly Detection, Predictive Analytics, NLP Symptom Analysis"""
    print("\n" + "=" * 60)
    print(" Test: Smart Alerts (Anomaly + Predictive + NLP)")
    print("=" * 60)
    
    from src.monitoring import LongitudinalMonitor, SmartAlertEngine
    
    test_dir = "data/_test_smart_alerts"
    monitor = LongitudinalMonitor(data_dir=test_dir)
    engine = SmartAlertEngine(monitor=monitor, gemini_rag=None)
    
    assert_test(engine is not None, "SmartAlertEngine created")
    
    try:
        # === Setup: Patient with declining data + anomaly ===
        pid = "smart_test_001"
        monitor.add_measurement(pid, "2025-01-01", egfr=85, creatinine=1.1, hba1c=7.0, uacr=20)
        monitor.add_measurement(pid, "2025-03-01", egfr=80, creatinine=1.2, hba1c=7.1, uacr=25)
        monitor.add_measurement(pid, "2025-05-01", egfr=76, creatinine=1.3, hba1c=7.3, uacr=30)
        monitor.add_measurement(pid, "2025-07-01", egfr=72, creatinine=1.4, hba1c=7.5, uacr=35)
        # Sudden spike — anomaly!
        monitor.add_measurement(pid, "2025-09-01", egfr=55, creatinine=2.5, hba1c=9.0, uacr=150)
        
        # --- Test 8: Anomaly Detection ---
        print("\n    Anomaly Detection:")
        anomaly = engine.detect_anomalies(pid)
        assert_test(anomaly is not None, "Anomaly result returned")
        assert_test(anomaly.is_anomaly, "Anomaly detected for sudden spike")
        assert_test(anomaly.severity in ["high", "medium"], f"Severity: {anomaly.severity}")
        assert_test(len(anomaly.anomalous_features) > 0, f"Found {len(anomaly.anomalous_features)} anomalous features")
        
        # Test with insufficient data
        pid_new = "smart_test_new"
        monitor.add_measurement(pid_new, "2025-01-01", egfr=90)
        anomaly_new = engine.detect_anomalies(pid_new)
        assert_test(not anomaly_new.is_anomaly, "No anomaly for insufficient data")
        
        # --- Test 9: Predictive Analytics ---
        print("\n    Predictive Analytics:")
        prediction = engine.predict_future_risk(pid)
        assert_test(prediction is not None, "Prediction result returned")
        assert_test(prediction.overall_risk_score > 0, f"Risk score: {prediction.overall_risk_score}")
        assert_test(prediction.risk_classification != "stable", f"Classification: {prediction.risk_classification}")
        assert_test('egfr' in prediction.biomarker_trends, "eGFR trend available")
        assert_test(prediction.alert_message is not None, "Alert message generated")
        
        # Check eGFR trend details
        egfr_trend = prediction.biomarker_trends.get('egfr', {})
        if egfr_trend.get('available'):
            assert_test(egfr_trend['slope_per_year'] < 0, f"eGFR declining ({egfr_trend['slope_per_year']} mL/min/year)")
        
        # Test with insufficient data
        pred_new = engine.predict_future_risk(pid_new)
        assert_test(pred_new.risk_classification == "insufficient_data", "Insufficient data handled")
        
        # --- Test 10: NLP Symptom Analysis ---
        print("\n    NLP Symptom Analysis (Keyword Fallback):")
        
        # Emergency symptoms (Arabic)
        result_emergency = engine.analyze_symptoms("ضيق تنفس شديد وألم صدر")
        assert_test(result_emergency.urgency == "emergency", f"Emergency detected: {result_emergency.urgency}")
        assert_test(len(result_emergency.recommendations) > 0, "Emergency recommendations given")
        
        # Urgent symptoms (Arabic)
        result_urgent = engine.analyze_symptoms("تورم في رجلي ودوخة وقلة البول")
        assert_test(result_urgent.urgency == "urgent", f"Urgent detected: {result_urgent.urgency}")
        
        # Routine symptoms (English)
        result_routine = engine.analyze_symptoms("I feel tired and thirsty")
        assert_test(result_routine.urgency == "routine", f"Routine detected: {result_routine.urgency}")
        
        # With patient context
        result_context = engine.analyze_symptoms("تورم في رجلي", patient_id=pid)
        assert_test(result_context.correlation_with_labs is not None, "Lab correlation provided")
        
        # --- Test: Alert Aggregation ---
        print("\n    Alert Aggregation:")
        alerts = engine.generate_smart_alerts(
            patient_id=pid,
            rule_based_alerts=[" تحذير: فشل كلوي!", "⚠️ كرياتينين مرتفع"]
        )
        assert_test(len(alerts) > 0, f"Generated {len(alerts)} smart alerts")
        
        # Check priority ordering (CRITICAL should come first)
        if len(alerts) > 1:
            priorities = [a.priority for a in alerts]
            assert_test(
                priorities[0] in ["CRITICAL", "WARNING"],
                f"Highest priority first: {priorities[0]}"
            )
        
        # Check serialization
        dicts = engine.alerts_to_dict(alerts)
        assert_test(len(dicts) == len(alerts), "Serialization works")
        assert_test('alert_type' in dicts[0], "Dict has alert_type")
        assert_test('priority' in dicts[0], "Dict has priority")
        
    finally:
        # Cleanup test data
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    print(" Kidney Disease AI - Comprehensive Test Suite")
    print("=" * 60)
    print("Testing all 10 AI tasks:\n"
          "  1. Multi-Biomarker Integration\n"
          "  2. Early-Stage Detection\n"
          "  3. CKD Staging\n"
          "  4. Risk Stratification\n"
          "  5. XAI (SHAP)\n"
          "  6. Longitudinal Monitoring\n"
          "  7. Fast Progressor Detection\n"
          "  8. Smart Alerts — Anomaly Detection\n"
          "  9. Smart Alerts — Predictive Analytics\n"
          "  10. Smart Alerts — NLP Symptom Analysis")
    
    # Run all tests
    test_multi_biomarker_integration()  # Task 1
    test_gfr_calculator()               # Task 2 + 3
    test_risk_assessor()                # Task 4
    test_shap_explainer()               # Task 5
    test_longitudinal_monitoring()      # Task 6 + 7
    test_smart_alerts()                 # Task 8 + 9 + 10
    test_main_system_import()           # Integration
    
    # Summary
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f" Results: {PASS}/{total} passed, {FAIL} failed")
    if FAIL == 0:
        print(" ALL TESTS PASSED!")
    else:
        print(f"⚠️ {FAIL} test(s) failed")
    print("=" * 60)

