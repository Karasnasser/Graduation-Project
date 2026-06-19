"""
Complex Patient Test Suite — Kidnefy-AI
========================================
Tests the trained CKD prediction models against 12 diverse, clinically
realistic patient profiles using the full feature set:

  Kidney Function : hba1c, sc, uacr, bu, bun
  Electrolytes    : sod, pot, cal, mag
  Demographics    : age, gender

Run from the project root:
    python scripts/test_complex_patients.py
"""

import sys
import os
from pathlib import Path

# ── project root on path ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from config import CKD_FEATURE_ORDER, CKD_FEATURE_DEFAULTS

# ═══════════════════════════════════════════════════════════════════════════════
# 12 Complex Patient Profiles
# ═══════════════════════════════════════════════════════════════════════════════
# Feature legend (matching CKD_FEATURE_ORDER):
#   age    – years
#   gender – 1=Male, 0=Female
#   hba1c  – % (glycated haemoglobin)
#   sc     – mg/dL  (serum creatinine)
#   uacr   – mg/g   (urine albumin-to-creatinine ratio)
#   bu     – mg/dL  (blood urea)
#   bun    – mg/dL  (blood urea nitrogen ≈ bu/2.14)
#   sod    – mEq/L  (sodium)
#   pot    – mEq/L  (potassium)
#   cal    – mg/dL  (calcium)
#   mag    – mg/dL  (magnesium)

COMPLEX_PATIENTS = [
    # ── 1. Perfectly healthy young adult ──────────────────────────────────────
    {
        "name": "Patient 01 — Healthy Young Adult (M, 28)",
        "expected": "NOT CKD",
        "clinical_note": "Annual check-up, all values well within normal range.",
        "features": {
            "age": 28, "gender": 1,
            "hba1c": 5.0, "sc": 0.85, "uacr": 5.0,
            "bu": 28.0, "bun": 13.0,
            "sod": 141.0, "pot": 4.2, "cal": 9.4, "mag": 2.1,
        },
    },
    # ── 2. Pre-CKD diabetic female ─────────────────────────────────────────────
    {
        "name": "Patient 02 — Pre-CKD Diabetic (F, 45)",
        "expected": "BORDERLINE / EARLY CKD",
        "clinical_note": "Type-2 DM with mildly elevated HbA1c and UACR. "
                         "eGFR still preserved (Stage G1-A2).",
        "features": {
            "age": 45, "gender": 0,
            "hba1c": 7.8, "sc": 1.05, "uacr": 55.0,
            "bu": 38.0, "bun": 18.0,
            "sod": 139.0, "pot": 4.6, "cal": 9.1, "mag": 1.9,
        },
    },
    # ── 3. Early CKD — Stage G2 A2 ─────────────────────────────────────────────
    {
        "name": "Patient 03 — Early CKD G2-A2 (M, 52)",
        "expected": "CKD",
        "clinical_note": "Hypertensive male, mildly reduced eGFR (60-89), "
                         "microalbuminuria present.",
        "features": {
            "age": 52, "gender": 1,
            "hba1c": 6.2, "sc": 1.45, "uacr": 120.0,
            "bu": 52.0, "bun": 24.0,
            "sod": 137.0, "pot": 4.8, "cal": 8.9, "mag": 1.8,
        },
    },
    # ── 4. Moderate CKD — Stage G3a A3 ─────────────────────────────────────────
    {
        "name": "Patient 04 — Moderate CKD G3a-A3 (F, 63)",
        "expected": "CKD",
        "clinical_note": "Proteinuria, eGFR 45-59. Phosphate regulation beginning to fail.",
        "features": {
            "age": 63, "gender": 0,
            "hba1c": 8.5, "sc": 1.95, "uacr": 380.0,
            "bu": 72.0, "bun": 34.0,
            "sod": 135.0, "pot": 5.1, "cal": 8.5, "mag": 2.4,
        },
    },
    # ── 5. Advanced CKD — Stage G4 A3 ──────────────────────────────────────────
    {
        "name": "Patient 05 — Advanced CKD G4-A3 (M, 71)",
        "expected": "CKD",
        "clinical_note": "eGFR 15-29, heavy proteinuria, anemia starting. "
                         "Referral to nephrologist mandatory.",
        "features": {
            "age": 71, "gender": 1,
            "hba1c": 9.1, "sc": 3.20, "uacr": 1200.0,
            "bu": 105.0, "bun": 49.0,
            "sod": 133.0, "pot": 5.5, "cal": 8.0, "mag": 2.8,
        },
    },
    # ── 6. End-Stage Kidney Disease (ESKD) — Stage G5 ──────────────────────────
    {
        "name": "Patient 06 — ESKD G5 (M, 58)",
        "expected": "CKD (ESKD)",
        "clinical_note": "Dialysis candidate. Severely elevated creatinine, "
                         "uremia, hyperkalaemia, hypocalcaemia.",
        "features": {
            "age": 58, "gender": 1,
            "hba1c": 10.2, "sc": 7.80, "uacr": 3500.0,
            "bu": 185.0, "bun": 86.0,
            "sod": 128.0, "pot": 6.2, "cal": 7.2, "mag": 3.5,
        },
    },
    # ── 7. Electrolyte Disorder — Hyponatremia focus ───────────────────────────
    {
        "name": "Patient 07 — Severe Hyponatremia (F, 77)",
        "expected": "BORDERLINE CKD",
        "clinical_note": "Elderly female, SIADH, acute-on-chronic kidney disease. "
                         "Na critically low. Creatinine mildly elevated.",
        "features": {
            "age": 77, "gender": 0,
            "hba1c": 6.4, "sc": 1.65, "uacr": 95.0,
            "bu": 60.0, "bun": 28.0,
            "sod": 118.0, "pot": 3.8, "cal": 8.7, "mag": 1.6,
        },
    },
    # ── 8. Hyperkalaemia with AKI ───────────────────────────────────────────────
    {
        "name": "Patient 08 — Hyperkalaemia + AKI (M, 44)",
        "expected": "CKD / AKI",
        "clinical_note": "Acute kidney injury after NSAID use. "
                         "Potassium dangerously elevated.",
        "features": {
            "age": 44, "gender": 1,
            "hba1c": 5.8, "sc": 2.85, "uacr": 210.0,
            "bu": 88.0, "bun": 41.0,
            "sod": 140.0, "pot": 6.9, "cal": 9.0, "mag": 2.2,
        },
    },
    # ── 9. Diabetic Nephropathy — Late stage ────────────────────────────────────
    {
        "name": "Patient 09 — Diabetic Nephropathy (F, 60)",
        "expected": "CKD",
        "clinical_note": "30-year T1DM, classic diabetic nephropathy. "
                         "Massive proteinuria, HbA1c chronically uncontrolled.",
        "features": {
            "age": 60, "gender": 0,
            "hba1c": 11.5, "sc": 2.40, "uacr": 2800.0,
            "bu": 92.0, "bun": 43.0,
            "sod": 134.0, "pot": 5.3, "cal": 8.2, "mag": 2.6,
        },
    },
    # ── 10. Healthy elderly — age risk only ─────────────────────────────────────
    {
        "name": "Patient 10 — Healthy Elderly (M, 80)",
        "expected": "BORDERLINE / NOT CKD",
        "clinical_note": "Healthy 80-year-old. eGFR naturally lower with age "
                         "but no proteinuria, otherwise normal labs.",
        "features": {
            "age": 80, "gender": 1,
            "hba1c": 5.4, "sc": 1.30, "uacr": 8.0,
            "bu": 44.0, "bun": 20.0,
            "sod": 140.0, "pot": 4.4, "cal": 9.2, "mag": 2.0,
        },
    },
    # ── 11. Young athlete — falsely elevated creatinine ─────────────────────────
    {
        "name": "Patient 11 — Athletic Muscle-mass Creatinine (M, 25)",
        "expected": "NOT CKD",
        "clinical_note": "Body-builder; elevated Sc due to high muscle mass, "
                         "NOT kidney disease. UACR normal.",
        "features": {
            "age": 25, "gender": 1,
            "hba1c": 4.9, "sc": 1.55, "uacr": 7.0,
            "bu": 30.0, "bun": 14.0,
            "sod": 142.0, "pot": 4.0, "cal": 9.6, "mag": 2.3,
        },
    },
    # ── 12. Mixed metabolic syndrome ────────────────────────────────────────────
    {
        "name": "Patient 12 — Metabolic Syndrome + CKD G3b (F, 55)",
        "expected": "CKD",
        "clinical_note": "Obesity, T2DM, hypertension, dyslipidemia. "
                         "Calcium low, magnesium low — secondary HPT.",
        "features": {
            "age": 55, "gender": 0,
            "hba1c": 8.9, "sc": 2.10, "uacr": 650.0,
            "bu": 78.0, "bun": 36.0,
            "sod": 136.0, "pot": 5.0, "cal": 7.8, "mag": 1.5,
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Inference helper
# ═══════════════════════════════════════════════════════════════════════════════

def compute_egfr(sc: float, age: float) -> float:
    """CKD-EPI 2009 (male approximation)."""
    sc = max(sc, 0.1)
    age = max(age, 1)
    kappa, alpha = 0.9, -0.411
    sc_k = sc / kappa
    return round(
        max(5.0, min(200.0,
            141.0 * (min(sc_k, 1.0) ** alpha) * (max(sc_k, 1.0) ** -1.209) * (0.993 ** age)
        )), 1
    )


def egfr_to_stage(egfr: float) -> str:
    if egfr >= 90:  return "G1 (≥90)"
    if egfr >= 60:  return "G2 (60-89)"
    if egfr >= 45:  return "G3a (45-59)"
    if egfr >= 30:  return "G3b (30-44)"
    if egfr >= 15:  return "G4 (15-29)"
    return              "G5 (<15) — ESKD"


def uacr_category(uacr: float) -> str:
    if uacr < 30:   return "A1 — Normal"
    if uacr < 300:  return "A2 — Microalbuminuria"
    return              "A3 — Macroalbuminuria"


def build_feature_vector(patient_features: dict,
                          full_feature_names: list,
                          feature_engineer,
                          scaler) -> np.ndarray:
    """Convert raw patient dict → scaled numpy vector."""
    defaults = CKD_FEATURE_DEFAULTS
    feat_order = CKD_FEATURE_ORDER

    # Compute egfr_computed from sc + age
    sc  = patient_features.get("sc",  defaults["sc"])
    age = patient_features.get("age", defaults["age"])
    egfr_computed = compute_egfr(sc, age)

    # Build ordered dict (only CKD_FEATURE_ORDER columns)
    row = {}
    for f in feat_order:
        row[f] = patient_features.get(f, defaults.get(f, 0))
    row["egfr_computed"] = egfr_computed

    df = pd.DataFrame([row])

    # Feature engineering (age bins → dummy columns)
    if feature_engineer is not None:
        df = feature_engineer.create_categorical_bins(df)

    # Align to the exact training column schema
    for col in full_feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[full_feature_names]

    arr = df.values.astype(np.float32)
    arr_scaled = scaler.transform(arr)
    return arr_scaled, egfr_computed


def run_tests():
    print("=" * 70)
    print("  Kidnefy-AI — Complex Patient Test Suite")
    print("  Features: age, gender, hba1c, sc, uacr, bu, bun, sod, pot, cal, mag")
    print("=" * 70)

    # ── Load saved artifacts ─────────────────────────────────────────────────
    model_dir = Path("models")

    try:
        scaler           = joblib.load(model_dir / "scaler.joblib")
        full_feat_names  = joblib.load(model_dir / "full_feature_names.joblib")
        feature_engineer = joblib.load(model_dir / "feature_engineer.joblib") \
                           if (model_dir / "feature_engineer.joblib").exists() else None
        print(f"[OK] Loaded preprocessing artifacts")
        print(f"     Training feature space ({len(full_feat_names)}): {full_feat_names}")
    except FileNotFoundError as e:
        print(f"[ERROR] Missing artifact: {e}")
        print("  → Please run training first:  python scripts/main.py train")
        sys.exit(1)

    # ── Load ML models ───────────────────────────────────────────────────────
    models = {}
    for name, fname in [
        ("RandomForest", "random_forest_model.joblib"),
        ("XGBoost",      "xgboost_model.joblib"),
        ("SVM",          "svm_model.joblib"),
    ]:
        p = model_dir / fname
        if p.exists():
            models[name] = joblib.load(p)
            print(f"[OK] Loaded {name}")
        else:
            print(f"[WARN] {name} not found at {p}")

    if not models:
        print("[ERROR] No ML models found. Please re-train.")
        sys.exit(1)

    # ── Ensemble weights ─────────────────────────────────────────────────────
    weights_path = model_dir / "ensemble_weights.joblib"
    ensemble_weights = joblib.load(weights_path) if weights_path.exists() else None

    print("\n" + "=" * 70)
    print(f"  Running {len(COMPLEX_PATIENTS)} complex patient cases")
    print("=" * 70)

    results_summary = []

    for i, patient in enumerate(COMPLEX_PATIENTS, 1):
        name      = patient["name"]
        expected  = patient["expected"]
        note      = patient["clinical_note"]
        feats     = patient["features"]

        # Build feature vector
        vec_scaled, egfr = build_feature_vector(
            feats, full_feat_names, feature_engineer, scaler
        )

        # ── Per-model predictions ────────────────────────────────────────────
        # IMPORTANT: LabelEncoder sorts alphabetically → 'ckd'=0, 'notckd'=1
        # predict_proba returns [P(class_0), P(class_1)] = [P(ckd), P(notckd)]
        # Therefore P(CKD) = proba[0]  ← index 0, NOT 1
        model_votes = {}
        model_probas = {}
        for mname, model in models.items():
            try:
                proba = model.predict_proba(vec_scaled)[0]  # [P(ckd), P(notckd)]
                # Predicted class: 0=ckd, so ckd when argmin (lowest class idx with highest proba)
                # More reliable: compare P(ckd) vs P(notckd) directly
                p_ckd = proba[0]      # P(CKD)   ← 'ckd' is class 0
                pred  = 1 if p_ckd >= 0.5 else 0   # 1=CKD, 0=notCKD
                model_votes[mname]  = pred
                model_probas[mname] = p_ckd         # P(CKD)
            except Exception as e:
                model_votes[mname]  = -1
                model_probas[mname] = float("nan")

        # ── Ensemble (weighted average of probas) ────────────────────────────
        valid_probas = [p for p in model_probas.values() if not np.isnan(p)]
        ensemble_proba = float(np.mean(valid_probas)) if valid_probas else 0.5
        ensemble_pred  = int(ensemble_proba >= 0.5)

        # Clinical metrics
        uacr_cat  = uacr_category(feats.get("uacr", 10))
        gfr_stage = egfr_to_stage(egfr)
        result_label = "⚠ CKD" if ensemble_pred else "✓ NOT CKD"

        # Risk level
        if ensemble_proba >= 0.80:   risk = "🔴 HIGH"
        elif ensemble_proba >= 0.50: risk = "🟠 MODERATE"
        elif ensemble_proba >= 0.30: risk = "🟡 BORDERLINE"
        else:                         risk = "🟢 LOW"

        # ── Print detailed result ────────────────────────────────────────────
        print(f"\n{'─'*70}")
        print(f"  [{i:02d}] {name}")
        print(f"{'─'*70}")
        print(f"  Clinical Note : {note}")
        print(f"  Expected      : {expected}")
        print()
        print(f"  ┌─ Input Labs ────────────────────────────────────────────┐")
        f = feats
        gender_str = "Male" if f.get("gender", 1) == 1 else "Female"
        print(f"  │  Age: {f.get('age')} yrs | Gender: {gender_str} | HbA1c: {f.get('hba1c')}%")
        print(f"  │  Serum Creatinine: {f.get('sc')} mg/dL | eGFR(calc): {egfr} mL/min/1.73m²")
        print(f"  │  UACR: {f.get('uacr')} mg/g | Blood Urea: {f.get('bu')} mg/dL | BUN: {f.get('bun')} mg/dL")
        print(f"  │  Na: {f.get('sod')} | K: {f.get('pot')} | Ca: {f.get('cal')} | Mg: {f.get('mag')} mEq/L")
        print(f"  └────────────────────────────────────────────────────────┘")
        print()
        print(f"  ┌─ Model Outputs ─────────────────────────────────────────┐")
        for mname, proba in model_probas.items():
            vote_str = "CKD" if model_votes[mname] == 1 else "NOT CKD"
            bar = "█" * int(proba * 20) + "░" * (20 - int(proba * 20))
            print(f"  │  {mname:<14} P(CKD)={proba:.3f}  [{bar}]  → {vote_str}")
        print(f"  │")
        print(f"  │  Ensemble P(CKD)  = {ensemble_proba:.4f}")
        print(f"  │  FINAL VERDICT    = {result_label}")
        print(f"  │  Risk Level       = {risk}")
        print(f"  │  GFR Stage        = {gfr_stage}")
        print(f"  │  Albuminuria      = {uacr_cat}")
        print(f"  └────────────────────────────────────────────────────────┘")

        results_summary.append({
            "patient": name,
            "expected": expected,
            "prediction": "CKD" if ensemble_pred else "NOT CKD",
            "proba": round(ensemble_proba, 4),
            "risk": risk,
            "gfr_stage": gfr_stage,
        })

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  SUMMARY TABLE")
    print(f"{'═'*70}")
    print(f"  {'#':<4} {'Patient':<42} {'Pred':<10} {'P(CKD)':<8} {'Risk'}")
    print(f"  {'─'*64}")
    for i, r in enumerate(results_summary, 1):
        short_name = r["patient"][:40]
        print(f"  {i:<4} {short_name:<42} {r['prediction']:<10} {r['proba']:<8.4f} {r['risk']}")

    ckd_count    = sum(1 for r in results_summary if r["prediction"] == "CKD")
    notckd_count = len(results_summary) - ckd_count
    print(f"\n  Total patients tested: {len(results_summary)}")
    print(f"  CKD predicted    : {ckd_count}")
    print(f"  NOT CKD predicted: {notckd_count}")
    print(f"{'═'*70}")
    print()


if __name__ == "__main__":
    run_tests()
