"""
Overfitting Analysis Script
============================
Comprehensive check for overfitting in the kidney disease prediction models.

Checks performed:
1. Train vs Test accuracy gap for all models
2. K-Fold Cross-Validation (stability check)
3. Learning curve analysis (DL model)
4. Dataset size analysis
5. Feature count vs sample count ratio
"""

import sys
import io
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Insert the project root (parent of scripts/) so `src.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import brier_score_loss

from src.preprocessing.data_loader import DataLoader
from src.models.ml_models import MLModels

print("=" * 70)
print("[*] COMPREHENSIVE OVERFITTING ANALYSIS")
print("=" * 70)

# --- 1. Load Data ---
loader = DataLoader(data_dir="data/raw")
X_train, X_val, X_test, y_train, y_val, y_test, feature_names = loader.load_and_prepare_data()

print(f"\n[DATA] Dataset Size Analysis:")
print(f"   Train samples:      {X_train.shape[0]}")
print(f"   Validation samples: {X_val.shape[0]}")
print(f"   Test samples:       {X_test.shape[0]}")
print(f"   Total samples:      {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")
print(f"   Number of features: {X_train.shape[1]}")
print(f"   Samples/Features ratio: {X_train.shape[0] / X_train.shape[1]:.1f}x")
print(f"   (Rule of thumb: ratio > 10x is good, < 5x is risky)")

# Class distribution
for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    total = len(y)
    print(f"   {name:5} class distribution: {dist} ({counts[0]/total*100:.1f}% / {counts[1]/total*100:.1f}%)")

# --- 2. Train ML Models ---
print("\n" + "=" * 70)
print("[>>] TRAINING ML MODELS")
print("=" * 70)

ml = MLModels()
test_metrics = ml.train_all_models(X_train, y_train, X_test, y_test, X_val=X_val, y_val=y_val, calibrate=True, calibration_method="isotonic")

# Show AUC computation issues (if any)
auc_errors = []
for model_name, m in (test_metrics or {}).items():
    err = m.get("auc_roc_error")
    if err:
        auc_errors.append((model_name, err))
if auc_errors:
    print("\n[!] AUC-ROC could not be computed for some models:")
    for name, err in auc_errors:
        print(f"   - {name}: {err}")

# Show imbalance-robust metrics if available
print("\n" + "=" * 70)
print("[*] IMBALANCE-ROBUST METRICS (Test Set)")
print("=" * 70)
print(f"{'Model':20} | {'BalAcc':>8} | {'F1 macro':>8} | {'F1 w':>8} | {'Recall macro':>11} | {'AUC':>8}")
print("-" * 75)
for name in ['Random Forest', 'XGBoost', 'SVM']:
    m = (test_metrics or {}).get(name, {})
    if not m:
        continue
    bal = m.get("balanced_accuracy", None)
    f1m = m.get("f1_macro", None)
    f1w = m.get("f1_weighted", None)
    rcm = m.get("recall_macro", None)
    auc = m.get("auc_roc", None)
    def _fmt(x):
        return f"{x:0.4f}" if isinstance(x, (int, float)) else "   n/a "
    print(f"{name:20} | {_fmt(bal):>8} | {_fmt(f1m):>8} | {_fmt(f1w):>8} | {_fmt(rcm):>11} | {_fmt(auc):>8}")

# 2.4) Probability calibration sanity check (binary only)
print("\n" + "=" * 70)
print("[~] PROBABILITY CALIBRATION CHECK (Binary)")
print("=" * 70)
if len(np.unique(y_test)) == 2 and ml.best_model_name:
    try:
        base_proba = ml.models[ml.best_model_name].predict_proba(X_test)[:, 1]
        base_brier = brier_score_loss(y_test, base_proba)
        print(f"   Best model: {ml.best_model_name}")
        print(f"   Brier (uncalibrated): {base_brier:.6f}")
    except Exception as e:
        print(f"   [!] Could not compute uncalibrated Brier: {type(e).__name__}: {e}")

    try:
        cal_proba = ml.predict_proba(X_test, ml.best_model_name)[:, 1]
        cal_brier = brier_score_loss(y_test, cal_proba)
        method = getattr(ml, "calibration_method", None) or "unknown"
        print(f"   Brier (calibrated, {method}): {cal_brier:.6f}")
    except Exception as e:
        print(f"   [!] Could not compute calibrated Brier: {type(e).__name__}: {e}")
else:
    print("   Skipped (not a binary test set).")

# --- 2.5. Leakage / Duplicates Audit ---
print("\n" + "=" * 70)
print("[?] LEAKAGE & DUPLICATES AUDIT")
print("=" * 70)

# 2.5a) Show label mapping
try:
    le = loader.label_encoders.get("target")
    if le is not None and hasattr(le, "classes_"):
        mapping = {int(i): str(c) for i, c in enumerate(le.classes_.tolist())}
        print(f"   Label mapping (encoded -> original): {mapping}")
except Exception as e:
    print(f"   [!] Could not read label mapping: {e}")

# 2.5b) Duplicate rows across splits
def _hash_rows(X: np.ndarray) -> np.ndarray:
    Xc = np.ascontiguousarray(X)
    return np.apply_along_axis(lambda r: hash(r.tobytes()), 1, Xc)

try:
    h_train = _hash_rows(X_train)
    h_val = _hash_rows(X_val)
    h_test = _hash_rows(X_test)
    dup_train_test = len(np.intersect1d(h_train, h_test))
    dup_train_val = len(np.intersect1d(h_train, h_val))
    dup_val_test = len(np.intersect1d(h_val, h_test))
    print(f"   Exact duplicate feature-rows across splits:")
    print(f"     - Train x Test: {dup_train_test}")
    print(f"     - Train x Val:  {dup_train_val}")
    print(f"     - Val   x Test: {dup_val_test}")
    if dup_train_test > 0:
        print("   [WARNING] Train/Test duplicates detected. This can inflate performance.")
except Exception as e:
    print(f"   [!] Duplicate audit failed: {e}")

# 2.5c) Mutual information
try:
    print("\n   Top features by Mutual Information (Train set):")
    mi = mutual_info_classif(X_train, y_train, random_state=42, discrete_features=False)
    top_idx = np.argsort(mi)[::-1][:15]
    for rank, i in enumerate(top_idx, start=1):
        fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        print(f"     {rank:2}. {fname:30} MI={mi[i]:.4f}")
except Exception as e:
    print(f"   [!] Mutual information analysis failed: {e}")

# --- 3. Overfitting Diagnostic: Train vs Test ---
print("\n" + "=" * 70)
print("[>>] OVERFITTING DIAGNOSTIC: Train Accuracy vs Test Accuracy")
print("=" * 70)
print(f"{'Model':20} | {'Train Acc':>10} | {'Val Acc':>10} | {'Test Acc':>10} | {'Train-Test Gap':>14} | {'Status'}")
print("-" * 95)

results = []
for name in ['Random Forest', 'XGBoost', 'SVM']:
    model = ml.models.get(name)
    if model is not None:
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        gap = train_acc - test_acc

        if gap < 0.03:
            status = "[OK] GOOD"
        elif gap < 0.05:
            status = "[!]  SLIGHT OVERFIT"
        elif gap < 0.10:
            status = "[!!] MODERATE OVERFIT"
        else:
            status = "[!!] SEVERE OVERFIT"

        print(f"{name:20} | {train_acc:10.4f} | {val_acc:10.4f} | {test_acc:10.4f} | {gap:14.4f} | {status}")

        results.append({
            'model': name,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'gap': gap,
            'status': status
        })

# --- 4. Train F1 vs Test F1 ---
print("\n" + "=" * 70)
print("[>>] OVERFITTING DIAGNOSTIC: Train F1 vs Test F1")
print("=" * 70)
print(f"{'Model':20} | {'Train F1':>10} | {'Val F1':>10} | {'Test F1':>10} | {'Train-Test Gap':>14} | {'Status'}")
print("-" * 95)

for name in ['Random Forest', 'XGBoost', 'SVM']:
    model = ml.models.get(name)
    if model is not None:
        train_f1 = f1_score(y_train, model.predict(X_train), average='weighted')
        val_f1 = f1_score(y_val, model.predict(X_val), average='weighted')
        test_f1 = f1_score(y_test, model.predict(X_test), average='weighted')
        gap = train_f1 - test_f1

        if gap < 0.03:
            status = "[OK] GOOD"
        elif gap < 0.05:
            status = "[!]  SLIGHT OVERFIT"
        elif gap < 0.10:
            status = "[!!] MODERATE OVERFIT"
        else:
            status = "[!!] SEVERE OVERFIT"

        print(f"{name:20} | {train_f1:10.4f} | {val_f1:10.4f} | {test_f1:10.4f} | {gap:14.4f} | {status}")

# --- 5. K-Fold Cross-Validation (Stability Check) ---
print("\n" + "=" * 70)
print("[CV] K-FOLD CROSS-VALIDATION (10-Fold on Training Set)")
print("=" * 70)
print(f"{'Model':20} | {'Mean F1':>8} | {'Std F1':>8} | {'Min':>8} | {'Max':>8} | {'Stability'}")
print("-" * 80)

for name in ['Random Forest', 'XGBoost', 'SVM']:
    model = ml.models.get(name)
    if model is not None:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')

        stability = "[OK] STABLE" if scores.std() < 0.03 else "[!] UNSTABLE" if scores.std() < 0.05 else "[!!] VERY UNSTABLE"
        print(f"{name:20} | {scores.mean():8.4f} | {scores.std():8.4f} | {scores.min():8.4f} | {scores.max():8.4f} | {stability}")
        print(f"{'':20} | Folds: {[f'{s:.4f}' for s in scores]}")

# --- 6. Classification Report on TEST set ---
print("\n" + "=" * 70)
print("[REPORT] CLASSIFICATION REPORT (Test Set - Best Model)")
print("=" * 70)
best_model_name = ml.best_model_name
print(f"Best model: {best_model_name}")
print(classification_report(y_test, ml.predict(X_test)))

# --- 7. Check for potential issues ---
print("\n" + "=" * 70)
print("[CHECK] POTENTIAL ISSUES")
print("=" * 70)

issues = []
warnings_list = []

# Check 1: Sample size
total_samples = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
if total_samples < 500:
    issues.append(f"[!] Small dataset ({total_samples} samples). Results may not generalize well.")
elif total_samples < 1000:
    warnings_list.append(f"[W] Moderate dataset size ({total_samples} samples). Consider more data if possible.")
else:
    print(f"[OK] Dataset size is reasonable ({total_samples} samples)")

# Check 2: Feature/sample ratio
ratio = X_train.shape[0] / X_train.shape[1]
if ratio < 5:
    issues.append(f"[!!] Very low samples/features ratio ({ratio:.1f}x). High risk of overfitting!")
elif ratio < 10:
    warnings_list.append(f"[W] Low samples/features ratio ({ratio:.1f}x). Some overfitting risk.")
else:
    print(f"[OK] Samples/features ratio is adequate ({ratio:.1f}x)")

# Check 3: Class imbalance
unique, counts = np.unique(y_train, return_counts=True)
minority_pct = min(counts) / sum(counts) * 100
if minority_pct < 20:
    warnings_list.append(f"[W] Class imbalance detected. Minority class: {minority_pct:.1f}%")
else:
    print(f"[OK] Class balance is acceptable (minority: {minority_pct:.1f}%)")

# Check 4: Perfect or near-perfect training accuracy
for name in ['Random Forest', 'XGBoost', 'SVM']:
    model = ml.models.get(name)
    if model is not None:
        train_acc = accuracy_score(y_train, model.predict(X_train))
        if train_acc > 0.99:
            issues.append(f"[!!] {name}: Train accuracy = {train_acc:.4f} (nearly perfect). This is suspicious!")
        elif train_acc > 0.97:
            warnings_list.append(f"[W] {name}: Train accuracy = {train_acc:.4f} (very high). Monitor for overfitting.")

# Check 5: Train-Test gap
for r in results:
    if r['gap'] > 0.10:
        issues.append(f"[!!] {r['model']}: Train-Test gap = {r['gap']:.4f}. OVERFITTING DETECTED!")

if warnings_list:
    print("\n[WARNINGS]:")
    for w in warnings_list:
        print(f"   {w}")

if issues:
    print("\n[ISSUES FOUND]:")
    for issue in issues:
        print(f"   {issue}")
else:
    print("\n[OK] No critical overfitting issues detected based on standard metrics.")

# --- 8. Final Verdict ---
print("\n" + "=" * 70)
print("[VERDICT] FINAL VERDICT")
print("=" * 70)

max_gap = max(r['gap'] for r in results) if results else 0
avg_gap = np.mean([r['gap'] for r in results]) if results else 0

if max_gap > 0.10:
    print("[!!] OVERFITTING DETECTED")
    print(f"   Maximum Train-Test gap: {max_gap:.4f}")
    print("   Recommendation: Reduce model complexity, add regularization, or get more data.")
elif max_gap > 0.05:
    print("[!] SLIGHT OVERFITTING")
    print(f"   Maximum Train-Test gap: {max_gap:.4f}")
    print("   Recommendation: Monitor, consider tuning hyperparameters.")
else:
    print("[OK] NO SIGNIFICANT OVERFITTING")
    print(f"   Maximum Train-Test gap: {max_gap:.4f}")
    print(f"   Average Train-Test gap: {avg_gap:.4f}")
    print("   The models appear to generalize well.")

print("\n" + "=" * 70)
