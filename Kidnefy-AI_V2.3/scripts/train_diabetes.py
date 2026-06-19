"""
Diabetes Model Training Pipeline
سكريبت تدريب موديلات التنبؤ بالسكري

Directly links the preprocessing algorithm (DiabetesPreprocessor) to:
1. ML Models (Random Forest, XGBoost, SVM)
2. Deep Learning Model (Neural Network)
3. Ensemble Model

Usage:
    python train_diabetes.py
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add project root to path to allow imports
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

try:
    from src.preprocessing.diabetes_preprocessing import DiabetesPreprocessor
    from src.models.ml_models import MLModels
    from src.models.dl_models import DeepLearningModel
    from src.models.ensemble import EnsembleModel
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("   Make sure you are running from the project root or src is in PYTHONPATH.")
    sys.exit(1)


def train_diabetes_models():
    print("=" * 60)
    print(" TARGETED TRAINING: Diabetes Prediction")
    print("   تدريب موجه: التنبؤ بالسكري")
    print("=" * 60)

    # -----------------------------------------------------------
    # 1. Run Preprocessing Algorithm
    # -----------------------------------------------------------
    print("\n[Stage 1] Running Preprocessing Algorithm...")
    preprocessor = DiabetesPreprocessor()
    try:
        X, y, report = preprocessor.run()
    except Exception as e:
        print(f"\n❌ Preprocessing Failed: {e}")
        return

    # -----------------------------------------------------------
    # 2. Split Data (80% Train, 20% Test)
    # -----------------------------------------------------------
    print("\n[Stage 2] Splitting Data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape:  {X_test.shape}")

    # -----------------------------------------------------------
    # 3. Train ML Models (RF, XGB, SVM)
    # -----------------------------------------------------------
    print("\n[Stage 3] Training ML Models...")
    ml_models = MLModels()
    ml_metrics = ml_models.train_all_models(X_train, y_train, X_test, y_test)
    
    # Save ML models
    models_dir = Path("models/diabetes")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # We save manually because ml_models.save_models uses a fixed path
    print(f"   Saving ML models to {models_dir}...")
    joblib.dump(ml_models.models['Random Forest'], models_dir / "rf_model.pkl")
    joblib.dump(ml_models.models['XGBoost'], models_dir / "xgb_model.pkl")
    joblib.dump(ml_models.models['SVM'], models_dir / "svm_model.pkl")

    # -----------------------------------------------------------
    # 4. Train Deep Learning Model
    # -----------------------------------------------------------
    print("\n[Stage 4] Training Deep Learning Model...")
    input_dim = X_train.shape[1]
    dl_model = DeepLearningModel()
    dl_model.build_model(input_dim=input_dim)
    
    # Train
    history = dl_model.train(X_train, y_train, epochs=20, batch_size=32)
    
    # Evaluate
    dl_metrics = dl_model.evaluate(X_test, y_test)
    dl_accuracy = dl_metrics.get('accuracy', 0.0)
    print(f"   Deep Learning Accuracy: {dl_accuracy:.4f}")
    
    # Save DL model
    dl_model.save_model(str(models_dir / "dl_model.h5"))

    # -----------------------------------------------------------
    # 5. Train Ensemble Model
    # -----------------------------------------------------------
    print("\n[Stage 5] Training Ensemble Model...")
    ensemble = EnsembleModel()
    
    # Set the trained models into the ensemble
    ensemble.ml_models.models = ml_models.models  # Pass the entire dictionary
    ensemble.dl_model = dl_model
    
    # Initialize weights (simple average for now, or optimize)
    ensemble.weights = {'Random Forest': 0.25, 'XGBoost': 0.25, 'SVM': 0.25, 'Deep Learning': 0.25}
    
    # Evaluate Ensemble
    final_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, final_pred)
    
    print("\n" + "=" * 60)
    print(" FINAL RESULTS (Accuracy)")
    print("=" * 60)
    
    rf_acc = ml_metrics.get('Random Forest', {}).get('accuracy', 0)
    xgb_acc = ml_metrics.get('XGBoost', {}).get('accuracy', 0)
    svm_acc = ml_metrics.get('SVM', {}).get('accuracy', 0)
    
    print(f"   Random Forest: {rf_acc:.4f}")
    print(f"   XGBoost:       {xgb_acc:.4f}")
    print(f"   SVM:           {svm_acc:.4f}")
    print(f"   Deep Learning: {dl_accuracy:.4f}")
    print("-" * 30)
    print(f"    ENSEMBLE:   {ensemble_acc:.4f}")
    print("=" * 60)
    
    print("\n✅ Training Complete. Models saved in 'models/diabetes/'")


if __name__ == "__main__":
    train_diabetes_models()
