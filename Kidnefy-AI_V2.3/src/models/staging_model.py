"""
AI Staging Model Wrapper
Loads the trained XGBoost model and Scaler to predict CKD Stages (0-5).
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier

try:
    from config import STAGING_MODEL_DIR
except ImportError:
    STAGING_MODEL_DIR = Path("models/staging")

class StagingModel:
    def __init__(self):
        self.model_path = STAGING_MODEL_DIR / "xgb_staging.json"
        self.scaler_path = STAGING_MODEL_DIR / "staging_scaler.pkl"
        self.metadata_path = STAGING_MODEL_DIR / "metadata.json"
        
        self.model = None
        self.scaler = None
        self.features = []
        self.classes = []
        
        self.load_model()
        
    def load_model(self):
        """Load model, scaler, and metadata."""
        try:
            if not self.model_path.exists():
                print(f"[WARN] Staging model not found at {self.model_path}")
                return

            self.model = XGBClassifier()
            self.model.load_model(str(self.model_path))
            
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
            
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    meta = json.load(f)
                    self.features = meta.get("features", [])
                    self.classes = meta.get("classes", [])
            
            print("[OK] AI Staging Model Loaded")
        except Exception as e:
            print(f"[ERROR] Error loading staging model: {e}")

    def predict_stage(self, input_data: dict) -> dict:
        """
        Predict stage from input dictionary.
        Returns: {
            "predicted_stage": int (0-5),
            "confidence": float (0-1),
            "probabilities": dict
        }
        """
        if not self.model or not self.scaler:
            return {"error": "Model not loaded"}
        
        # Prepare input DataFrame with correct feature order
        # Fill missing features with defaults (0 or mean)
        data = {f: [input_data.get(f, 0)] for f in self.features}
        df = pd.DataFrame(data)
        
        # Scale
        X_scaled = self.scaler.transform(df)
        
        # Predict
        probs = self.model.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(probs)
        pred_stage = self.classes[pred_idx] if self.classes else pred_idx
        confidence = float(probs[pred_idx])
        
        return {
            "predicted_stage": int(pred_stage),
            "confidence": confidence,
            "probabilities": {str(k): float(v) for k, v in zip(self.classes, probs)}
        }
