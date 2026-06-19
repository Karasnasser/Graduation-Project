"""
Train Staging Model
Trains an XGBoost Classifier to predict CKD Stages (0-5).
"""

import sys
from pathlib import Path
import joblib
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Add project root to path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from src.preprocessing.staging_data_loader import StagingDataLoader
try:
    import config as settings
except ImportError:
    class settings:
        MODEL_DIR = Path("models")

def train_staging_model():
    print("=" * 60)
    print(" Training AI Staging Model (Stages 0-5)")
    print("=" * 60)
    
    # 1. Load Data
    loader = StagingDataLoader()
    try:
        X_train, X_test, y_train, y_test = loader.get_train_test_split()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Train XGBoost
    print("\n Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        objective='multi:softprob',
        num_class=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=10,
        eval_metric='mlogloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # 3. Evaluate
    print("\n Evaluation Results:")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 4. Save Model
    models_dir = settings.MODEL_DIR / "staging"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "xgb_staging.json"
    model.save_model(str(model_path))
    print(f"\n Model saved to {model_path}")
    
    # Save metadata (feature names)
    metadata = {
        "features": loader.features,
        "classes": [0, 1, 2, 3, 4, 5],
        "accuracy": float(accuracy)
    }
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("✅ Training Complete!")

if __name__ == "__main__":
    train_staging_model()
