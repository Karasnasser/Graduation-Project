"""
Verify AI Staging Integration.
Checks if model exists, loads it, and runs a prediction.
"""
import sys
from pathlib import Path

import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from src.models.staging_model import StagingModel
    print("✅ Successfully imported StagingModel")
except ImportError as e:
    print(f"❌ Failed to import StagingModel: {e}")
    sys.exit(1)

def verify():
    print(" Verifying Staging Model...")
    
    # 1. Initialize
    try:
        model = StagingModel()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    if not model.model:
        print("⚠️ Model not loaded (check if xgb_staging.json exists)")
        return

    # 2. Predict (Stage 4 case)
    # High Creatinine (4.5), High BP (160), Low Hemoglobin (10)
    input_data = {
        "age": 65,
        "pressure_level": 160,
        "serum_creatinine": 4.5,
        "bun": 80,
        "serum_calcium": 8.5,
        "hemoglobin": 10.0
    }
    
    print(f"\n Input: {input_data}")
    
    try:
        result = model.predict_stage(input_data)
        print("\n Prediction Result:")
        print(f"   Stage: {result['predicted_stage']} (0=Healthy, 1-5=CKD)")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Probabilities: {result['probabilities']}")
        
        if result['predicted_stage'] in [4, 5]:
            print("✅ Logic Check: High creatinine correctly predicted advanced stage.")
        else:
            print("⚠️ Logic Check: Prediction seems unexpected for high creatinine.")
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

if __name__ == "__main__":
    verify()
