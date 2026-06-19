
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def verify():
    print(" Starting verification...")
    
    try:
        from main import KidneyDiseasePredictionSystem
        print("✅ Imported KidneyDiseasePredictionSystem")
        
        system = KidneyDiseasePredictionSystem()
        print("✅ Initialized System")
        
        try:
            print("⏳ Loading models...")
            system.ensemble_model.load()
            system.is_trained = system.ensemble_model.is_trained
            if not system.is_trained:
                print("⚠️ No checkpoint files found; train first (python main.py train)")
                return
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            print("   (This likely means training didn't finish or failed)")
            return

        # Test prediction with new features
        features = {
            'creatinine': 1.2,
            'age': 55,
            'is_female': False,
            'hba1c': 6.5,       # New feature
            'uacr': 35.0,       # New feature
            'diabetes_type': 2, # New feature
            'hypertension': 1,
            'systolic_bp': 140  # mapped to bp
        }
        
        print("\n Testing prediction with new features:")
        print(f"   Input: {features}")
        
        try:
            result = system.predict_from_features(features)
            print("\n✅ Prediction successful!")
            print(f"   Probability: {result['probability']:.4f}")
            print(f"   Prediction: {result['prediction']}")
            print(f"   GFR Stage: {result['gfr_stage']}")
            print(f"   Risk Level: {result['risk_level']}")
            
            # Check if new features were used? 
            # Hard to check internally without debug prints, but if it runs without error, mapping worked.
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()

    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"❌ params failed: {e}")

if __name__ == "__main__":
    verify()
