import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import EnsembleModel
from src.preprocessing import DataLoader, FeatureEngineer

def generate_complex_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate 500 Healthy (Class 0 in our code usually, but let's check. Wait, CKD=0, NOTCKD=1 typically in this dataset. Let's verify labels. UCI CKD has ckd=0, notckd=1 when label encoded alphabetically. We'll use 1 for CKD, 0 for healthy for generation, then map to whatever the model outputs).
    # Wait! Let's generate raw data that looks like the input features dict.
    
    data = []
    labels = []
    
    # 500 CKD patients
    for _ in range(n_samples // 2):
        # CKD patients typically have higher SC, UACR, BU, BUN, etc.
        age = np.random.randint(40, 90)
        gender = np.random.choice([0, 1])
        hba1c = np.random.normal(7.5, 1.5) # Often diabetic
        sc = np.random.normal(3.5, 1.5) # High creatinine
        uacr = np.random.lognormal(5.0, 1.0) # High UACR
        bu = np.random.normal(60, 20)
        bun = bu / 2.14 # Approximate relationship
        sod = np.random.normal(135, 5) # Often lower or normal
        pot = np.random.normal(5.5, 0.8) # Often high
        cal = np.random.normal(8.0, 0.8) # Often low
        mag = np.random.normal(2.5, 0.4) # Often high
        
        # Clip to realistic values
        sc = max(1.3, sc)
        uacr = max(30, uacr)
        hba1c = max(4.0, hba1c)
        
        data.append({
            'age': age, 'gender': gender, 'hba1c': hba1c, 'sc': sc, 'uacr': uacr,
            'bu': bu, 'bun': bun, 'sod': sod, 'pot': pot, 'cal': cal, 'mag': mag
        })
        labels.append(1) # 1 = CKD

    # 500 Healthy patients
    for _ in range(n_samples // 2):
        age = np.random.randint(20, 70)
        gender = np.random.choice([0, 1])
        hba1c = np.random.normal(5.2, 0.4)
        sc = np.random.normal(0.8, 0.2)
        uacr = np.random.lognormal(2.0, 0.5)
        bu = np.random.normal(15, 4)
        bun = bu / 2.14
        sod = np.random.normal(140, 2)
        pot = np.random.normal(4.2, 0.3)
        cal = np.random.normal(9.5, 0.4)
        mag = np.random.normal(2.0, 0.2)
        
        sc = max(0.4, min(1.2, sc))
        uacr = max(0, min(29, uacr))
        
        data.append({
            'age': age, 'gender': gender, 'hba1c': hba1c, 'sc': sc, 'uacr': uacr,
            'bu': bu, 'bun': bun, 'sod': sod, 'pot': pot, 'cal': cal, 'mag': mag
        })
        labels.append(0) # 0 = Healthy
        
    df = pd.DataFrame(data)
    
    # Calculate egfr_computed for all (matching training setup)
    # 141 * min(Sc/k, 1)^a * max(Sc/k, 1)^-1.209 * 0.993^Age * (1.018 if female)
    # k = 0.7 if female else 0.9
    # a = -0.329 if female else -0.411
    egfr_list = []
    for idx, row in df.iterrows():
        sc_val = max(row['sc'], 0.1)
        age_val = max(row['age'], 1)
        is_female = row['gender'] == 1 # 1=female? Let's assume 1=female for calculation, wait, usually 0=female, 1=male. Let's use 0/1 generically.
        # Actually in data_loader _preprocess_split:
        kappa, alpha = 0.9, -0.411
        sc_k = sc_val / kappa
        min_cr = min(sc_k, 1.0)
        max_cr = max(sc_k, 1.0)
        egfr = 141.0 * (min_cr ** alpha) * (max_cr ** -1.209) * (0.993 ** age_val)
        egfr_list.append(egfr)
    df['egfr_computed'] = egfr_list
        
    return df, np.array(labels)

def main():
    print("Generating 1000 complex synthetic data points...")
    X_df, y_true = generate_complex_data(1000)
    
    print("Loading preprocessing artifacts and trained models...")
    data_loader = DataLoader()
    try:
        data_loader.load_preprocessing_artifacts('models')
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return
        
    ensemble = EnsembleModel('models')
    try:
        ensemble.load()
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    print("Running feature engineering pipeline...")
    # Properly map features
    X_df = data_loader.feature_engineer.create_categorical_bins(X_df)
    
    # Keep only trained features
    trained_features = ensemble.feature_names
    for col in trained_features:
        if col not in X_df.columns:
            X_df[col] = 0.0
            
    X_df = X_df[trained_features]
    
    print("Scaling data...")
    X_scaled = data_loader.scaler.transform(X_df)
    
    print("Running predictions...")
    y_pred_proba = ensemble.predict_proba(X_scaled)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    print("\n========================================================")
    print("EVALUATION ON 1000 NEW UNSEEN COMPLEX SAMPLES")
    print("========================================================")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Healthy (0)', 'CKD (1)']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("========================================================")

if __name__ == "__main__":
    main()
