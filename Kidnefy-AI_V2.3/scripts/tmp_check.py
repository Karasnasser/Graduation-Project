import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(os.getcwd())))

from src.preprocessing.data_loader import DataLoader
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    loader = DataLoader(data_dir="data/raw")
    X_train, X_val, X_test, y_train, y_val, y_test, features = loader.load_and_prepare_data()
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    
    # Sort top features
    feature_imp = list(zip(features, importances))
    feature_imp.sort(key=lambda x: x[1], reverse=True)
    
    print("\n--- TOP 10 FEATURES CAUSING 99% ACCURACY ---")
    for f, imp in feature_imp[:10]:
        print(f"{f:20s}: {imp:.4f}")
