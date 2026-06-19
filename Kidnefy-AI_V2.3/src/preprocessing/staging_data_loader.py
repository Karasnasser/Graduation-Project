"""
Staging Data Loader
Handles loading and preprocessing of the 5-stage CKD dataset.
Merges two compatible CKD datasets for richer training data.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict

try:
    from config import settings
except ImportError:
    class settings:
        RAW_DATA_DIR = Path("data/raw")
        MODEL_DIR = Path("models")

class StagingDataLoader:
    """
    Data loader for multi-class kidney disease staging (Stages 0-5).
    Merges two compatible CKD datasets for richer training data (~800 rows vs ~400).
    """
    
    PRIMARY_DATASET = "updated_ckd_dataset_with_stages.csv"
    SECONDARY_DATASET = "ckd_stages_dataset.csv"
    
    def __init__(self, data_path: str = None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = settings.RAW_DATA_DIR / self.PRIMARY_DATASET
            
        self.secondary_path = settings.RAW_DATA_DIR / self.SECONDARY_DATASET
        self.scaler = StandardScaler()
        self.features = []
        
    def load_data(self) -> pd.DataFrame:
        """Load and merge the two CKD staging datasets."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Primary dataset not found at {self.data_path}")
            
        df_primary = pd.read_csv(self.data_path)
        print(f"[OK] Loaded primary dataset: {self.data_path.name} ({len(df_primary)} rows)")
        
        if 'ckd_stage' not in df_primary.columns:
            raise ValueError("Target column 'ckd_stage' not found in primary dataset!")
        
        if self.secondary_path.exists():
            try:
                df_secondary = pd.read_csv(self.secondary_path)
                print(f"[OK] Loaded secondary dataset: {self.secondary_path.name} ({len(df_secondary)} rows)")
                
                if 'ckd_stage' not in df_secondary.columns:
                    print("[WARN]  Secondary dataset has no 'ckd_stage' column. Skipping merge.")
                else:
                    # Keep only common columns
                    common_cols = [c for c in df_primary.columns if c in df_secondary.columns]
                    df_secondary_filtered = df_secondary[common_cols].copy()
                    
                    df_merged = pd.concat(
                        [df_primary, df_secondary_filtered],
                        axis=0,
                        ignore_index=True
                    )
                    
                    print(f" Merged: {len(df_merged)} total rows (was {len(df_primary)}, "
                          f"added {len(df_secondary_filtered)} from secondary)")
                    return df_merged
                    
            except Exception as e:
                print(f"[WARN]  Could not merge secondary dataset: {e}. Using primary only.")
        else:
            print(f"[WARN]  Secondary dataset not found at {self.secondary_path}. Using primary only.")
        
        return df_primary

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess features and target (safe ops only — NO imputation here).
        
        [ALERT] ANTI-LEAKAGE: Imputation is deferred to get_train_test_split()
        so that it can be fit on the training set ONLY.
        """
        df = df.copy()
        
        # Drop non-feature columns
        drop_cols = ['cluster', 'id', '_source']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        target_col = 'ckd_stage'
        
        # Convert text CKD classification column to numeric if present
        if 'classification' in df.columns:
            df['classification'] = (df['classification'].astype(str)
                                    .str.strip().str.lower()
                                    .map({'ckd': 1, 'notckd': 0}))
        
        # Clean garbage values
        df = df.replace(['?', '\t?', 'ckd\t', 'notckd\t', '\t'], np.nan)
        
        # Encode CKD text columns
        text_encode_map = {
            'rbc':   {'normal': 0, 'abnormal': 1},
            'pc':    {'normal': 0, 'abnormal': 1},
            'pcc':   {'notpresent': 0, 'present': 1},
            'ba':    {'notpresent': 0, 'present': 1},
            'htn':   {'no': 0, 'yes': 1},
            'dm':    {'no': 0, 'yes': 1, ' yes': 1, '\tyes': 1},
            'cad':   {'no': 0, 'yes': 1, '\tno': 0},
            'appet': {'good': 1, 'poor': 0},
            'pe':    {'no': 0, 'yes': 1},
            'ane':   {'no': 0, 'yes': 1},
        }
        for col, mapping in text_encode_map.items():
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower().map(mapping)

        # [ALERT] DO NOT impute here — imputation must happen AFTER the split
        # to prevent preprocessing leakage.
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Keep only numeric
        X = X.select_dtypes(include=[np.number])
        self.features = X.columns.tolist()
        
        return X, y

    def get_train_test_split(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load, preprocess, and split data. Returns scaled features.
        
        [ALERT] ANTI-LEAKAGE PIPELINE:
          1. Load & clean (safe ops)
          2. Split into Train/Test (RAW data with NaNs)
          3. Impute: fit mean on TRAIN only, transform both
          4. Scale: fit scaler on TRAIN only, transform both
        """
        df = self.load_data()
        X, y = self.preprocess(df)
        
        print(f"   Features: {len(self.features)}")
        print(f"   Samples:  {len(X)}")
        print(f"   Stage distribution:\n{y.value_counts().sort_index().to_string()}")

        # ═══ SPLIT FIRST — before any imputation! ═══
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # ═══ IMPUTE AFTER SPLIT — fit on TRAIN only ═══
        train_means = X_train.mean(numeric_only=True)
        X_train = X_train.fillna(train_means)
        X_test = X_test.fillna(train_means)  # Use TRAIN means for test set
        print(f"   [OK] Imputation: fit on train ({len(X_train)} rows), applied to test ({len(X_test)} rows)")
        
        # ═══ SCALE — fit on TRAIN only ═══
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        scaler_path = settings.MODEL_DIR / "staging" / "staging_scaler.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, scaler_path)
        print(f"   Scaler saved to {scaler_path}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    loader = StagingDataLoader()
    X_train, X_test, y_train, y_test = loader.get_train_test_split()
    print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes:     X={X_test.shape}, y={y_test.shape}")
    print(f"Features: {loader.features}")
