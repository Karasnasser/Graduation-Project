"""
Data Loader Module
Handles loading, merging, and preprocessing of kidney disease datasets.
يقوم هذا الملف بتحميل ودمج ومعالجة مجموعات بيانات أمراض الكلى.
"""

import sys
from pathlib import Path

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import warnings
import joblib
warnings.filterwarnings('ignore')

from src.preprocessing.feature_engineering import FeatureEngineer


# ======================================================================
# ضع أسماء ملفات الداتاسيت هنا
# Place your dataset filenames here
# ======================================================================
# ======================================================================
CKD_DATASET_FILE = "kidney_disease.csv"                    # ← اسم ملف داتاسيت CKD (مثال: "kidney_disease.csv")
DIABETIC_NEPHROPATHY_FILE = "Diabetic_Nephropathy_v1.xlsx"           # ← اسم ملف داتاسيت Diabetic Nephropathy (مثال: "dn_data.xlsx")
DIABETIC_NEPHROPATHY_2_FILE = "diabetic_nephropathy2_dataset.csv"      # ← Dataset with HbA1c, eGFR, UACR
DIABETES_PREDICTION_FILE = "diabetes_prediction_dataset.csv"            # ← اسم ملف داتاسيت Diabetes Prediction (مثال: "diabetes_prediction_dataset.csv")
# ======================================================================


class DataLoader:
    """Load, merge, and preprocess kidney disease datasets."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_names: list = []
        
    def load_ckd_dataset(self) -> pd.DataFrame:
        """Load the Chronic Kidney Disease dataset."""
        if CKD_DATASET_FILE:
            filepath = self.data_dir / CKD_DATASET_FILE
            if filepath.exists():
                if filepath.suffix == '.xlsx':
                    df = pd.read_excel(filepath)
                else:
                    df = pd.read_csv(filepath)
                print(f"[OK] Loaded CKD dataset from {filepath}")
                return df
            else:
                raise FileNotFoundError(f"CKD dataset file not found: {filepath}")
        
        raise FileNotFoundError(
            "[ERROR] CKD dataset filename is empty!\n"
            "   Please set CKD_DATASET_FILE in data_loader.py\n"
            "   Example: CKD_DATASET_FILE = \"kidney_disease.csv\""
        )
    
    def load_diabetic_nephropathy_dataset(self) -> pd.DataFrame:
        """Load the Diabetic Nephropathy dataset."""
        if DIABETIC_NEPHROPATHY_FILE:
            filepath = self.data_dir / DIABETIC_NEPHROPATHY_FILE
            if filepath.exists():
                if filepath.suffix == '.xlsx':
                    df = pd.read_excel(filepath)
                else:
                    df = pd.read_csv(filepath)
                print(f"[OK] Loaded Diabetic Nephropathy dataset from {filepath}")
                return df
            else:
                raise FileNotFoundError(f"Diabetic Nephropathy dataset file not found: {filepath}")
        
        raise FileNotFoundError(
            "   Please set DIABETIC_NEPHROPATHY_FILE in data_loader.py\n"
            "   Example: DIABETIC_NEPHROPATHY_FILE = \"Diabetic_Nephropathy_v1.xlsx\""
        )

    def load_dn2_dataset(self) -> pd.DataFrame:
        """Load the new Diabetic Nephropathy dataset (dataset 2)."""
        if DIABETIC_NEPHROPATHY_2_FILE:
            filepath = self.data_dir / DIABETIC_NEPHROPATHY_2_FILE
            if filepath.exists():
                df = pd.read_csv(filepath)
                print(f"[OK] Loaded Diabetic Nephropathy 2 dataset from {filepath}")
                return df
            else:
                # Optional: don't fail hard if this specific file is missing, just warn
                print(f"[WARN] Diabetic Nephropathy 2 dataset file not found: {filepath}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def merge_datasets(self, df_ckd: pd.DataFrame, df_dn: pd.DataFrame, df_dn2: pd.DataFrame = None) -> pd.DataFrame:
        """
        Merge CKD, Diabetic Nephropathy 1, and Diabetic Nephropathy 2 datasets.
        دمج الداتاسيتات الثلاثة في داتاسيت واحد.
        
        - Maps columns to standard names.
        - Adds a 'source' column.
        - Concatenates DataFrames.
        
        Returns:
            Merged DataFrame
        """
        df_ckd = df_ckd.copy()
        df_dn = df_dn.copy()
        
        # Add source column
        df_ckd['source'] = 'ckd'
        df_dn['source'] = 'diabetic_nephropathy'
        
        # Map columns in df_dn to match training features
        df_dn_mapping = {
            'Age': 'age',
            'Smoking': 'smoking',
            'BMI (kg/m2)': 'bmi',
            'Diabetes duration (y)': 'diabetes_duration',
            'Sex': 'gender',
            'Diabetic nephropathy (DN)': 'classification'
        }
        df_dn = df_dn.rename(columns=df_dn_mapping)
        if 'classification' in df_dn.columns:
            df_dn['classification'] = df_dn['classification'].map({'Yes': 'ckd', 'No': 'notckd', 'yes': 'ckd', 'no': 'notckd', 1: 'ckd', 0: 'notckd'})
        if 'smoking' in df_dn.columns:
            df_dn['smoking'] = df_dn['smoking'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
        # Set default values for diabetic nephropathy dataset (all are diabetic)
        df_dn['dm'] = 1
        
        dfs_to_merge = [df_ckd, df_dn]
        
        if df_dn2 is not None and not df_dn2.empty:
            df_dn2 = df_dn2.copy()
            df_dn2['source'] = 'diabetic_nephropathy_2'
            
            # Map columns in df_dn2 to match training features
            column_mapping = {
                'serum_creatinine': 'sc',
                'BUN': 'bun',
                'blood_pressure_systolic': 'bp',
                'blood_glucose': 'bgr',
                'hypertension': 'htn',
                # New features renames
                'blood_pressure_diastolic': 'bp_dia',
                'albumin': 'serum_albumin',  # Distinguish from urine albumin 'al'
                'diabetes_duration_years': 'diabetes_duration',
                'HbA1c': 'hba1c',
                'eGFR': 'egfr',
                'UACR': 'uacr',
                'BMI': 'bmi',
                'uric_acid': 'uric_acid',
                'smoking': 'smoking',
                'dyslipidemia': 'dyslipidemia',
                'diabetes_type': 'diabetes_type',
                'gender': 'gender',
                'CKD_stage': 'ckd_stage',
                'DN_present': 'dn_present',
                'risk_level': 'risk_level',
                'Calcium': 'cal',
                'Magnesium': 'mag',
                'Sodium': 'sod',
                'Potassium': 'pot'
            }
            df_dn2 = df_dn2.rename(columns=column_mapping)
            
            # Normalize htn (Yes/No -> 1/0)
            if 'htn' in df_dn2.columns:
                df_dn2['htn'] = df_dn2['htn'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
            
            # Normalize dn_present/dm (column was renamed from DN_present -> dn_present)
            if 'dn_present' in df_dn2.columns:
                df_dn2['dm'] = df_dn2['dn_present'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
            else:
                df_dn2['dm'] = 1  # default to 1 since it is diabetic nephropathy dataset
            
            # Normalize CKD_stage to classification
            if 'risk_level' in df_dn2.columns:
                # 'Low' risk represents normal UACR (~17) and normal eGFR (~103), meaning notckd.
                df_dn2['classification'] = df_dn2['risk_level'].map({'Low': 'notckd', 'Moderate': 'ckd', 'High': 'ckd'})
            elif 'ckd_stage' in df_dn2.columns:
                df_dn2['classification'] = df_dn2['ckd_stage'].map({
                    '1': 'notckd', '2': 'notckd', '3a': 'ckd', '3b': 'ckd', '4': 'ckd', '5': 'ckd'
                })
            else:
                df_dn2['classification'] = 'ckd'

            dfs_to_merge.append(df_dn2)
            print(f"[OK] Included Diabetic Nephropathy 2 dataset ({len(df_dn2)} rows)")

        # Find common columns
        print("Merging datasets...")
        merged = pd.concat(dfs_to_merge, axis=0, ignore_index=True)
        
        print(f" Merged dataset shape: {merged.shape}")
        print(f"   - CKD rows: {len(df_ckd)}")
        print(f"   - Diabetic Nephropathy rows: {len(df_dn)}")
        print(f"   - Total rows: {len(merged)}")
        
        return merged
    
    def _clean_raw_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str, list, list]:
        """
        Phase 1: Clean raw data (safe to do before split).
        - Drop leakage/id columns
        - Replace ? with NaN
        - Identify column types
        
        Returns:
            df, target_col, numerical_cols, categorical_cols
        """
        df = df.copy()
        
        # Drop non-feature columns
        from config import CKD_FEATURE_ORDER
        
        # Identify target column
        target_col = None
        for possible_target in ['classification', 'class', 'target', 'label']:
            if possible_target in df.columns:
                target_col = possible_target
                break
        if target_col is None:
            target_col = df.columns[-1]

        # Restrict features strictly to the allowed list + target
        cols_to_keep = set(CKD_FEATURE_ORDER + [target_col])
        cols_to_drop = [c for c in df.columns if c not in cols_to_keep]
        
        print(f"   [INFO] Restricting dataset to exact features: {CKD_FEATURE_ORDER}")
        df = df.drop(columns=cols_to_drop)

        
        # Replace '?' and 'ckd\t' with NaN
        df = df.replace(['?', '\t?', 'ckd\t', 'notckd\t'], 
                        [np.nan, np.nan, 'ckd', 'notckd'])
        


        # Normalize common target strings and drop missing/unknown labels.
        # IMPORTANT: We must NOT allow missing labels to become a separate class like "nan".
        raw_target = df[target_col]
        # Keep NaN as NaN (don't convert to "nan" string yet)
        if pd.api.types.is_object_dtype(raw_target) or pd.api.types.is_string_dtype(raw_target):
            t = raw_target.astype(str).str.strip().str.lower()
            # Map common variants
            t = t.replace(
                {
                    "ckd\t": "ckd",
                    "notckd\t": "notckd",
                    "not ckd": "notckd",
                    "no ckd": "notckd",
                    "normal": "notckd",
                    "none": np.nan,
                    "null": np.nan,
                    "nan": np.nan,
                    "?": np.nan,
                    "unknown": np.nan,
                    "": np.nan,
                }
            )
            # Restore real NaNs where appropriate
            df[target_col] = t

        # Drop rows with missing target labels (cannot be used for supervised training)
        before = len(df)
        df = df.dropna(subset=[target_col])
        dropped = before - len(df)
        if dropped:
            print(f"   [WARN] Dropped {dropped} rows with missing target label in '{target_col}'")
        
        # Encode target variable BEFORE split (safe - it's the label, not a feature)
        le_target = LabelEncoder()
        df[target_col] = le_target.fit_transform(df[target_col].astype(str).str.strip())
        self.label_encoders['target'] = le_target
        
        # Separate feature columns by type
        feature_cols = [c for c in df.columns if c != target_col]
        
        categorical_cols = []
        numerical_cols = []
        
        for col in feature_cols:
            converted = pd.to_numeric(df[col], errors='coerce')
            non_null_ratio = converted.notna().sum() / len(df)
            
            if non_null_ratio > 0.5:
                numerical_cols.append(col)
                df[col] = converted
            else:
                categorical_cols.append(col)
        
        print(f"   Numerical features: {len(numerical_cols)}")
        print(f"   Categorical features: {len(categorical_cols)}")
        print(f"   Target column: {target_col}")
        
        return df, target_col, numerical_cols, categorical_cols
    
    def _preprocess_split(
        self, 
        df_train: pd.DataFrame, 
        df_val: pd.DataFrame, 
        df_test: pd.DataFrame,
        numerical_cols: list, 
        categorical_cols: list,
        feature_engineer: 'FeatureEngineer' = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Phase 2: Preprocess AFTER split.
        
        [ALERT] CRITICAL: All imputers/encoders are FIT on train ONLY,
        then TRANSFORM val and test. This prevents preprocessing leakage.

        Args:
            feature_engineer: A shared FeatureEngineer instance. The first call to
                create_categorical_bins (on df_train) records the dummy column schema;
                subsequent calls (df_val, df_test) reindex to that exact schema.
                If None, a fresh instance is created (fine for single-set inference).
        """
        # ─── Categorical: fill missing with mode from TRAIN only ───
        cat_fill_values = {}
        for col in categorical_cols:
            if col in df_train.columns:
                mode_val = df_train[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else 'unknown'
                cat_fill_values[col] = fill_val
                df_train[col] = df_train[col].fillna(fill_val)
                df_val[col] = df_val[col].fillna(fill_val)
                df_test[col] = df_test[col].fillna(fill_val)
        
        # ─── Numerical: KNN Imputer FIT on train ONLY ───
        if numerical_cols:
            imputer = KNNImputer(n_neighbors=5)
            # FIT on train
            df_train[numerical_cols] = imputer.fit_transform(df_train[numerical_cols])
            # TRANSFORM val and test
            df_val[numerical_cols] = imputer.transform(df_val[numerical_cols])
            df_test[numerical_cols] = imputer.transform(df_test[numerical_cols])
            self._imputer = imputer

        # ─── Computed eGFR feature (NO leakage — pure deterministic math) ───
        # eGFR = f(sc, age) is derived from features already in the training set.
        # The dataset's pre-labeled eGFR column was dropped (that IS leakage).
        # This computed version gives the model the KDIGO boundary (eGFR<60 = CKD)
        # which dramatically reduces false positives on patients with incidentally
        # elevated creatinine (muscle mass, dehydration, controlled conditions).
        # No fitting step: pure math applied identically to all splits.
        for df_set in [df_train, df_val, df_test]:
            if 'sc' in df_set.columns and 'age' in df_set.columns:
                sc = df_set['sc'].astype(float).clip(lower=0.1)
                age = df_set['age'].astype(float).clip(lower=1)
                # CKD-EPI (2009) — male approximation (conservative; slight overestimate for females)
                kappa, alpha = 0.9, -0.411
                sc_k = sc / kappa
                min_cr = sc_k.clip(upper=1.0)
                max_cr = sc_k.clip(lower=1.0)
                egfr_vals = (141.0 * (min_cr ** alpha) * (max_cr ** -1.209) * (0.993 ** age))
                df_set['egfr_computed'] = egfr_vals.clip(lower=5, upper=200).round(1)
        
        # ─── Categorical: LabelEncoder FIT on train ONLY ───
        for col in categorical_cols:
            if col in df_train.columns:
                le = LabelEncoder()
                # Fit on train
                le.fit(df_train[col].astype(str))
                self.label_encoders[col] = le
                # Transform all sets
                df_train[col] = le.transform(df_train[col].astype(str))
                # Handle unseen labels in val/test
                for df_set in [df_val, df_test]:
                    known = set(le.classes_)
                    df_set[col] = df_set[col].astype(str).apply(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df_set[col] = le.transform(df_set[col])
        
        # ─── Feature Engineering: single shared instance (fit/transform pattern) ───
        # [ALERT] ANTI-LEAKAGE: The SAME FeatureEngineer instance is used for all
        # three splits. The first call (df_train) records the dummy column schema;
        # df_val and df_test are reindexed to match — no extra information leaks.
        fe = feature_engineer if feature_engineer is not None else FeatureEngineer()
        df_train = fe.create_categorical_bins(df_train)  # fit  (records schema)
        df_val   = fe.create_categorical_bins(df_val)    # transform (reindex)
        df_test  = fe.create_categorical_bins(df_test)   # transform (reindex)
        
        return df_train, df_val, df_test
    
    def load_and_prepare_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        feature_selection: bool = False,
        use_smote: bool = False,
        ckd_only: bool = False   # True = use ONLY real UCI CKD data (realistic 83-90% acc)
                                  # False = merge all datasets (higher acc but synthetic data)
    ) -> Tuple:
        """
        Complete LEAKAGE-FREE pipeline with anti-overfitting measures:
          1. Load & merge datasets
          2. Clean raw data (safe ops only)
          3. Split into Train/Val/Test (RAW data)
          4. Preprocess each set (fit on train ONLY)
          5. Scale features (fit on train ONLY)
          6. Feature Selection (fit on train ONLY) — reduces noise features
          7. SMOTE (on train ONLY, optional) — handles class imbalance
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, feature_names
        """
        print("=" * 60)
        print("Loading and Merging Datasets...")
        print("=" * 60)
        
        # Load datasets
        df_ckd = self.load_ckd_dataset()

        if ckd_only:
            # ── REALISTIC MODE: UCI CKD only (400 real clinical records) ──
            # Accuracy expected: 83-90%  (real-world generalizable)
            print("[INFO] ckd_only=True -> training on real UCI CKD dataset only (realistic accuracy)")
            df_merged = df_ckd
        else:
            # ── COMBINED MODE: all datasets merged (synthetic data included) ──
            # Accuracy expected: 95-99%  (inflated by synthetic patterns)
            df_dn  = self.load_diabetic_nephropathy_dataset()
            df_dn2 = self.load_dn2_dataset()
            df_merged = self.merge_datasets(df_ckd, df_dn, df_dn2)

        # Phase 1: Clean raw data (safe to do on full dataset)
        print("\nCleaning raw data...")
        df_clean, target_col, numerical_cols, categorical_cols = self._clean_raw_data(df_merged)
        
        # ═══════════════════════════════════════════════════════
        # [ALERT] SPLIT FIRST — before any imputation or encoding!
        # ═══════════════════════════════════════════════════════
        print("\nSplitting data BEFORE preprocessing (anti-leakage)...")
        
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        # Step 1: Split off Test set (20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Step 2: Split remaining into Train (60%) and Val (20%)
        val_fraction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=random_state, stratify=y_temp
        )
        
        print(f"   Raw split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Phase 2: Preprocess AFTER split (fit on train ONLY)
        print("   Preprocessing (fit on train only, transform val/test)...")
        
        # Keep numerical/categorical cols that exist in features
        num_cols = [c for c in numerical_cols if c in X_train.columns]
        cat_cols = [c for c in categorical_cols if c in X_train.columns]

        # Create ONE shared FeatureEngineer so the dummy column schema learned on
        # X_train is reused (not re-fit) for X_val and X_test.
        self.feature_engineer = FeatureEngineer()
        
        X_train, X_val, X_test = self._preprocess_split(
            X_train.copy(), X_val.copy(), X_test.copy(),
            num_cols, cat_cols,
            feature_engineer=self.feature_engineer
        )
        
        # Store feature names (after feature engineering)
        self.feature_names = X_train.columns.tolist()
        
        # Phase 3: Scale features (fit on train ONLY)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # ═══════════════════════════════════════════════════════
        # [ALERT] Phase 4: FEATURE SELECTION — reduces noise features
        #    Fit on TRAIN only, transform all sets
        #    This forces the model to learn from STRONG signals only
        # ═══════════════════════════════════════════════════════
        if feature_selection:
            from sklearn.feature_selection import SelectFromModel
            from sklearn.ensemble import RandomForestClassifier as _RFC
            
            n_features_before = X_train_scaled.shape[1]
            
            # Use a lightweight RF to identify important features
            selector_model = _RFC(
                n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
            )
            self.feature_selector = SelectFromModel(
                selector_model, threshold='median'
            )
            self.feature_selector.fit(X_train_scaled, y_train)
            
            X_train_scaled = self.feature_selector.transform(X_train_scaled)
            X_val_scaled = self.feature_selector.transform(X_val_scaled)
            X_test_scaled = self.feature_selector.transform(X_test_scaled)
            
            # Update feature names to match selected features
            mask = self.feature_selector.get_support()
            self.feature_names = [f for f, m in zip(self.feature_names, mask) if m]
            
            print(f"   [TEST] Feature Selection: {n_features_before} -> {X_train_scaled.shape[1]} features (kept top {X_train_scaled.shape[1]})")
        
        # ═══════════════════════════════════════════════════════
        # [ALERT] Phase 5: SMOTE — synthetic oversampling (TRAIN only)
        #    Generates synthetic minority samples to balance classes
        #    WITHOUT using any real test/val data
        # ═══════════════════════════════════════════════════════
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                n_before = X_train_scaled.shape[0]
                sm = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y_train.astype(int))) - 1))
                X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
                print(f"   [SMOTE] Training samples {n_before} -> {X_train_scaled.shape[0]}")
            except ImportError:
                print("   [WARN]  SMOTE skipped: install imbalanced-learn (`pip install imbalanced-learn`)")
            except Exception as e:
                print(f"   [WARN]  SMOTE failed: {e}")
        
        print(f"\n[OK] Data ready (LEAKAGE-FREE + ANTI-OVERFITTING pipeline)!")
        print(f"   Train set:      {X_train_scaled.shape} ({len(y_train)} samples)")
        print(f"   Validation set: {X_val_scaled.shape} ({len(y_val)} samples)")
        print(f"   Test set:       {X_test_scaled.shape} ({len(y_test)} samples)")
        print(f"   Features:       {len(self.feature_names)}")
        print("=" * 60)
        
        y_train_out = y_train.values if hasattr(y_train, 'values') else y_train
        y_val_out = y_val.values if hasattr(y_val, 'values') else y_val
        y_test_out = y_test.values if hasattr(y_test, 'values') else y_test
        
        # Store full engineered feature names list for alignment at inference time
        self.full_feature_names = X_train.columns.tolist()
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_out, y_val_out, y_test_out, self.feature_names

    def save_preprocessing_artifacts(self, path: Path = Path("models")):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.joblib")
        joblib.dump(self.label_encoders, path / "label_encoders.joblib")
        if hasattr(self, 'feature_selector') and self.feature_selector is not None:
            joblib.dump(self.feature_selector, path / "feature_selector.joblib")
        # Save the FeatureEngineer so its dummy column schema (_dummy_columns) is
        # preserved for inference — ensuring identical column alignment at prediction time.
        if hasattr(self, 'feature_engineer') and self.feature_engineer is not None:
            joblib.dump(self.feature_engineer, path / "feature_engineer.joblib")
        # full_feature_names: pre-selection column list used to align raw inference
        # input before the scaler and feature_selector are applied.
        ff_names = getattr(self, 'full_feature_names', None)
        if ff_names is not None:
            joblib.dump(ff_names, path / "full_feature_names.joblib")
        print("[OK] Saved preprocessing artifacts (scaler, encoders, selector, feature_engineer) successfully!")
        
    def load_preprocessing_artifacts(self, path: Path = Path("models")):
        path = Path(path)
        if (path / "scaler.joblib").exists():
            self.scaler = joblib.load(path / "scaler.joblib")
        if (path / "label_encoders.joblib").exists():
            self.label_encoders = joblib.load(path / "label_encoders.joblib")
        if (path / "feature_selector.joblib").exists():
            self.feature_selector = joblib.load(path / "feature_selector.joblib")
        # Restore the FeatureEngineer with its trained dummy column schema so that
        # inference calls to create_categorical_bins reindex correctly.
        if (path / "feature_engineer.joblib").exists():
            self.feature_engineer = joblib.load(path / "feature_engineer.joblib")
        if (path / "full_feature_names.joblib").exists():
            self.full_feature_names = joblib.load(path / "full_feature_names.joblib")
        print("[OK] Loaded preprocessing artifacts successfully!")


# Convenience function
def load_data(data_dir: str = "data/raw") -> Tuple:
    """Load, merge, and prepare kidney disease data (3-way split, leakage-free)."""
    loader = DataLoader(data_dir)
    return loader.load_and_prepare_data()


if __name__ == "__main__":
    loader = DataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test, features = loader.load_and_prepare_data()
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {len(features)}")
    print(f"Features: {features}")
