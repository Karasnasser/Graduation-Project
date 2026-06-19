"""
Diabetes Preprocessing Algorithm
خوارزمية تحضير وتحليل داتاسيت التنبؤ بالسكري

This module preprocesses the Diabetes Prediction Dataset and performs
medical classification based on HbA1c and blood glucose levels.

Output: X, y ready for model.fit(X, y) + summary report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# ======================================================================
# ضع اسم ملف داتاسيت السكري هنا
# Place your Diabetes Prediction Dataset filename here
# ======================================================================
DIABETES_PREDICTION_FILE = "diabetes_prediction_dataset.csv"  # ← تم التحديد
# ======================================================================


class DiabetesPreprocessor:
    """
    Preprocessing algorithm for the Diabetes Prediction Dataset.
    خوارزمية تحضير وتحليل بيانات التنبؤ بالسكري.

    Pipeline:
        1. Load dataset
        2. Handle missing values (median for numerical, mode for categorical)
        3. Remove duplicate rows
        4. Handle outliers using IQR
        5. Medical classification (HbA1c + Blood Glucose)
        6. Encode categorical columns (LabelEncoder)
        7. Scale numerical features (StandardScaler)
        8. Return X, y + summary report

    Usage:
        preprocessor = DiabetesPreprocessor()
        X, y, report = preprocessor.run()
        # X, y are ready for model.fit(X, y)
    """

    # ======================================================================
    # Medical Classification Thresholds (المعايير الطبية)
    # ======================================================================
    # HbA1c Thresholds
    HBAC1_NORMAL_MAX = 5.7        # أقل من 5.7% = طبيعي
    HBAC1_PREDIABETIC_MAX = 6.4   # من 5.7% لـ 6.4% = ما قبل السكري
    # أكبر من 6.5% = مريض سكري

    # Blood Glucose Thresholds (mg/dL)
    GLUCOSE_NORMAL_MAX = 100      # أقل من 100 = طبيعي
    GLUCOSE_PREDIABETIC_MAX = 125 # من 100 لـ 125 = ما قبل السكري
    # أكبر من 126 = مريض سكري

    # Dataset columns
    CATEGORICAL_COLS = ['gender', 'smoking_history']
    NUMERICAL_COLS = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    OUTLIER_COLS = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    TARGET_COL = 'diabetes'

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.report: Dict[str, Any] = {}

    # ==================================================================
    # Step 1: Load Dataset
    # ==================================================================
    def load_dataset(self) -> pd.DataFrame:
        """
        Load the Diabetes Prediction Dataset.
        تحميل داتاسيت التنبؤ بالسكري.
        """
        if not DIABETES_PREDICTION_FILE:
            raise FileNotFoundError(
                "❌ Diabetes dataset filename is empty!\n"
                "   Please set DIABETES_PREDICTION_FILE in diabetes_preprocessing.py\n"
                '   Example: DIABETES_PREDICTION_FILE = "diabetes_prediction_dataset.csv"'
            )

        filepath = self.data_dir / DIABETES_PREDICTION_FILE
        if not filepath.exists():
            raise FileNotFoundError(f"❌ Dataset file not found: {filepath}")

        if filepath.suffix == '.xlsx':
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)

        print(f"[OK] Loaded Diabetes dataset from {filepath}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")

        self.report['original_shape'] = df.shape
        return df

    # ==================================================================
    # Step 2: Handle Missing Values
    # ==================================================================
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values.
        معالجة القيم المفقودة — median للأرقام، mode للنصوص.
        """
        df = df.copy()
        missing_before = df.isnull().sum().sum()

        # Numerical columns: fill with median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"   Filled {col} with median: {median_val:.2f}")

        # Categorical columns: fill with mode
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().any():
                mode_val = df[col].mode().iloc[0]
                df[col] = df[col].fillna(mode_val)
                print(f"   Filled {col} with mode: {mode_val}")

        missing_after = df.isnull().sum().sum()
        print(f"   Missing values: {missing_before} -> {missing_after}")

        self.report['missing_before'] = int(missing_before)
        self.report['missing_after'] = int(missing_after)
        return df

    # ==================================================================
    # Step 3: Remove Duplicates
    # ==================================================================
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows.
        حذف الصفوف المكررة.
        """
        df = df.copy()
        rows_before = len(df)
        df = df.drop_duplicates()
        rows_after = len(df)
        removed = rows_before - rows_after

        print(f"   Duplicates removed: {removed}")
        print(f"   Rows: {rows_before} -> {rows_after}")

        self.report['duplicates_removed'] = removed
        return df.reset_index(drop=True)

    # ==================================================================
    # Step 4: Handle Outliers (IQR Method)
    # ==================================================================
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers using IQR (Interquartile Range) method.
        معالجة القيم الشاذة باستخدام IQR — القيم خارج النطاق تُستبدل بحدود النطاق.
        """
        df = df.copy()
        outliers_total = 0

        for col in self.OUTLIER_COLS:
            if col not in df.columns:
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Count outliers
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            outliers_total += outlier_count

            # Cap outliers (clip to bounds instead of removing rows)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

            if outlier_count > 0:
                print(f"   {col}: {outlier_count} outliers capped "
                      f"[{lower_bound:.2f} — {upper_bound:.2f}]")

        print(f"   Total outliers handled: {outliers_total}")
        self.report['outliers_handled'] = outliers_total
        return df

    # ==================================================================
    # Step 5: Medical Classification (التصنيف الطبي)
    # ==================================================================
    def classify_medical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Medical classification based on HbA1c and Blood Glucose levels.
        تصنيف طبي بناءً على مستويات HbA1c والجلوكوز في الدم.

        HbA1c Classification:
            < 5.7%  → Normal (طبيعي)
            5.7–6.4% → Pre-diabetic (ما قبل السكري)
            ≥ 6.5%  → Diabetic (مريض سكري)

        Blood Glucose Classification:
            < 100 mg/dL   → Normal (طبيعي)
            100–125 mg/dL → Pre-diabetic (ما قبل السكري)
            ≥ 126 mg/dL   → Diabetic (مريض سكري)
        """
        df = df.copy()

        # --- HbA1c Classification ---
        if 'HbA1c_level' in df.columns:
            df['hba1c_category'] = pd.cut(
                df['HbA1c_level'],
                bins=[-np.inf, self.HBAC1_NORMAL_MAX, self.HBAC1_PREDIABETIC_MAX, np.inf],
                labels=['Normal', 'Pre-diabetic', 'Diabetic']
            )
            print("   [OK] HbA1c classification added")
        else:
            print("   [WARN] HbA1c_level column not found, skipping HbA1c classification")

        # --- Blood Glucose Classification ---
        if 'blood_glucose_level' in df.columns:
            df['glucose_category'] = pd.cut(
                df['blood_glucose_level'],
                bins=[-np.inf, self.GLUCOSE_NORMAL_MAX, self.GLUCOSE_PREDIABETIC_MAX, np.inf],
                labels=['Normal', 'Pre-diabetic', 'Diabetic']
            )
            print("   [OK] Blood Glucose classification added")
        else:
            print("   [WARN] blood_glucose_level column not found, skipping glucose classification")

        # --- Combined Medical Risk Class ---
        if 'hba1c_category' in df.columns and 'glucose_category' in df.columns:
            risk_map = {'Normal': 0, 'Pre-diabetic': 1, 'Diabetic': 2}

            hba1c_risk = df['hba1c_category'].map(risk_map).fillna(0).astype(int)
            glucose_risk = df['glucose_category'].map(risk_map).fillna(0).astype(int)

            # Combined risk = max of both (worst-case)
            combined_risk = np.maximum(hba1c_risk, glucose_risk)
            reverse_map = {0: 'Normal', 1: 'Pre-diabetic', 2: 'Diabetic'}
            df['medical_risk_class'] = combined_risk.map(reverse_map)

            print("   [OK] Combined medical risk class added")

        return df

    # ==================================================================
    # Step 6: Encode Categorical Columns
    # ==================================================================
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns using LabelEncoder.
        تحويل الأعمدة النصية إلى أرقام باستخدام Label Encoding.
        """
        df = df.copy()

        # Columns to encode: original categoricals + new medical classifications
        cols_to_encode = self.CATEGORICAL_COLS + [
            'hba1c_category', 'glucose_category', 'medical_risk_class'
        ]

        for col in cols_to_encode:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"   Encoded {col}: {list(le.classes_)}")

        return df

    # ==================================================================
    # Step 7: Scale Numerical Features
    # ==================================================================
    def scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Scale all features using StandardScaler.
        تطبيع الأعمدة الرقمية باستخدام StandardScaler.
        """
        X_scaled = self.scaler.fit_transform(X)
        print(f"   [OK] Features scaled with StandardScaler ({X.shape[1]} features)")
        return X_scaled

    # ==================================================================
    # Step 8: Generate Summary Report
    # ==================================================================
    def generate_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary report showing the distribution of medical classifications.
        إنشاء تقرير مختصر بتوزيع الحالات الطبيعية وما قبل السكري والمرضى.
        """
        report = self.report.copy()

        # HbA1c Distribution
        if 'hba1c_category' in df.columns:
            hba1c_dist = df['hba1c_category'].value_counts()
            report['hba1c_distribution'] = hba1c_dist.to_dict()

        # Glucose Distribution
        if 'glucose_category' in df.columns:
            glucose_dist = df['glucose_category'].value_counts()
            report['glucose_distribution'] = glucose_dist.to_dict()

        # Combined Risk Distribution
        if 'medical_risk_class' in df.columns:
            risk_dist = df['medical_risk_class'].value_counts()
            report['medical_risk_distribution'] = risk_dist.to_dict()

        # Target Distribution
        if self.TARGET_COL in df.columns:
            target_dist = df[self.TARGET_COL].value_counts()
            report['target_distribution'] = target_dist.to_dict()

        report['final_shape'] = df.shape

        return report

    def print_report(self, report: Dict[str, Any]):
        """Print the summary report in a readable format."""
        print("\n" + "=" * 60)
        print(" PREPROCESSING SUMMARY REPORT")
        print("   تقرير ملخص المعالجة")
        print("=" * 60)

        print(f"\n Original shape:  {report.get('original_shape', 'N/A')}")
        print(f" Final shape:     {report.get('final_shape', 'N/A')}")
        print(f" Missing values handled: {report.get('missing_before', 0)}")
        print(f" Duplicates removed:     {report.get('duplicates_removed', 0)}")
        print(f" Outliers handled:       {report.get('outliers_handled', 0)}")

        # HbA1c Distribution
        if 'hba1c_distribution' in report:
            print(f"\n HbA1c Classification Distribution:")
            for category, count in report['hba1c_distribution'].items():
                print(f"   {category}: {count}")

        # Glucose Distribution
        if 'glucose_distribution' in report:
            print(f"\n Blood Glucose Classification Distribution:")
            for category, count in report['glucose_distribution'].items():
                print(f"   {category}: {count}")

        # Combined Risk Distribution
        if 'medical_risk_distribution' in report:
            print(f"\n[MEDICAL] Combined Medical Risk Distribution:")
            for category, count in report['medical_risk_distribution'].items():
                total = sum(report['medical_risk_distribution'].values())
                pct = (count / total * 100) if total > 0 else 0
                print(f"   {category}: {count} ({pct:.1f}%)")

        # Target Distribution
        if 'target_distribution' in report:
            print(f"\n Target (diabetes) Distribution:")
            for label, count in report['target_distribution'].items():
                total = sum(report['target_distribution'].values())
                pct = (count / total * 100) if total > 0 else 0
                label_name = "Diabetic" if label == 1 else "Not Diabetic"
                print(f"   {label} ({label_name}): {count} ({pct:.1f}%)")

        print("\n" + "=" * 60)

    # ==================================================================
    # Main Pipeline: run()
    # ==================================================================
    def run(self, save_artifacts: bool = False, artifacts_dir: str = "models/diabetes/artifacts") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Run the complete preprocessing pipeline.
        تشغيل خط الأنابيب الكامل للمعالجة.

        Args:
            save_artifacts: Whether to save encoders/scalers for inference.
            artifacts_dir: Directory to save artifacts in.

        Returns:
            X: np.ndarray — Features scaled and ready for training
            y: np.ndarray — Target variable
            report: dict — Summary report with distributions

        Usage:
            preprocessor = DiabetesPreprocessor()
            X, y, report = preprocessor.run(save_artifacts=True)
            model.fit(X, y)  # Ready!
        """
        print("=" * 60)
        print(" Diabetes Preprocessing Algorithm")
        print("   خوارزمية تحضير بيانات السكري")
        print("=" * 60)

        # Step 1: Load
        print("\n Step 1: Loading dataset...")
        df = self.load_dataset()

        # Step 2: Missing Values
        print("\n Step 2: Handling missing values...")
        df = self.handle_missing_values(df)

        # Step 3: Duplicates
        print("\n Step 3: Removing duplicates...")
        df = self.remove_duplicates(df)

        # Step 4: Outliers
        print("\n Step 4: Handling outliers (IQR)...")
        df = self.handle_outliers(df)

        # Step 5: Medical Classification
        print("\n[MEDICAL] Step 5: Medical classification (HbA1c + Glucose)...")
        df = self.classify_medical(df)

        # Generate report BEFORE encoding (to get readable labels)
        report = self.generate_report(df)

        # Step 6: Encode
        print("\n Step 6: Encoding categorical columns...")
        df = self.encode_categorical(df)

        # Separate X and y
        if self.TARGET_COL not in df.columns:
            raise ValueError(
                f"❌ Target column '{self.TARGET_COL}' not found!\n"
                f"   Available columns: {list(df.columns)}"
            )

        y = df[self.TARGET_COL].values
        X_df = df.drop(columns=[self.TARGET_COL])

        # Store feature names
        self.feature_names = X_df.columns.tolist()
        report['feature_names'] = self.feature_names

        # Step 7: Scale
        print("\n Step 7: Scaling features...")
        X = self.scale_features(X_df)
        
        # Save artifacts if requested
        if save_artifacts:
            print("\n Saving artifacts for inference...")
            self.save_artifacts(artifacts_dir)

        # Step 8: Report
        self.print_report(report)

        print(f"\n[OK] OUTPUT READY:")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Features: {self.feature_names}")
        print(f"\n   Ready for: model.fit(X, y)")
        print("=" * 60)

        return X, y, report

    # ==================================================================
    # Backend Integration Methods
    # ==================================================================
    def save_artifacts(self, artifacts_dir: str = "models/diabetes/artifacts"):
        """
        Save preprocessing artifacts (encoders, scalers) for inference.
        حفظ أدوات المعالجة لاستخدامها لاحقاً.
        """
        path = Path(artifacts_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save Scaler
        joblib.dump(self.scaler, path / "scaler.pkl")
        
        # Save Label Encoders
        joblib.dump(self.label_encoders, path / "label_encoders.pkl")
        
        # Save feature names (crucial for ensuring correct order)
        if hasattr(self, 'feature_names'):
             joblib.dump(self.feature_names, path / "feature_names.pkl")
             
        print(f"[OK] Preprocessing artifacts saved to {path}")

    def load_artifacts(self, artifacts_dir: str = "models/diabetes/artifacts"):
        """
        Load preprocessing artifacts for inference.
        تحميل أدوات المعالجة.
        """
        path = Path(artifacts_dir)
        if not path.exists():
            raise FileNotFoundError(f"❌ Artifacts directory not found: {path}")
            
        self.scaler = joblib.load(path / "scaler.pkl")
        self.label_encoders = joblib.load(path / "label_encoders.pkl")
        if (path / "feature_names.pkl").exists():
            self.feature_names = joblib.load(path / "feature_names.pkl")
            
        print(f"[OK] Loaded artifacts from {path}")

    def preprocess_single(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess a single instance for prediction.
        معالجة حالة واحدة للتنبؤ الفوري.
        
        Args:
            input_data: Dictionary containing patient data (e.g. {'age': 50, 'glucose': 100...})
            
        Returns:
            np.ndarray: Preprocessed feature vector (1, n_features)
        """
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # 1. Medical Classification
        df = self.classify_medical(df)
        
        # 2. Encode Categoricals
        # Note: We use the SAVED encoders, handle unseen labels if necessary
        cols_to_encode = self.CATEGORICAL_COLS + [
            'hba1c_category', 'glucose_category', 'medical_risk_class'
        ]
        
        for col in cols_to_encode:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unknown labels by assigning a default or raising error
                # Here we strictly map, assuming backend validates input
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError:
                    # Fallback for unseen labels: encode as -1 or 0 (risky but keeps running)
                    print(f"[WARN] Warning: Unseen label in {col}, setting to -1")
                    df[col] = -1
        
        # 3. Ensure correct column order
        if hasattr(self, 'feature_names'):
            # filling missing columns with 0 if any
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        
        # 4. Scale
        X_scaled = self.scaler.transform(df)
        
        return X_scaled


if __name__ == "__main__":
    # ====================================================================
    # تأكد إنك حطيت اسم الملف في المتغير أعلى الملف
    # Make sure you set the filename in DIABETES_PREDICTION_FILE above
    # ====================================================================
    preprocessor = DiabetesPreprocessor()
    X, y, report = preprocessor.run()
