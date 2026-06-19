"""
SHAP Explainer Module
Provides Explainable AI (XAI) using SHAP values for kidney disease predictions.

يوفر تفسيرات واضحة لقرارات النموذج باستخدام SHAP:
- أي Features أثرت على القرار أكتر
- Feature Importance عامة للنموذج
- تفسير كل حالة مريض بشكل فردي
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    Explainable AI using SHAP (SHapley Additive exPlanations).
    
    Supports:
        - TreeExplainer for XGBoost / Random Forest (fast)
        - KernelExplainer as fallback for any model
        - Global feature importance ranking
        - Local (per-prediction) explanations
        - Human-readable explanation reports
    """
    
    def __init__(self):
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self._shap_imported = False
        self._shap = None
    
    def _import_shap(self):
        """Lazy import of shap to avoid slow startup."""
        if not self._shap_imported:
            try:
                import shap
                self._shap = shap
                self._shap_imported = True
            except ImportError:
                raise ImportError(
                    "SHAP is required for explainability. "
                    "Install it with: pip install shap"
                )
        return self._shap
    
    def fit(self, model, X_background: np.ndarray, model_type: str = "tree"):
        """
        Fit the SHAP explainer to a trained model.
        
        Args:
            model: Trained ML model (XGBoost, RandomForest, SVM, etc.)
            X_background: Background dataset for SHAP (use training data or a sample)
            model_type: "tree" for tree-based models, "kernel" for others
        """
        shap = self._import_shap()
        
        if model_type == "tree":
            try:
                self.explainer = shap.TreeExplainer(model)
                print("   [OK] SHAP TreeExplainer initialized")
            except Exception:
                # Fallback to KernelExplainer
                print("   [WARN] TreeExplainer failed, using KernelExplainer (slower)")
                # Use a small sample for KernelExplainer background
                if len(X_background) > 100:
                    indices = np.random.choice(len(X_background), 100, replace=False)
                    X_background = X_background[indices]
                self.explainer = shap.KernelExplainer(model.predict_proba, X_background)
        else:
            if len(X_background) > 100:
                indices = np.random.choice(len(X_background), 100, replace=False)
                X_background = X_background[indices]
            self.explainer = shap.KernelExplainer(model.predict_proba, X_background)
            print("   [OK] SHAP KernelExplainer initialized")
    
    def explain_prediction(
        self,
        X: np.ndarray,
        feature_names: List[str],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Explain a single prediction (local explanation).
        
        Args:
            X: Feature vector (1, n_features) or (n_features,)
            feature_names: List of feature names
            top_k: Number of top contributing features to return
            
        Returns:
            Dictionary with:
                - shap_values: Raw SHAP values for all features
                - top_positive: Features pushing toward CKD (risk factors)
                - top_negative: Features pushing away from CKD (protective factors)
                - expected_value: Base prediction value
                - explanation_text: Human-readable explanation
        """
        if self.explainer is None:
            return {"error": "Explainer not fitted. Call fit() first."}
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Multi-class: use class 1 (CKD positive)
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                sv = shap_values[0, :, 1]  # (samples, features, classes) -> class 1
            elif shap_values.ndim == 2:
                sv = shap_values[0]
            else:
                sv = shap_values
        else:
            # shap Explanation object
            try:
                sv = shap_values.values[0]
                if sv.ndim > 1:
                    sv = sv[:, 1]  # class 1
            except Exception:
                sv = np.array(shap_values)[0]
        
        # Get expected value
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        
        # Create feature-importance pairs
        feature_shap = list(zip(feature_names, sv, X[0]))
        
        # Sort by absolute SHAP value
        feature_shap_sorted = sorted(feature_shap, key=lambda x: abs(x[1]), reverse=True)
        
        # Separate positive (risk) and negative (protective) contributions
        top_positive = [
            {"feature": name, "shap_value": float(val), "feature_value": float(fv)}
            for name, val, fv in feature_shap_sorted if val > 0
        ][:top_k]
        
        top_negative = [
            {"feature": name, "shap_value": float(val), "feature_value": float(fv)}
            for name, val, fv in feature_shap_sorted if val < 0
        ][:top_k]
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            top_positive, top_negative, expected_value
        )
        
        return {
            "shap_values": {name: float(val) for name, val, _ in feature_shap},
            "top_risk_factors": top_positive,
            "top_protective_factors": top_negative,
            "top_features": [
                {"feature": name, "shap_value": float(val), "feature_value": float(fv)}
                for name, val, fv in feature_shap_sorted[:top_k]
            ],
            "expected_value": float(expected_value),
            "explanation_text": explanation_text
        }
    
    def global_feature_importance(
        self,
        X: np.ndarray,
        feature_names: List[str],
        max_samples: int = 500
    ) -> Dict[str, Any]:
        """
        Calculate global feature importance using mean |SHAP| values.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            max_samples: Maximum samples to use (for speed)
            
        Returns:
            Dictionary with ranked feature importances
        """
        if self.explainer is None:
            return {"error": "Explainer not fitted. Call fit() first."}
        
        # Subsample if too large
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Calculate SHAP values for all samples
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle formats
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            try:
                sv = shap_values.values
                if sv.ndim > 2:
                    sv = sv[:, :, 1]
            except Exception:
                sv = np.array(shap_values)
        
        # Store for later use
        self.shap_values = sv
        
        # Mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        
        # Create ranked list
        importance_pairs = list(zip(feature_names, mean_abs_shap))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "feature_importance": [
                {"feature": name, "importance": float(imp), "rank": i + 1}
                for i, (name, imp) in enumerate(importance_pairs)
            ],
            "total_features": len(feature_names),
            "samples_used": len(X_sample)
        }
    
    def get_explanation_report(
        self,
        X: np.ndarray,
        feature_names: List[str],
        patient_info: Dict[str, Any] = None
    ) -> str:
        """
        Generate a complete human-readable explanation report for a prediction.
        
        Args:
            X: Feature vector for a single patient
            feature_names: List of feature names
            patient_info: Optional patient metadata (name, age, etc.)
            
        Returns:
            Formatted explanation report string
        """
        explanation = self.explain_prediction(X, feature_names)
        
        if "error" in explanation:
            return f"Error: {explanation['error']}"
        
        report = []
        report.append("=" * 60)
        report.append("  AI Prediction Explanation Report")
        report.append("  تقرير تفسير التنبؤ بالذكاء الاصطناعي")
        report.append("=" * 60)
        
        if patient_info:
            report.append(f"\nPatient: {patient_info.get('name', 'N/A')}")
            report.append(f"Age: {patient_info.get('age', 'N/A')}")
        
        report.append(f"\nBase Risk Score: {explanation['expected_value']:.4f}")
        
        report.append("\n--- Top Risk Factors (عوامل الخطر) ---")
        for i, factor in enumerate(explanation['top_risk_factors'][:5], 1):
            report.append(
                f"  {i}. {factor['feature']}: "
                f"value={factor['feature_value']:.3f}, "
                f"impact=+{factor['shap_value']:.4f}"
            )
        
        report.append("\n--- Protective Factors (العوامل الحامية) ---")
        for i, factor in enumerate(explanation['top_protective_factors'][:5], 1):
            report.append(
                f"  {i}. {factor['feature']}: "
                f"value={factor['feature_value']:.3f}, "
                f"impact={factor['shap_value']:.4f}"
            )
        
        report.append("\n" + explanation['explanation_text'])
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _generate_explanation_text(
        self,
        top_positive: List[Dict],
        top_negative: List[Dict],
        expected_value: float
    ) -> str:
        """Generate human-readable explanation text."""
        
        # Feature name to Arabic/English description mapping
        feature_descriptions = {
            'sc': 'Serum Creatinine (الكرياتينين)',
            'creatinine': 'Serum Creatinine (الكرياتينين)',
            'egfr': 'eGFR (معدل الترشيح)',
            'hba1c': 'HbA1c (السكر التراكمي)',
            'uacr': 'UACR (نسبة الألبومين/الكرياتينين)',
            'bu': 'Blood Urea (يوريا الدم)',
            'uric_acid': 'Uric Acid (حمض اليوريك)',
            'bmi': 'BMI (مؤشر كتلة الجسم)',
            'age': 'Age (العمر)',
            'bp': 'Blood Pressure (ضغط الدم)',
            'hemo': 'Hemoglobin (الهيموجلوبين)',
            'al': 'Albumin in Urine (الألبومين في البول)',
            'su': 'Sugar in Urine (السكر في البول)',
            'bgr': 'Blood Glucose (جلوكوز الدم)',
            'sod': 'Sodium (الصوديوم)',
            'pot': 'Potassium (البوتاسيوم)',
            'dm': 'Diabetes (السكري)',
            'htn': 'Hypertension (ارتفاع ضغط الدم)',
            'smoking': 'Smoking (التدخين)',
            'diabetes_duration': 'Diabetes Duration (مدة السكري)',
            # Binned feature categories (One-Hot Encoded)
            'age_cat_adult': 'Age Category: Adult (18-40)',
            'age_cat_middle_aged': 'Age Category: Middle-Aged (40-60)',
            'age_cat_elderly': 'Age Category: Elderly (60+)',
            'bp_cat_normal': 'BP Category: Normal (90-120)',
            'bp_cat_elevated': 'BP Category: Elevated (120-140)',
            'bp_cat_high': 'BP Category: High (140+)',
            'sc_cat_normal': 'Creatinine Category: Normal (0.5-1.2)',
            'sc_cat_high': 'Creatinine Category: High (1.2-5.0)',
            'sc_cat_critical': 'Creatinine Category: Critical (5.0+)',
            'hemo_cat_anemia': 'Hemoglobin Category: Anemia (8-12)',
            'hemo_cat_normal': 'Hemoglobin Category: Normal (12-17)',
            'hemo_cat_high': 'Hemoglobin Category: High (17+)',
            'bgr_cat_normal': 'Glucose Category: Normal (70-100)',
            'bgr_cat_prediabetes': 'Glucose Category: Pre-Diabetes (100-125)',
            'bgr_cat_diabetes': 'Glucose Category: Diabetes (125+)',
            'bu_cat_normal': 'Blood Urea Category: Normal (7-20)',
            'bu_cat_high': 'Blood Urea Category: High (20-50)',
            'bu_cat_critical': 'Blood Urea Category: Critical (50+)',
        }
        
        lines = []
        lines.append("Summary / الملخص:")
        
        if top_positive:
            main_factor = top_positive[0]
            fname = feature_descriptions.get(
                main_factor['feature'], main_factor['feature']
            )
            lines.append(
                f"  The most significant risk factor is {fname} "
                f"(value: {main_factor['feature_value']:.2f})."
            )
        
        if len(top_positive) > 1:
            other_names = [
                feature_descriptions.get(f['feature'], f['feature'])
                for f in top_positive[1:3]
            ]
            lines.append(f"  Other contributing factors: {', '.join(other_names)}.")
        
        if top_negative:
            protective_names = [
                feature_descriptions.get(f['feature'], f['feature'])
                for f in top_negative[:2]
            ]
            lines.append(
                f"  Protective factors: {', '.join(protective_names)}."
            )
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Quick test with dummy data
    print(" Testing SHAP Explainer...")
    
    explainer = SHAPExplainer()
    print("[OK] SHAPExplainer created successfully")
    
    # Full test requires a trained model
    print("ℹ️  Full test requires a trained model. Use with main.py.")
