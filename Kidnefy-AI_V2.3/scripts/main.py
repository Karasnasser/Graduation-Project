"""
Kidney Disease Prediction System
Main entry point for training and using the AI model.
"""

import sys
from pathlib import Path

# Add src and config to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Import modules
from src.preprocessing import DataLoader, FeatureEngineer, calculate_egfr
from src.models import EnsembleModel, MLModels, DeepLearningModel
from src.staging import GFRCalculator, RiskAssessor
from src.reports import PDFReportGenerator, PatientInfo, TestResult
from src.explainability import SHAPExplainer
from src.monitoring import LongitudinalMonitor
from config import CKD_FEATURE_ORDER, CKD_FEATURE_DEFAULTS


class KidneyDiseasePredictionSystem:
    """
    Complete system for kidney disease prediction.
    Combines ML/DL models, staging, and report generation.
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.ensemble_model = EnsembleModel(str(self.model_dir))
        self.gfr_calculator = GFRCalculator()
        self.risk_assessor = RiskAssessor()
        self.report_generator = PDFReportGenerator()
        
        # XAI - SHAP Explainer
        self.shap_explainer = SHAPExplainer()
        
        # Longitudinal Monitoring
        self.longitudinal_monitor = LongitudinalMonitor()
        
        # Store feature names and training data reference for SHAP
        self._feature_names = None
        self._X_train_sample = None
        
        self.is_trained = False
    

    
    def train(self, epochs: int = 50, ckd_only: bool = False):
        """Train all models on the dataset.
        
        Args:
            epochs: Training epochs for Deep Learning model
            ckd_only: If True, train on real UCI CKD data only (83-90% acc).
                      If False, merge all datasets including synthetic (95-99% acc).
        """
        print("=" * 60)
        print("Kidney Disease Prediction System - Training")
        print(f"   Mode: {'UCI CKD only (realistic)' if ckd_only else 'All datasets (combined)'}")
        print("=" * 60)
        
        # Load and prepare data (3-way split: Train/Val/Test)
        # use_smote=True balances the CKD/non-CKD class imbalance on training data only
        print("\n Loading and preprocessing data...")
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
            self.data_loader.load_and_prepare_data(ckd_only=ckd_only, use_smote=True)

        print(f"   Training samples:   {X_train.shape[0]}")
        print(f"   Validation samples: {X_val.shape[0]}")
        print(f"   Test samples:       {X_test.shape[0]}")
        print(f"   Features:           {len(feature_names)}")
        
        # Train ensemble (Val set for DL Early Stopping, Test for final eval only)
        print("\n Training models...")
        metrics = self.ensemble_model.train(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            dl_epochs=epochs
        )
        
        # Initialize SHAP Explainer with XGBoost model
        print("\n Initializing SHAP Explainer...")
        try:
            xgb_model = self.ensemble_model.ml_models.models.get('XGBoost')
            if xgb_model:
                self.shap_explainer.fit(xgb_model, X_train, model_type="tree")
            self._feature_names = feature_names
            self._X_train_sample = X_train[:200]  # Store sample for SHAP background
        except Exception as e:
            print(f"   [WARN] SHAP initialization failed: {e}")
        
        # Save models
        print("\n Saving models...")
        self.ensemble_model.feature_names = feature_names  # Persist for inference
        self.ensemble_model.ml_models.feature_names = feature_names
        self.ensemble_model.save()
        self.data_loader.save_preprocessing_artifacts(self.model_dir)
        
        self.is_trained = True
        
        print("\n[OK] Training complete!")
        return metrics
    
    def predict_from_features(
        self,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Make prediction from feature dictionary.
        
        Args:
            features: Dictionary mapping feature names to values
                Required: 'sc' (creatinine), optional: age, egfr, acr, etc.
        
        Returns:
            Complete prediction result
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Ensure preprocessing artifacts are loaded for inference
        if not hasattr(self, '_preprocessing_loaded') or not self._preprocessing_loaded:
            try:
                self.data_loader.load_preprocessing_artifacts(self.model_dir)
                self._preprocessing_loaded = True
            except Exception as e:
                print(f"   [WARN] Could not load preprocessing artifacts: {e}")
        
        # Extract key values
        creatinine = features.get('sc', features.get('creatinine', 1.0))
        age = features.get('age', 50)
        is_female = features.get('is_female', False)
        acr = features.get('acr', None)
        
        # Calculate eGFR for clinical assessment (NOT added to feature dict —
        # egfr was dropped as a leakage column during training and must stay out
        # of the model input to preserve the same feature space).
        egfr = features.get('egfr')
        if egfr is None:
            egfr = calculate_egfr(creatinine, age, is_female)

        # Compute egfr_computed: the same CKD-EPI formula applied in _preprocess_split
        # during training. This is derived from sc+age (both already in the feature set),
        # so it carries no extra information the model wasn't trained with.
        sc_val = max(creatinine, 0.1)
        age_val = max(age, 1)
        kappa, alpha = 0.9, -0.411
        sc_k = sc_val / kappa
        min_cr = min(sc_k, 1.0)
        max_cr = max(sc_k, 1.0)
        egfr_computed = float(
            141.0 * (min_cr ** alpha) * (max_cr ** -1.209) * (0.993 ** age_val)
        )
        egfr_computed = round(max(5.0, min(200.0, egfr_computed)), 1)

        # Prepare feature vector matching the training features
        expected_features = CKD_FEATURE_ORDER
        defaults = CKD_FEATURE_DEFAULTS

        # Inject egfr_computed so the loop below picks it up from the features dict
        # (mirrors what _preprocess_split does during training)
        features = dict(features)  # make a copy to avoid mutating the caller's dict
        features['egfr_computed'] = egfr_computed

        feature_dict = {}
        for f in expected_features:
            # Handle potential aliases in input features dict
            val = features.get(f)
            if val is None and f == "uacr":
                val = features.get("acr")
            if val is None:
                # Aliases mapping
                aliases = {
                    'bp_dia': 'blood_pressure_diastolic',
                    'serum_albumin': 'albumin', # if user sends 'albumin' for serum
                    'hba1c': 'HbA1c'
                }
                for alias in aliases:
                    if f == alias and aliases[alias] in features:
                        val = features[aliases[alias]]
                        break
            
            # Use default if still None
            if val is None:
                val = defaults.get(f, 0)
            
            feature_dict[f] = [val] # Wrap in list for DataFrame

        # Convert to DataFrame
        df_features = pd.DataFrame(feature_dict)
        
        # Apply exactly the same Medical Feature Categorization as training.
        # [ALERT] ANTI-LEAKAGE: Use the SAVED feature_engineer (loaded from disk)
        # so the dummy column schema matches the one learned during training exactly.
        # A fresh FeatureEngineer() would produce an unfit schema and misalign columns.
        if hasattr(self.data_loader, 'feature_engineer') and self.data_loader.feature_engineer is not None:
            df_features = self.data_loader.feature_engineer.create_categorical_bins(df_features)
        else:
            from src.preprocessing import FeatureEngineer
            df_features = FeatureEngineer().create_categorical_bins(df_features)
        
        # Ensure the feature space perfectly matches the full engineered feature space of training
        if hasattr(self.data_loader, 'full_feature_names') and self.data_loader.full_feature_names:
            full_features = self.data_loader.full_feature_names
            # Add missing dummy columns with 0
            for col in full_features:
                if col not in df_features.columns:
                    df_features[col] = 0
            # Keep only these columns in this exact order
            df_features = df_features[full_features]
            
        # Convert to float32 NumPy array
        feature_array = df_features.values.astype(np.float32)
        
        # Scale features
        if hasattr(self.data_loader, 'scaler') and hasattr(self.data_loader.scaler, 'mean_'):
            feature_scaled = self.data_loader.scaler.transform(feature_array)
        else:
            feature_scaled = feature_array
            
        # Apply Feature Selection if selector exists
        if hasattr(self.data_loader, 'feature_selector') and hasattr(self.data_loader.feature_selector, 'estimator_'):
            feature_vector = self.data_loader.feature_selector.transform(feature_scaled)
        else:
            # Fallback to model's expected features if selector is missing
            if hasattr(self.ensemble_model.ml_models, 'feature_names') and self.ensemble_model.ml_models.feature_names:
                trained_features = self.ensemble_model.ml_models.feature_names
                # Add missing dummy columns with 0
                for col in trained_features:
                    if col not in df_features.columns:
                        df_features[col] = 0
                df_features_aligned = df_features[trained_features]
                feature_vector = df_features_aligned.values.astype(np.float32)
            else:
                feature_vector = feature_scaled
        
        # Get ensemble prediction
        pred, confidence, details = self.ensemble_model.predict_with_confidence(feature_vector)
        probability = details['ensemble_proba'][0]
        
        # Get complete assessment with enhanced biomarkers
        other_values = {
            'hba1c': features.get('hba1c', 5.5),
            'uric_acid': features.get('uric_acid', 5.0),
            'bmi': features.get('bmi', 25.0),
            'smoking': features.get('smoking', 0),
            'diabetes_duration': features.get('diabetes_duration', 0),
        }
        assessment = self.risk_assessor.complete_assessment(
            ckd_probability=probability,
            creatinine=creatinine,
            egfr=egfr,
            acr=acr,
            age=age,
            is_female=is_female,
            other_values=other_values
        )
        
        result = {
            'prediction': bool(pred[0]),
            'probability': float(probability),
            'confidence': float(confidence[0]),
            'egfr': egfr,
            'gfr_stage': assessment.gfr_stage.value,
            'albuminuria_category': assessment.albuminuria_category.value if assessment.albuminuria_category else None,
            'risk_level': assessment.risk_level.value,
            'progression_risk': assessment.progression_risk.risk_percentage,
            'enhanced_risk_score': assessment.enhanced_risk_score,
            'recommendations': assessment.recommendations,
            'alerts': assessment.alerts
        }
        
        # Add SHAP explanation if available
        if self.shap_explainer.explainer is not None and self._feature_names:
            try:
                explanation = self.shap_explainer.explain_prediction(
                    feature_vector, self._feature_names
                )
                result['xai_explanation'] = {
                    'top_risk_factors': explanation.get('top_risk_factors', [])[:5],
                    'top_protective_factors': explanation.get('top_protective_factors', [])[:5],
                    'explanation_text': explanation.get('explanation_text', '')
                }
            except Exception as e:
                result['xai_explanation'] = {'error': str(e)}
        
        return result
    
    def add_patient_measurement(
        self,
        patient_id: str,
        date: str,
        egfr: float,
        creatinine: float = None,
        uacr: float = None,
        hba1c: float = None,
    ) -> Dict[str, Any]:
        """
        Add a longitudinal measurement for a patient.
        
        Args:
            patient_id: Unique patient identifier
            date: Date (YYYY-MM-DD)
            egfr: eGFR value
            creatinine: Serum creatinine
            uacr: UACR value
            hba1c: HbA1c percentage
        """
        return self.longitudinal_monitor.add_measurement(
            patient_id=patient_id,
            date=date,
            egfr=egfr,
            creatinine=creatinine,
            uacr=uacr,
            hba1c=hba1c
        )
    
    def get_patient_trend(self, patient_id: str) -> Dict[str, Any]:
        """
        Get longitudinal trend analysis for a patient.
        Includes fast progressor detection.
        """
        from dataclasses import asdict
        trend = self.longitudinal_monitor.calculate_trend(patient_id)
        return asdict(trend)
    

    
    def generate_report(
        self,
        result: Dict[str, Any],
        patient_name: str = "Patient",
        output_path: str = None
    ) -> str:
        """
        Generate PDF report from prediction result.
        
        Returns:
            Path to generated PDF
        """
        # Create patient info
        patient = PatientInfo(
            name=patient_name,
            age=result.get('patient_info', {}).get('age', 50),
            sex=result.get('patient_info', {}).get('sex', 'Unknown'),
            date=result.get('patient_info', {}).get('date', 'Today'),
            lab_no=result.get('patient_info', {}).get('lab_no', '')
        )
        
        # Create lab results
        lab_results = []
        extracted = result.get('extracted_values', {})
        for name, data in extracted.items():
            if isinstance(data, dict) and 'value' in data:
                lab_results.append(TestResult(
                    name=name.replace('_', ' ').title(),
                    value=data['value'],
                    unit=data.get('unit', ''),
                    reference_range=data.get('reference_range', ''),
                    is_abnormal=data.get('is_abnormal', False)
                ))
        
        # Generate report
        filepath = self.report_generator.generate_report(
            patient=patient,
            prediction=result.get('prediction', True),
            probability=result.get('probability', 0.5),
            risk_level=result.get('risk_level', 'Unknown'),
            gfr_stage=result.get('gfr_stage', 'Unknown'),
            egfr=result.get('egfr', 0),
            alb_category=result.get('albuminuria_category'),
            acr=extracted.get('acr', {}).get('value'),
            lab_results=lab_results,
            recommendations=result.get('recommendations', []),
            alerts=result.get('alerts', []),
            filename=output_path
        )
        
        return filepath


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Kidney Disease Prediction System'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    train_parser.add_argument(
        '--merge-synthetic',
        action='store_true',
        default=False,
        help=(
            'Merge all datasets including synthetic data '
            '(expected accuracy: 95-99% but higher risk of overfitting). '
            'Omit this flag to train on real clinical UCI CKD data only '
            '(realistic 83-90% accuracy).'
        )
    )

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make prediction')
    predict_parser.add_argument('--creatinine', type=float, required=True, help='Serum creatinine value')
    predict_parser.add_argument('--age', type=int, default=50, help='Patient age')
    predict_parser.add_argument('--acr', type=float, help='ACR value')
    predict_parser.add_argument('--female', action='store_true', help='Is patient female')
    
    # Stage command
    stage_parser = subparsers.add_parser('stage', help='Calculate kidney stage')
    stage_parser.add_argument('--creatinine', type=float, required=True)
    stage_parser.add_argument('--age', type=int, required=True)
    stage_parser.add_argument('--acr', type=float)
    stage_parser.add_argument('--female', action='store_true')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        system = KidneyDiseasePredictionSystem()
        system.train(epochs=args.epochs, ckd_only=not args.merge_synthetic)

    elif args.command == 'predict':
        system = KidneyDiseasePredictionSystem()
        # Load trained models
        try:
            system.ensemble_model.load()
            system.is_trained = system.ensemble_model.is_trained
        except Exception as e:
            print(f"Error: Could not load trained models: {e}")
            print("Please train the model first: python main.py train")
            return
        
        features = {
            'sc': args.creatinine,
            'age': args.age,
            'acr': args.acr,
            'is_female': args.female
        }
        try:
            result = system.predict_from_features(features)
            print("\nPrediction Result:")
            print(f"  GFR Stage: {result['gfr_stage']}")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  CKD Probability: {result['probability']:.4f}")
            print(f"  Progression Risk: {result['progression_risk']}%")
            if result.get('recommendations'):
                print("\nRecommendations:")
                for r in result['recommendations'][:3]:
                    print(f"  - {r}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            
    elif args.command == 'stage':
        calculator = GFRCalculator()
        result = calculator.calculate_stage(
            creatinine=args.creatinine,
            acr=args.acr,
            age=args.age,
            is_female=args.female
        )
        print(calculator.format_result(result))
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
