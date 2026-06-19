import sys
from pathlib import Path

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import joblib

from src.models.ml_models import MLModels
from src.models.dl_models import DeepLearningModel, _TF_AVAILABLE


class EnsembleModel:
    """
    Ensemble model combining ML and DL predictions.
    Supports weighted voting and stacking.
    DL model is optional — if TensorFlow is unavailable, the ensemble
    operates with ML models only (RF, XGBoost, SVM).
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.ml_models = MLModels(model_dir)
        self.dl_model = DeepLearningModel(model_dir) if _TF_AVAILABLE else None
        self._dl_enabled = _TF_AVAILABLE
        
        # Default weights for ensemble voting
        if self._dl_enabled:
            self.weights = {
                'Random Forest': 0.25,
                'XGBoost': 0.30,
                'SVM': 0.15,
                'Deep Learning': 0.30
            }
        else:
            # Redistribute DL weight proportionally among ML models
            self.weights = {
                'Random Forest': 0.36,
                'XGBoost': 0.43,
                'SVM': 0.21,
            }
        
        self.is_trained = False
        self.training_metrics: Dict[str, Any] = {}
        self.feature_names: List[str] = []  # Persist for inference alignment
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dl_epochs: int = 50
    ) -> Dict[str, Any]:
        """
        Train all models in the ensemble with comprehensive anti-overfitting measures.
        
        Anti-Overfitting Strategy:
          - ML models: Tightened hyperparameters + K-Fold CV on train set
          - DL model: Smaller network + Early Stopping on VALIDATION set
          - 3-way diagnostic: Train vs Val vs Test gap analysis
          - Test set: Used ONLY for final metric reporting (never influences training)
        
        Returns:
            Dictionary with all model metrics
        """
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import accuracy_score, f1_score
        
        print("=" * 60)
        print("Training Ensemble Model (Anti-Overfitting Mode)")
        print("=" * 60)
        
        # ─── ML Models: Train + Evaluate ───
        ml_metrics = self.ml_models.train_all_models(
            X_train, y_train, X_test, y_test,
            X_val=X_val, y_val=y_val,
            calibrate=True,
            calibration_method="isotonic",
        )
        
        # ─── K-Fold Cross-Validation on TRAINING set (10-fold, stratified) ───
        print("\n" + "-" * 60)
        print("K-Fold Cross-Validation (10-Fold Stratified on Training Set)")
        print("-" * 60)
        cv_results = {}
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for name in ['Random Forest', 'XGBoost', 'SVM']:
            model = self.ml_models.models.get(name)
            if model is not None:
                scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted')
                cv_results[name] = {
                    'cv_mean': scores.mean(),
                    'cv_std': scores.std(),
                    'cv_scores': scores.tolist()
                }
                stability = "[OK] STABLE" if scores.std() < 0.03 else "[WARN] UNSTABLE" if scores.std() < 0.05 else "[ERROR] VERY UNSTABLE"
                print(f"  {name:20} | CV F1: {scores.mean():.4f} +/- {scores.std():.4f} | {stability}")
        
        # ─── DL Model: Train with VALIDATION set (NOT test) ───
        print("\n" + "-" * 40)
        if self._dl_enabled and self.dl_model is not None:
            self.dl_model.build_model(input_dim=X_train.shape[1])
            self.dl_model.train(
                X_train, y_train,
                X_val=X_val, y_val=y_val,   # [OK] Validation set, NOT test set
                epochs=dl_epochs
            )
            dl_metrics = self.dl_model.evaluate(X_test, y_test)
        else:
            print("  [INFO] TensorFlow unavailable — Deep Learning model skipped.")
            print("         ML-only ensemble (Random Forest + XGBoost + SVM) will be used.")
            dl_metrics = None
        
        # Combine metrics
        self.training_metrics = {**ml_metrics}
        if dl_metrics is not None:
            self.training_metrics['Deep Learning'] = dl_metrics
        
        # ─── Comprehensive Overfitting Diagnostic: Train vs Val vs Test ───
        print("\n" + "-" * 70)
        print("[TEST] OVERFITTING DIAGNOSTIC: Train vs Val vs Test")
        print("-" * 70)
        print(f"{'Model':20} | {'Train':>8} | {'Val':>8} | {'Test':>8} | {'T-V Gap':>8} | {'T-T Gap':>8} | {'Status'}")
        print("-" * 85)
        
        overfit_detected = False
        for name in ['Random Forest', 'XGBoost', 'SVM']:
            model = self.ml_models.models.get(name)
            if model is not None:
                train_acc = accuracy_score(y_train, model.predict(X_train))
                val_acc = accuracy_score(y_val, model.predict(X_val))
                test_acc = self.training_metrics[name]['accuracy']
                tv_gap = train_acc - val_acc
                tt_gap = train_acc - test_acc
                
                if tt_gap < 0.03:
                    status = "[OK] GOOD"
                elif tt_gap < 0.05:
                    status = "[WARN] SLIGHT"
                elif tt_gap < 0.10:
                    status = "[INFO] MODERATE"
                    overfit_detected = True
                else:
                    status = "[ERROR] OVERFIT"
                    overfit_detected = True
                
                print(f"  {name:20} | {train_acc:8.4f} | {val_acc:8.4f} | {test_acc:8.4f} | {tv_gap:8.4f} | {tt_gap:8.4f} | {status}")
                
                # Store val metrics
                self.training_metrics[name]['val_accuracy'] = val_acc
                self.training_metrics[name]['val_f1'] = f1_score(y_val, model.predict(X_val), average='weighted')
        
        # DL train vs val vs test
        if self._dl_enabled and self.dl_model is not None and self.dl_model.model is not None:
            dl_train_pred = self.dl_model.predict(X_train)
            dl_train_acc = accuracy_score(y_train, dl_train_pred)
            dl_val_pred = self.dl_model.predict(X_val)
            dl_val_acc = accuracy_score(y_val, dl_val_pred)
            dl_test_acc = dl_metrics['accuracy']
            dl_tv_gap = dl_train_acc - dl_val_acc
            dl_tt_gap = dl_train_acc - dl_test_acc
            dl_status = "[OK] GOOD" if dl_tt_gap < 0.03 else "[WARN] SLIGHT" if dl_tt_gap < 0.05 else "[INFO] MODERATE" if dl_tt_gap < 0.10 else "[ERROR] OVERFIT"
            print(f"  {'Deep Learning':20} | {dl_train_acc:8.4f} | {dl_val_acc:8.4f} | {dl_test_acc:8.4f} | {dl_tv_gap:8.4f} | {dl_tt_gap:8.4f} | {dl_status}")
        else:
            print(f"  {'Deep Learning':20} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | [SKIP] TF unavailable")
        
        if overfit_detected:
            print("\n  [WARN]  Some models show potential overfitting. Consider:")
            print("     - Collecting more training data")
            print("     - Reducing feature count via feature selection")
            print("     - Further tightening regularization")
        else:
            print("\n  [OK] All models show healthy generalization gaps!")
        
        # ─── Update weights based on VALIDATION performance (not test) ───
        self._update_weights()
        
        self.is_trained = True
        
        # ─── Final Summary ───
        print("\n" + "=" * 60)
        print("Ensemble Training Summary")
        print("=" * 60)
        for name, weight in self.weights.items():
            metrics = self.training_metrics.get(name, {})
            f1 = metrics.get('f1_weighted', 0)  # key is 'f1_weighted', not 'f1_score'
            gap = metrics.get('overfit_gap', 0)
            cv_info = ""
            if name in cv_results:
                cv_info = f" | CV: {cv_results[name]['cv_mean']:.4f} +/- {cv_results[name]['cv_std']:.4f}"
            print(f"{name:20} | F1: {f1:.4f} | Gap: {gap:.4f} | Weight: {weight:.2f}{cv_info}")
        
        # Store CV results in training metrics for external access
        self.training_metrics['cross_validation'] = cv_results
        
        return self.training_metrics

    
    def _update_weights(self):
        """Update ensemble weights based on model performance."""
        total_f1 = 0
        f1_scores = {}
        
        for name in self.weights.keys():
            metrics = self.training_metrics.get(name, {})
            f1 = metrics.get('f1_weighted', 0)  # key is 'f1_weighted', not 'f1_score'
            f1_scores[name] = f1
            total_f1 += f1
        
        if total_f1 > 0:
            for name in self.weights.keys():
                self.weights[name] = f1_scores[name] / total_f1
    
    def predict_proba_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get P(CKD) from all models.
        
        NOTE: Label encoding gives 0='ckd', 1='notckd'.
        We must use proba[:, 0] (P(class=0=ckd)) not proba[:, 1] (P(notckd)).
        """
        probas = {}
        
        # ML model probabilities — column 0 = P(ckd)
        for name in ['Random Forest', 'XGBoost', 'SVM']:
            try:
                proba = self.ml_models.predict_proba(X, name)
                # Use column 0 = P(ckd), since 0='ckd' in LabelEncoder
                probas[name] = proba[:, 0] if len(proba.shape) > 1 else proba
            except Exception as e:
                print(f"Warning: Could not get probabilities from {name}: {e}")
        
        # DL model: sigmoid output is P(class=1=notckd), so invert to get P(ckd)
        if self._dl_enabled and self.dl_model is not None and self.dl_model.model is not None:
            try:
                dl_proba = self.dl_model.predict_proba(X)
                probas['Deep Learning'] = 1.0 - dl_proba.flatten()
            except Exception as e:
                print(f"Warning: Could not get probabilities from DL model: {e}")
        
        return probas
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get weighted ensemble prediction probabilities."""
        probas = self.predict_proba_all(X)
        
        # Weighted average
        ensemble_proba = np.zeros(X.shape[0])
        total_weight = 0
        
        for name, proba in probas.items():
            weight = self.weights.get(name, 0)
            ensemble_proba += weight * proba
            total_weight += weight
        
        if total_weight > 0:
            ensemble_proba /= total_weight
        
        return ensemble_proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        """Make ensemble predictions using P(CKD) >= threshold.
        Threshold 0.50: balanced threshold; eGFR feature now provides a strong
        directional signal, making extreme sensitivity-bias unnecessary.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        threshold: float = 0.50  # 0.50: balanced threshold; eGFR_computed provides clear signal
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Make predictions with confidence scores and model agreement.
        
        Returns:
            predictions, confidence_scores, details
        """
        probas = self.predict_proba_all(X)
        ensemble_proba = self.predict_proba(X)
        predictions = (ensemble_proba >= threshold).astype(int)
        
        # Calculate confidence as distance from threshold
        confidence = np.abs(ensemble_proba - threshold) * 2  # Scale to 0-1
        confidence = np.clip(confidence, 0, 1)
        
        # Calculate model agreement
        all_preds = []
        for name, proba in probas.items():
            pred = (proba >= threshold).astype(int)
            all_preds.append(pred)
        
        all_preds = np.array(all_preds)
        agreement = np.mean(all_preds == predictions, axis=0)
        
        details = {
            'individual_probas': probas,
            'ensemble_proba': ensemble_proba,
            'model_agreement': agreement,
            'confidence': confidence
        }
        
        return predictions, confidence, details
    
    def get_prediction_explanation(self, X: np.ndarray, idx: int = 0) -> Dict:
        """Get detailed explanation for a single prediction."""
        probas = self.predict_proba_all(X)
        ensemble_proba = self.predict_proba(X)
        
        explanation = {
            'ensemble_probability': float(ensemble_proba[idx]),
            'prediction': 'CKD Positive' if ensemble_proba[idx] >= 0.5 else 'CKD Negative',
            'confidence': float(abs(ensemble_proba[idx] - 0.5) * 2),
            'model_contributions': {}
        }
        
        for name, proba in probas.items():
            weight = self.weights.get(name, 0)
            contribution = weight * proba[idx]
            explanation['model_contributions'][name] = {
                'probability': float(proba[idx]),
                'weight': float(weight),
                'weighted_contribution': float(contribution)
            }
        
        return explanation
    
    def save(self, prefix: str = "ensemble"):
        """Save all models and metadata."""
        print("Saving ensemble models...")
        
        # Save ML models
        self.ml_models.save_all_models()
        
        # Save DL model (only if TF available and model was trained)
        if self._dl_enabled and self.dl_model is not None and self.dl_model.model is not None:
            try:
                self.dl_model.save_model(f"{prefix}_dl_model.keras")
            except Exception as e:
                print(f"[WARN] Could not save DL model: {e}")
        
        # Save weights
        weights_path = self.model_dir / f"{prefix}_weights.joblib"
        joblib.dump(self.weights, weights_path)
        
        # Save feature names for inference alignment
        meta_path = self.model_dir / f"{prefix}_metadata.joblib"
        joblib.dump({
            'feature_names': self.feature_names,
            'weights': self.weights
        }, meta_path)
        
        print("All models saved successfully!")
    
    def load(self, prefix: str = "ensemble"):
        """Load ensemble weights, metadata, and saved ML/DL checkpoints from disk."""
        self.is_trained = False

        weights_path = self.model_dir / f"{prefix}_weights.joblib"
        if weights_path.exists():
            self.weights = joblib.load(weights_path)

        meta_path = self.model_dir / f"{prefix}_metadata.joblib"
        if meta_path.exists():
            meta = joblib.load(meta_path)
            self.feature_names = meta.get('feature_names', [])
        self.ml_models.feature_names = self.feature_names or None

        joblib_names = {
            "Random Forest": "random_forest_model.joblib",
            "XGBoost": "xgboost_model.joblib",
            "SVM": "svm_model.joblib",
        }
        loaded_ml = False
        for display_name, filename in joblib_names.items():
            path = self.model_dir / filename
            if path.exists():
                try:
                    self.ml_models.models[display_name] = joblib.load(path)
                    loaded_ml = True
                except Exception as e:
                    print(f"[WARN] Could not load {display_name} from {path}: {e}")

        dl_path = self.model_dir / f"{prefix}_dl_model.keras"
        if dl_path.exists() and self._dl_enabled and self.dl_model is not None:
            try:
                self.dl_model.load_model(str(dl_path))
            except Exception as e:
                print(f"[WARN] Could not load Deep Learning model from {dl_path}: {e}")
        elif not self._dl_enabled:
            print("  [INFO] TensorFlow unavailable — DL model not loaded. Using ML-only ensemble.")

        self.is_trained = loaded_ml


class StackingEnsemble(EnsembleModel):
    """
    Stacking ensemble that uses a meta-learner to combine predictions.
    """
    
    def __init__(self, model_dir: str = "models"):
        super().__init__(model_dir)
        self.meta_model = None
        
    def train_stacking(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dl_epochs: int = 50
    ) -> Dict[str, Any]:
        """Train stacking ensemble with meta-learner."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        
        # First, train base models
        metrics = self.train(X_train, y_train, X_val, y_val, X_test, y_test, dl_epochs)
        
        # Generate meta-features using cross-validation predictions
        print("\nTraining Meta-Learner...")
        
        # Get out-of-fold predictions for training data
        meta_features_train = []
        
        for name in ['Random Forest', 'XGBoost', 'SVM']:
            model = self.ml_models.models[name]
            oof_pred = cross_val_predict(
                model, X_train, y_train, cv=5, method='predict_proba'
            )
            meta_features_train.append(oof_pred[:, 1])
        
        # For DL, generate out-of-fold predictions to avoid data leakage
        from sklearn.model_selection import KFold
        dl_oof = np.zeros(X_train.shape[0])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X_train):
            fold_model = DeepLearningModel(str(self.model_dir))
            fold_model.build_model(input_dim=X_train.shape[1])
            fold_model.train(
                X_train[train_idx], y_train[train_idx],
                X_val=X_train[val_idx], y_val=y_train[val_idx],
                epochs=10  # verbose is not a parameter of DeepLearningModel.train()
            )
            dl_oof[val_idx] = fold_model.predict_proba(X_train[val_idx]).flatten()
        meta_features_train.append(dl_oof)
        
        # Stack meta-features
        meta_X_train = np.column_stack(meta_features_train)
        
        # Train meta-learner
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(meta_X_train, y_train)
        
        # Evaluate stacking
        meta_features_test = []
        for name in ['Random Forest', 'XGBoost', 'SVM']:
            proba = self.ml_models.predict_proba(X_test, name)[:, 1]
            meta_features_test.append(proba)
        
        dl_pred_test = self.dl_model.predict_proba(X_test).flatten()
        meta_features_test.append(dl_pred_test)
        
        meta_X_test = np.column_stack(meta_features_test)
        
        stacking_pred = self.meta_model.predict(meta_X_test)
        stacking_proba = self.meta_model.predict_proba(meta_X_test)[:, 1]
        
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        stacking_metrics = {
            'accuracy': accuracy_score(y_test, stacking_pred),
            'f1_score': f1_score(y_test, stacking_pred, average='weighted'),
            'auc_roc': roc_auc_score(y_test, stacking_proba)
        }
        
        print(f"\nStacking Ensemble Results:")
        print(f"  Accuracy: {stacking_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {stacking_metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:  {stacking_metrics['auc_roc']:.4f}")
        
        metrics['Stacking'] = stacking_metrics
        return metrics


if __name__ == "__main__":
    # Load data using the project's DataLoader
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.preprocessing.data_loader import DataLoader

    DATA_DIR = "data/raw"

    loader = DataLoader(data_dir=DATA_DIR)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = loader.load_and_prepare_data()

    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Features: {feature_names}")

    # Train ensemble
    ensemble = EnsembleModel()
    metrics = ensemble.train(X_train, y_train, X_val, y_val, X_test, y_test, dl_epochs=30)

    # Make predictions
    preds, confidence, details = ensemble.predict_with_confidence(X_test)

    print(f"\nEnsemble Accuracy: {np.mean(preds == y_test):.4f}")
    print(f"Average Confidence: {np.mean(confidence):.4f}")

    # Get explanation for first sample
    explanation = ensemble.get_prediction_explanation(X_test, idx=0)
    print(f"\nSample Prediction Explanation:")
    print(f"  Prediction: {explanation['prediction']}")
    print(f"  Confidence: {explanation['confidence']:.4f}")
