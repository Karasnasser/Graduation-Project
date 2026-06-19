"""
Machine Learning Models Module
Implements Random Forest, XGBoost, and SVM classifiers for kidney disease prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, balanced_accuracy_score
)
import xgboost as xgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MLModels:
    """
    Machine Learning models for kidney disease classification.
    Supports Random Forest, XGBoost, and SVM.
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, Any] = {}
        self.best_model_name: str = None
        self.best_model: Any = None
        self.training_metrics: Dict[str, Dict] = {}
        self.feature_names: Optional[List[str]] = None
        self.calibrated_model: Any = None
        self.calibration_method: Optional[str] = None
        
    def get_random_forest(self, params: Dict = None) -> RandomForestClassifier:
        """Get Random Forest classifier with STRONG anti-overfitting parameters."""
        default_params = {
            'n_estimators': 200,         # More trees → reduces variance (averaging)
            'max_depth': 5,              # Shallow trees → can't memorize complex patterns
            'min_samples_split': 15,     # High threshold → prevents splitting on noise
            'min_samples_leaf': 8,       # Each leaf needs 8+ samples → no memorization
            'max_features': 0.3,         # Use only 30% of features per split → forces learning from electrolytes
            'max_samples': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
        if params:
            default_params.update(params)
        return RandomForestClassifier(**default_params)
    
    def get_xgboost(self, params: Dict = None, class_ratio: float = None) -> xgb.XGBClassifier:
        """Get XGBoost classifier with STRONG anti-overfitting parameters + Early Stopping.
        
        Args:
            class_ratio: neg/pos ratio for scale_pos_weight. If None, uses 1.0.
                         Pass np.bincount(y)[0]/np.bincount(y)[1] for imbalanced data.
        """
        # Dynamically set scale_pos_weight to match real class imbalance
        spw = float(class_ratio) if class_ratio is not None else 1.0
        default_params = {
            'n_estimators': 300,         # High ceiling — early stopping will pick the right point
            'max_depth': 3,              # Very shallow → prevents memorization
            'learning_rate': 0.03,       # Slow learning → forces gradual, generalizable fitting
            'subsample': 0.6,            # Only 60% of rows per tree → strong stochasticity
            'colsample_bytree': 0.3,     # Only 30% of features per tree (forces AI to use non-sc features)
            'colsample_bylevel': 0.4,    # Only 40% of features per level
            'reg_alpha': 0.3,            # L1 regularization → feature sparsity
            'reg_lambda': 2.0,           # L2 regularization → weight shrinkage
            'min_child_weight': 5,       # Higher → blocks splits on small groups
            'gamma': 0.3,                # Minimum loss reduction → prunes weak splits
            'random_state': 42,
            'eval_metric': 'logloss',
            'scale_pos_weight': spw      # Correct class imbalance ratio (neg/pos)
        }
        if params:
            default_params.update(params)
        return xgb.XGBClassifier(**default_params)
    
    def get_svm(self, params: Dict = None) -> SVC:
        """Get SVM classifier with STRONG anti-overfitting parameters."""
        default_params = {
            'kernel': 'rbf',
            'C': 0.3,                    # Very soft margin → prioritizes generalization
            'gamma': 'scale',
            'probability': True,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        if params:
            default_params.update(params)
        return SVC(**default_params)
    
    def train_model(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Train a single model and evaluate it.
        Also calculates TRAIN metrics for overfitting gap analysis.
        
        [ALERT] XGBoost uses EARLY STOPPING on validation set to prevent overfitting.
        The model stops training when validation performance stops improving.
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Compute class-balanced sample weights (handles severe imbalance better than accuracy)
        classes = np.unique(y_train)
        class_weights = None
        sample_weight = None
        try:
            cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
            class_weights = {int(c): float(w) for c, w in zip(classes, cw)}
            sample_weight = np.asarray([class_weights[int(y)] for y in y_train], dtype=float)
        except Exception:
            class_weights = None
            sample_weight = None

        # Ensure XGBoost uses the right objective in multi-class setups
        try:
            if isinstance(model, xgb.XGBClassifier):
                if len(classes) > 2:
                    model.set_params(
                        objective="multi:softprob",
                        num_class=int(len(classes)),
                        eval_metric="mlogloss",
                    )
                else:
                    model.set_params(objective="binary:logistic", eval_metric="logloss")
        except Exception:
            pass

        # ═══════════════════════════════════════════════════════
        # [ALERT] EARLY STOPPING for XGBoost — stops when val loss
        #    stops improving, preventing the model from memorizing
        # ═══════════════════════════════════════════════════════
        try:
            if isinstance(model, xgb.XGBClassifier) and X_val is not None and y_val is not None:
                # XGBoost with early stopping on validation set
                fit_params = {
                    'eval_set': [(X_val, y_val)],
                    'verbose': False
                }
                if sample_weight is not None:
                    fit_params['sample_weight'] = sample_weight
                model.set_params(early_stopping_rounds=15)
                model.fit(X_train, y_train, **fit_params)
                actual_rounds = model.best_iteration + 1 if hasattr(model, 'best_iteration') else model.n_estimators
                print(f"    [Early Stopping] used {actual_rounds}/{model.n_estimators} rounds")
                # [ALERT] Reset early stopping so it doesn't break subsequent cross_val_score calls
                model.set_params(early_stopping_rounds=None)
            elif sample_weight is not None:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train)
        except TypeError:
            # Some estimators may not accept sample_weight in fit()
            model.fit(X_train, y_train)
        
        # Predict on TEST set
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
                if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                    # Keep full matrix for multi-class AUC; for binary we'll use column 1 later.
                    y_proba = proba
            except Exception:
                y_proba = None
        
        # Predict on TRAIN set (for overfitting detection)
        y_train_pred = model.predict(X_train)
        
        # Calculate TEST metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        }
        
        # Calculate TRAIN metrics (for gap analysis)
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        metrics['train_f1_weighted'] = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        metrics['overfit_gap'] = metrics['train_accuracy'] - metrics['accuracy']
        if class_weights is not None:
            metrics['class_weights'] = class_weights
        
        # ROC-AUC (binary or multi-class)
        if y_proba is not None:
            try:
                y_unique = np.unique(y_test)
                if len(y_unique) < 2:
                    # AUC is undefined if only one class exists in y_test
                    metrics['auc_roc_error'] = f"AUC undefined: y_test has a single class ({y_unique.tolist()})"
                else:
                    proba_arr = np.asarray(y_proba, dtype=float)
                    if not np.all(np.isfinite(proba_arr)):
                        metrics['auc_roc_error'] = "AUC undefined: y_proba contains NaN/Inf"
                    else:
                        # Binary vs multi-class
                        if proba_arr.shape[1] == 2 and len(y_unique) == 2:
                            metrics['auc_roc'] = roc_auc_score(y_test, proba_arr[:, 1])
                        else:
                            # Multi-class: compute OVR AUC (more common than OVO for reporting)
                            metrics['auc_roc'] = roc_auc_score(
                                y_test,
                                proba_arr,
                                multi_class="ovr",
                                average="weighted",
                            )
            except Exception as e:
                metrics['auc_roc_error'] = f"{type(e).__name__}: {e}"
        
        # Store model and metrics
        self.models[model_name] = model
        self.training_metrics[model_name] = metrics
        
        return metrics
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        calibrate: bool = True,
        calibration_method: str = "isotonic",
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all models and compare performance.
        
        Returns:
            Dictionary mapping model names to their metrics
        """
        # Compute class imbalance ratio for XGBoost (neg / pos count)
        counts = np.bincount(y_train.astype(int))
        class_ratio = float(counts[0]) / float(counts[1]) if len(counts) >= 2 and counts[1] > 0 else 1.0
        print(f"\n[INFO] Class distribution: {dict(enumerate(counts.tolist()))} | XGBoost scale_pos_weight = {class_ratio:.1f}")

        models_to_train = [
            ('Random Forest', self.get_random_forest()),
            ('XGBoost', self.get_xgboost(class_ratio=class_ratio)),
            ('SVM', self.get_svm())
        ]
        
        all_metrics = {}
        best_f1 = 0
        
        print("Training Machine Learning Models...")
        print("=" * 50)
        
        for name, model in models_to_train:
            print(f"\nTraining {name}...")
            metrics = self.train_model(model, X_train, y_train, X_test, y_test, name, X_val=X_val, y_val=y_val)
            all_metrics[name] = metrics
            
            print(f"  Train Acc: {metrics['train_accuracy']:.4f} | Test Acc: {metrics['accuracy']:.4f} | Gap: {metrics['overfit_gap']:.4f}")
            print(f"  Precision: {metrics['precision_weighted']:.4f} (weighted) | {metrics['precision_macro']:.4f} (macro)")
            print(f"  Recall:    {metrics['recall_weighted']:.4f} (weighted) | {metrics['recall_macro']:.4f} (macro)")
            print(f"  F1 Score:  {metrics['f1_weighted']:.4f} (weighted) | {metrics['f1_macro']:.4f} (macro)")
            if 'auc_roc' in metrics:
                print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
            
            # Overfitting warning
            if metrics['overfit_gap'] > 0.05:
                print(f"  [WARN]  WARNING: Overfit gap = {metrics['overfit_gap']:.4f} (Train >> Test)")
            elif metrics['overfit_gap'] < 0.03:
                print(f"  [OK] Good generalization (gap < 3%)")
            
            # Track best model
            if metrics['f1_weighted'] > best_f1:
                best_f1 = metrics['f1_weighted']
                self.best_model_name = name
                self.best_model = model
        
        print("\n" + "=" * 50)
        print(f"Best Model: {self.best_model_name} (F1: {best_f1:.4f})")

        # Optional probability calibration on a held-out VALIDATION set (never test)
        self.calibrated_model = None
        self.calibration_method = None
        if calibrate and self.best_model is not None and X_val is not None and y_val is not None:
            try:
                if hasattr(self.best_model, "predict_proba"):
                    # NOTE: cv="prefit" is used intentionally to calibrate on X_val/y_val only.
                    calibrator = CalibratedClassifierCV(
                        estimator=self.best_model,
                        method=calibration_method,
                        cv="prefit",
                    )
                    calibrator.fit(X_val, y_val)
                    self.calibrated_model = calibrator
                    self.calibration_method = calibration_method
                    print(f"[OK] Calibrated best model probabilities using {calibration_method} on validation set")
            except Exception as e:
                print(f"[WARN] Calibration failed: {type(e).__name__}: {e}")
        
        return all_metrics
    
    def hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str = 'XGBoost'
    ) -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV.

        [ALERT] ANTI-LEAKAGE: X_train MUST be the training split only.
        GridSearchCV creates internal k-fold splits from whatever data is passed.
        Passing val or test rows here would let hyperparameter selection see
        held-out data, inflating reported performance (a form of leakage).
        Always call this with the same X_train used in train_all_models().

        Returns:
            Best estimator after tuning
        """
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }
        
        if model_name not in param_grids:
            raise ValueError(f"Unknown model: {model_name}")
        
        if model_name == 'Random Forest':
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_name == 'XGBoost':
            base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        else:
            base_model = SVC(random_state=42, probability=True)
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        grid_search = GridSearchCV(
            base_model,
            param_grids[model_name],
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Returns:
            Dictionary with mean and std of scores
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        return {
            'mean_f1': scores.mean(),
            'std_f1': scores.std(),
            'scores': scores.tolist()
        }
    
    def get_feature_importance(
        self,
        model_name: str = None,
        feature_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Get feature importance from trained models.
        
        Returns:
            DataFrame with feature importances sorted descending
        """
        if model_name is None:
            model_name = self.best_model_name
            
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            return pd.DataFrame()
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        return df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    def save_model(self, model_name: str = None, filename: str = None):
        """Save a trained model to disk."""
        if model_name is None:
            model_name = self.best_model_name
            
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        if filename is None:
            filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
        
        filepath = self.model_dir / filename
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str) -> Any:
        """Load a trained model from disk."""
        model = joblib.load(filepath)
        return model
    
    def save_all_models(self):
        """Save all trained models."""
        for name in self.models:
            self.save_model(name)
    
    def predict(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """Make predictions using a trained model."""
        if model_name is None:
            model_name = self.best_model_name
            
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        return model.predict(X)
    
    def predict_proba(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """Get prediction probabilities (uses calibrated best model if available)."""
        if model_name is None:
            model_name = self.best_model_name

        if self.calibrated_model is not None and model_name == self.best_model_name:
            return self.calibrated_model.predict_proba(X)
            
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            raise ValueError(f"Model {model_name} does not support probability predictions")
    
    def get_classification_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = None
    ) -> str:
        """Get detailed classification report."""
        y_pred = self.predict(X_test, model_name)
        return classification_report(y_test, y_pred)
    
    def get_confusion_matrix(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = None
    ) -> np.ndarray:
        """Get confusion matrix."""
        y_pred = self.predict(X_test, model_name)
        return confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
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

    # Train models
    ml = MLModels()
    metrics = ml.train_all_models(X_train, y_train, X_test, y_test)

    # Get classification report
    print("\nClassification Report (Best Model):")
    print(ml.get_classification_report(X_test, y_test))
