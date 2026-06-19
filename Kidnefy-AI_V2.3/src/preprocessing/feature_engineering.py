"""
Feature Engineering Module
Advanced feature creation and selection for kidney disease prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Advanced feature engineering for kidney disease data."""
    
    def __init__(self):
        self.selected_features: List[str] = []
        self.pca = None
        self.feature_importance: Dict[str, float] = {}
        # Stores the full dummy-column schema learned from the training set.
        # All subsequent transforms (val/test/inference) are reindexed to this.
        self._dummy_columns: Optional[List[str]] = None
        
    def calculate_egfr(
        self,
        creatinine: float,
        age: int,
        is_female: bool = False,
        is_black: bool = False
    ) -> float:
        """
        Calculate estimated Glomerular Filtration Rate (eGFR) using CKD-EPI formula.
        
        Args:
            creatinine: Serum creatinine in mg/dL
            age: Patient age in years
            is_female: True if patient is female
            is_black: True if patient is African American
            
        Returns:
            eGFR value in mL/min/1.73m²
        """
        if creatinine <= 0 or age <= 0:
            return 0.0
            
        # CKD-EPI equation constants
        if is_female:
            kappa = 0.7
            alpha = -0.329
            sex_factor = 1.018
        else:
            kappa = 0.9
            alpha = -0.411
            sex_factor = 1.0
            
        race_factor = 1.159 if is_black else 1.0
        
        # Calculate eGFR
        min_cr = min(creatinine / kappa, 1)
        max_cr = max(creatinine / kappa, 1)
        
        egfr = 141 * (min_cr ** alpha) * (max_cr ** -1.209) * (0.993 ** age) * sex_factor * race_factor
        
        return round(egfr, 2)
    
    def get_gfr_stage(self, egfr: float) -> str:
        """
        Determine CKD stage based on eGFR value.
        
        Returns:
            Stage string (G1, G2, G3a, G3b, G4, G5)
        """
        if egfr >= 90:
            return "G1"  # Normal or high
        elif egfr >= 60:
            return "G2"  # Mildly decreased
        elif egfr >= 45:
            return "G3a"  # Mildly to moderately decreased
        elif egfr >= 30:
            return "G3b"  # Moderately to severely decreased
        elif egfr >= 15:
            return "G4"  # Severely decreased
        else:
            return "G5"  # Kidney failure
    
    def get_albuminuria_category(self, acr: float) -> str:
        """
        Determine albuminuria category based on Albumin-to-Creatinine Ratio.
        
        Args:
            acr: Albumin-to-Creatinine Ratio in mg/g
            
        Returns:
            Category string (A1, A2, A3)
        """
        if acr < 30:
            return "A1"  # Normal to mildly increased
        elif acr < 300:
            return "A2"  # Moderately increased
        else:
            return "A3"  # Severely increased
    
    def get_risk_level(self, gfr_stage: str, albuminuria_category: str) -> Dict[str, any]:
        """
        Determine overall risk level based on GFR stage and albuminuria category.
        
        Returns:
            Dictionary with risk level, description, and recommendations
        """
        # Risk matrix based on KDIGO guidelines
        risk_matrix = {
            ("G1", "A1"): {"level": "Low", "color": "green", "action": "Monitor"},
            ("G1", "A2"): {"level": "Moderate", "color": "yellow", "action": "Monitor closely"},
            ("G1", "A3"): {"level": "High", "color": "orange", "action": "Refer to specialist"},
            ("G2", "A1"): {"level": "Low", "color": "green", "action": "Monitor"},
            ("G2", "A2"): {"level": "Moderate", "color": "yellow", "action": "Monitor closely"},
            ("G2", "A3"): {"level": "High", "color": "orange", "action": "Refer to specialist"},
            ("G3a", "A1"): {"level": "Moderate", "color": "yellow", "action": "Monitor closely"},
            ("G3a", "A2"): {"level": "High", "color": "orange", "action": "Refer to specialist"},
            ("G3a", "A3"): {"level": "Very High", "color": "red", "action": "Urgent referral"},
            ("G3b", "A1"): {"level": "High", "color": "orange", "action": "Refer to specialist"},
            ("G3b", "A2"): {"level": "Very High", "color": "red", "action": "Urgent referral"},
            ("G3b", "A3"): {"level": "Very High", "color": "red", "action": "Urgent referral"},
            ("G4", "A1"): {"level": "Very High", "color": "red", "action": "Urgent referral"},
            ("G4", "A2"): {"level": "Very High", "color": "red", "action": "Urgent referral"},
            ("G4", "A3"): {"level": "Very High", "color": "red", "action": "Urgent referral"},
            ("G5", "A1"): {"level": "Critical", "color": "darkred", "action": "Immediate intervention"},
            ("G5", "A2"): {"level": "Critical", "color": "darkred", "action": "Immediate intervention"},
            ("G5", "A3"): {"level": "Critical", "color": "darkred", "action": "Immediate intervention"},
        }
        
        key = (gfr_stage, albuminuria_category)
        risk = risk_matrix.get(key, {"level": "Unknown", "color": "gray", "action": "Consult doctor"})
        
        return {
            "gfr_stage": gfr_stage,
            "albuminuria_category": albuminuria_category,
            "risk_level": risk["level"],
            "risk_color": risk["color"],
            "recommended_action": risk["action"]
        }
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features from existing features."""
        df = df.copy()
        
        # Common feature interactions for kidney disease
        if 'bu' in df.columns and 'sc' in df.columns:
            df['bu_sc_ratio'] = df['bu'] / (df['sc'] + 0.01)  # BUN/Creatinine ratio
            
        if 'bun' in df.columns and 'sc' in df.columns:
            df['bun_sc_ratio'] = df['bun'] / (df['sc'] + 0.01)
            
        if 'sod' in df.columns and 'pot' in df.columns:
            df['sod_pot_ratio'] = df['sod'] / (df['pot'] + 0.01)
            
        return df
    
    def create_categorical_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create categorical bins for KEY numerical features based on medical guidelines.
        Uses a fit/transform pattern to prevent column mismatch between splits:
          - First call (training): expands dummies and records the column schema.
          - Subsequent calls (val/test/inference): reindexes to that schema exactly,
            filling any missing category columns with 0 and dropping any extra ones.

        [ALERT] ANTI-LEAKAGE: dummy column schema is learned from training data only.
        [ALERT] ANTI-OVERFITTING: Only key kidney-relevant markers are binned.
        """
        df = df.copy()

        # 1. Age Categories
        if 'age' in df.columns:
            df['age_cat'] = pd.cut(
                df['age'],
                bins=[0, 18, 40, 60, 120],
                labels=['young', 'adult', 'middle_aged', 'elderly']
            )

        # One-Hot Encode new categorical columns
        cat_cols = [c for c in ['age_cat'] if c in df.columns]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # ── Fit/Transform pattern ────────────────────────────────────────────
        # [ALERT] ANTI-LEAKAGE: On first call (training), record the dummy column
        # schema. On all subsequent calls (val/test/inference), reindex to exactly
        # that schema — filling unseen categories with 0, dropping unseen extras.
        if self._dummy_columns is None:
            # First call: this IS the training set — memorise its column order.
            self._dummy_columns = df.columns.tolist()
        else:
            # Subsequent calls: align to training schema (fill missing with 0).
            df = df.reindex(columns=self._dummy_columns, fill_value=0)

        return df
    
    def select_best_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        k: int = 15,
        method: str = 'f_classif'
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Select top k features using statistical tests.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            k: Number of features to select
            method: 'f_classif' or 'mutual_info'
            
        Returns:
            Selected features matrix and names
        """
        score_func = f_classif if method == 'f_classif' else mutual_info_classif
        
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        mask = selector.get_support()
        self.selected_features = [f for f, m in zip(feature_names, mask) if m]
        
        # Store feature importance
        scores = selector.scores_
        self.feature_importance = {
            name: score for name, score in zip(feature_names, scores)
        }
        
        return X_selected, self.selected_features
    
    def apply_pca(
        self,
        X: np.ndarray,
        n_components: float = 0.95
    ) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X: Feature matrix
            n_components: Number of components or variance ratio to keep
            
        Returns:
            Transformed features
        """
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        print(f"PCA: Reduced from {X.shape[1]} to {X_pca.shape[1]} components")
        print(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
        
        return X_pca


# Convenience functions for eGFR and staging
def calculate_egfr(creatinine: float, age: int, is_female: bool = False) -> float:
    """Calculate eGFR from creatinine, age, and sex."""
    fe = FeatureEngineer()
    return fe.calculate_egfr(creatinine, age, is_female)


def get_kidney_stage(egfr: float, acr: float = None) -> Dict:
    """Get complete kidney disease staging."""
    fe = FeatureEngineer()
    gfr_stage = fe.get_gfr_stage(egfr)
    
    if acr is not None:
        alb_category = fe.get_albuminuria_category(acr)
        return fe.get_risk_level(gfr_stage, alb_category)
    
    return {"gfr_stage": gfr_stage}


if __name__ == "__main__":
    # Test feature engineering
    fe = FeatureEngineer()
    
    # Test eGFR calculation
    egfr = fe.calculate_egfr(creatinine=2.3, age=70, is_female=False)
    print(f"eGFR: {egfr} mL/min/1.73m²")
    print(f"GFR Stage: {fe.get_gfr_stage(egfr)}")
    
    # Test ACR staging
    acr = 44.44
    alb_category = fe.get_albuminuria_category(acr)
    print(f"Albuminuria Category: {alb_category}")
    
    # Test risk assessment
    risk = fe.get_risk_level(fe.get_gfr_stage(egfr), alb_category)
    print(f"Risk Assessment: {risk}")
