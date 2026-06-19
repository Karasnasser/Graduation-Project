"""
GFR Calculator and Disease Staging Module
Implements kidney disease staging based on GFR and albuminuria levels.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class GFRStage(Enum):
    """GFR-based CKD stages."""
    G1 = "G1"  # Normal or high (≥90)
    G2 = "G2"  # Mildly decreased (60-89)
    G3a = "G3a"  # Mild-moderate decrease (45-59)
    G3b = "G3b"  # Moderate-severe decrease (30-44)
    G4 = "G4"  # Severely decreased (15-29)
    G5 = "G5"  # Kidney failure (<15)


class AlbuminuriaCategory(Enum):
    """Albuminuria categories based on ACR."""
    A1 = "A1"  # Normal to mildly increased (<30 mg/g)
    A2 = "A2"  # Moderately increased (30-299 mg/g)
    A3 = "A3"  # Severely increased (≥300 mg/g)


class RiskLevel(Enum):
    """Overall CKD risk levels."""
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"
    VERY_HIGH = "Very High Risk"
    CRITICAL = "Critical - Kidney Failure"


@dataclass
class KidneyStageResult:
    """Complete kidney staging result."""
    gfr_stage: GFRStage
    albuminuria_category: Optional[AlbuminuriaCategory]
    risk_level: RiskLevel
    egfr_value: float
    acr_value: Optional[float]
    description: str
    recommendations: list
    urgency_color: str


class GFRCalculator:
    """
    Calculate eGFR using various formulas and determine kidney disease stage.
    """
    
    # GFR Stage thresholds
    GFR_THRESHOLDS = {
        'G1': 90,
        'G2': 60,
        'G3a': 45,
        'G3b': 30,
        'G4': 15,
        'G5': 0
    }
    
    # ACR thresholds (mg/g)
    ACR_THRESHOLDS = {
        'A1': 30,
        'A2': 300,
        'A3': float('inf')
    }
    
    # Risk matrix based on KDIGO guidelines
    RISK_MATRIX = {
        # (GFR Stage, Albuminuria Category): Risk Level
        ('G1', 'A1'): RiskLevel.LOW,
        ('G1', 'A2'): RiskLevel.MODERATE,
        ('G1', 'A3'): RiskLevel.HIGH,
        ('G2', 'A1'): RiskLevel.LOW,
        ('G2', 'A2'): RiskLevel.MODERATE,
        ('G2', 'A3'): RiskLevel.HIGH,
        ('G3a', 'A1'): RiskLevel.MODERATE,
        ('G3a', 'A2'): RiskLevel.HIGH,
        ('G3a', 'A3'): RiskLevel.VERY_HIGH,
        ('G3b', 'A1'): RiskLevel.HIGH,
        ('G3b', 'A2'): RiskLevel.VERY_HIGH,
        ('G3b', 'A3'): RiskLevel.VERY_HIGH,
        ('G4', 'A1'): RiskLevel.VERY_HIGH,
        ('G4', 'A2'): RiskLevel.VERY_HIGH,
        ('G4', 'A3'): RiskLevel.VERY_HIGH,
        ('G5', 'A1'): RiskLevel.CRITICAL,
        ('G5', 'A2'): RiskLevel.CRITICAL,
        ('G5', 'A3'): RiskLevel.CRITICAL,
    }
    
    # Stage descriptions
    STAGE_DESCRIPTIONS = {
        'G1': "وظائف الكلى طبيعية أو مرتفعة",
        'G2': "انخفاض طفيف في وظائف الكلى",
        'G3a': "انخفاض طفيف إلى متوسط في وظائف الكلى",
        'G3b': "انخفاض متوسط إلى شديد في وظائف الكلى",
        'G4': "انخفاض شديد في وظائف الكلى",
        'G5': "[WARN] فشل كلوي - يحتاج تدخل فوري",
    }
    
    # Risk colors for UI
    RISK_COLORS = {
        RiskLevel.LOW: "#4CAF50",  # Green
        RiskLevel.MODERATE: "#FFC107",  # Yellow
        RiskLevel.HIGH: "#FF9800",  # Orange
        RiskLevel.VERY_HIGH: "#F44336",  # Red
        RiskLevel.CRITICAL: "#9C27B0",  # Purple (urgent)
    }
    
    def __init__(self):
        self.last_calculation = None
    
    def calculate_egfr_ckdepi(
        self,
        creatinine: float,
        age: int,
        is_female: bool = False,
        is_black: bool = False
    ) -> float:
        """
        Calculate eGFR using CKD-EPI 2021 equation.
        
        Args:
            creatinine: Serum creatinine in mg/dL
            age: Patient age in years
            is_female: True if patient is female
            is_black: True if patient is African American
            
        Returns:
            eGFR in mL/min/1.73m²
        """
        if creatinine <= 0 or age <= 0:
            return 0.0
        
        # CKD-EPI 2021 (race-free equation)
        if is_female:
            if creatinine <= 0.7:
                egfr = 142 * ((creatinine / 0.7) ** -0.241) * (0.9938 ** age) * 1.012
            else:
                egfr = 142 * ((creatinine / 0.7) ** -1.200) * (0.9938 ** age) * 1.012
        else:
            if creatinine <= 0.9:
                egfr = 142 * ((creatinine / 0.9) ** -0.302) * (0.9938 ** age)
            else:
                egfr = 142 * ((creatinine / 0.9) ** -1.200) * (0.9938 ** age)
        
        return round(egfr, 2)
    
    def calculate_egfr_mdrd(
        self,
        creatinine: float,
        age: int,
        is_female: bool = False,
        is_black: bool = False
    ) -> float:
        """
        Calculate eGFR using MDRD equation (older formula).
        
        Returns:
            eGFR in mL/min/1.73m²
        """
        if creatinine <= 0 or age <= 0:
            return 0.0
        
        egfr = 175 * (creatinine ** -1.154) * (age ** -0.203)
        
        if is_female:
            egfr *= 0.742
        if is_black:
            egfr *= 1.212
        
        return round(egfr, 2)
    
    def get_gfr_stage(self, egfr: float) -> GFRStage:
        """Determine GFR stage from eGFR value."""
        if egfr >= 90:
            return GFRStage.G1
        elif egfr >= 60:
            return GFRStage.G2
        elif egfr >= 45:
            return GFRStage.G3a
        elif egfr >= 30:
            return GFRStage.G3b
        elif egfr >= 15:
            return GFRStage.G4
        else:
            return GFRStage.G5
    
    def get_albuminuria_category(self, acr: float) -> AlbuminuriaCategory:
        """Determine albuminuria category from ACR value."""
        if acr < 30:
            return AlbuminuriaCategory.A1
        elif acr < 300:
            return AlbuminuriaCategory.A2
        else:
            return AlbuminuriaCategory.A3
    
    def get_risk_level(
        self,
        gfr_stage: GFRStage,
        alb_category: Optional[AlbuminuriaCategory] = None
    ) -> RiskLevel:
        """Determine overall risk level based on GFR and albuminuria."""
        if alb_category is None:
            # Use GFR stage alone
            if gfr_stage in [GFRStage.G1, GFRStage.G2]:
                return RiskLevel.LOW
            elif gfr_stage == GFRStage.G3a:
                return RiskLevel.MODERATE
            elif gfr_stage == GFRStage.G3b:
                return RiskLevel.HIGH
            elif gfr_stage == GFRStage.G4:
                return RiskLevel.VERY_HIGH
            else:
                return RiskLevel.CRITICAL
        
        key = (gfr_stage.value, alb_category.value)
        return self.RISK_MATRIX.get(key, RiskLevel.MODERATE)
    
    def get_recommendations(self, risk_level: RiskLevel, gfr_stage: GFRStage) -> list:
        """Get recommendations based on risk level."""
        general = [
            "المتابعة الدورية مع طبيب الكلى",
            "الحفاظ على ضغط الدم في المعدل الطبيعي",
            "التحكم في مستوى السكر (لمرضى السكري)",
        ]
        
        if risk_level == RiskLevel.LOW:
            return [
                "[OK] وظائف الكلى جيدة",
                "الفحص الدوري كل 12 شهر",
                "الحفاظ على نمط حياة صحي",
            ]
        elif risk_level == RiskLevel.MODERATE:
            return general + [
                "[WARN] الفحص الدوري كل 6 أشهر",
                "تقليل الملح في الطعام",
            ]
        elif risk_level == RiskLevel.HIGH:
            return general + [
                " مراجعة طبيب الكلى",
                "الفحص الدوري كل 3 أشهر",
                "مراقبة البروتين في البول",
            ]
        elif risk_level == RiskLevel.VERY_HIGH:
            return general + [
                " يجب مراجعة طبيب الكلى فوراً",
                "الفحص الدوري كل شهر",
                "التحضير لعلاجات متقدمة",
            ]
        else:  # CRITICAL
            return [
                " حالة طوارئ - فشل كلوي",
                "يحتاج غسيل كلى أو زراعة",
                "التوجه لمستشفى متخصص فوراً",
                "متابعة مستمرة مع فريق طبي متخصص",
            ]
    
    def calculate_stage(
        self,
        creatinine: Optional[float] = None,
        egfr: Optional[float] = None,
        acr: Optional[float] = None,
        age: int = 50,
        is_female: bool = False
    ) -> KidneyStageResult:
        """
        Complete kidney staging calculation.
        
        Args:
            creatinine: Serum creatinine in mg/dL (optional if eGFR provided)
            egfr: Pre-calculated eGFR (optional if creatinine provided)
            acr: Albumin-to-Creatinine Ratio in mg/g
            age: Patient age
            is_female: Patient sex
            
        Returns:
            KidneyStageResult with complete staging information
        """
        # Calculate eGFR if not provided
        if egfr is None and creatinine is not None:
            egfr = self.calculate_egfr_ckdepi(creatinine, age, is_female)
        elif egfr is None:
            raise ValueError("Either creatinine or egfr must be provided")
        
        # Get GFR stage
        gfr_stage = self.get_gfr_stage(egfr)
        
        # Get albuminuria category if ACR provided
        alb_category = self.get_albuminuria_category(acr) if acr is not None else None
        
        # Get risk level
        risk_level = self.get_risk_level(gfr_stage, alb_category)
        
        # Get recommendations
        recommendations = self.get_recommendations(risk_level, gfr_stage)
        
        # Get description
        description = self.STAGE_DESCRIPTIONS.get(gfr_stage.value, "Unknown stage")
        
        # Get color
        urgency_color = self.RISK_COLORS.get(risk_level, "#9E9E9E")
        
        result = KidneyStageResult(
            gfr_stage=gfr_stage,
            albuminuria_category=alb_category,
            risk_level=risk_level,
            egfr_value=egfr,
            acr_value=acr,
            description=description,
            recommendations=recommendations,
            urgency_color=urgency_color
        )
        
        self.last_calculation = result
        return result
    
    def format_result(self, result: KidneyStageResult) -> str:
        """Format result as readable string."""
        lines = [
            "=" * 50,
            "Kidney Function Analysis Result",
            "=" * 50,
            f"",
            f"  eGFR: {result.egfr_value} mL/min/1.73m2",
            f"  Kidney Stage: {result.gfr_stage.value}",
        ]
        
        if result.albuminuria_category:
            lines.append(f"  Albuminuria: {result.albuminuria_category.value} (ACR: {result.acr_value} mg/g)")
        
        lines.extend([
            f"",
            f"  Risk Level: {result.risk_level.value}",
            f"  Status: {result.description}",
            f"",
            "  Recommendations:",
        ])
        
        for rec in result.recommendations:
            lines.append(f"  - {rec}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test the calculator
    calc = GFRCalculator()
    
    # Example: 70-year-old male with creatinine 2.3 and ACR 44.44
    result = calc.calculate_stage(
        creatinine=2.3,
        acr=44.44,
        age=70,
        is_female=False
    )
    
    print(calc.format_result(result))
