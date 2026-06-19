"""
Risk Assessor Module
Complete risk assessment for kidney disease patients.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from .gfr_calculator import GFRCalculator, GFRStage, AlbuminuriaCategory, RiskLevel


@dataclass
class ProgressionRisk:
    """Risk of disease progression."""
    risk_percentage: float
    time_to_esrd: Optional[str]  # Estimated time to End-Stage Renal Disease
    confidence: float


@dataclass
class CompleteAssessment:
    """Complete patient risk assessment."""
    ckd_prediction: bool
    ckd_probability: float
    gfr_stage: GFRStage
    albuminuria_category: Optional[AlbuminuriaCategory]
    risk_level: RiskLevel
    egfr_value: float
    acr_value: Optional[float]
    progression_risk: ProgressionRisk
    recommendations: List[str]
    alerts: List[str]
    enhanced_risk_score: Optional[float] = None  # 0-100 multi-factor score
    is_fast_progressor: bool = False  # True if eGFR decline > 5 mL/min/year


class RiskAssessor:
    """
    Complete risk assessment combining ML predictions and staging.
    """
    
    def __init__(self):
        self.gfr_calculator = GFRCalculator()
    
    def assess_progression_risk(
        self,
        gfr_stage: GFRStage,
        alb_category: Optional[AlbuminuriaCategory],
        egfr: float,
        age: int = 50
    ) -> ProgressionRisk:
        """
        Assess risk of disease progression to ESRD.
        
        Based on KDIGO risk classification and epidemiological data.
        """
        # Base progression rates (approximate 5-year risk)
        base_risks = {
            (GFRStage.G1, AlbuminuriaCategory.A1): 0.01,
            (GFRStage.G1, AlbuminuriaCategory.A2): 0.03,
            (GFRStage.G1, AlbuminuriaCategory.A3): 0.08,
            (GFRStage.G2, AlbuminuriaCategory.A1): 0.02,
            (GFRStage.G2, AlbuminuriaCategory.A2): 0.05,
            (GFRStage.G2, AlbuminuriaCategory.A3): 0.12,
            (GFRStage.G3a, AlbuminuriaCategory.A1): 0.05,
            (GFRStage.G3a, AlbuminuriaCategory.A2): 0.12,
            (GFRStage.G3a, AlbuminuriaCategory.A3): 0.25,
            (GFRStage.G3b, AlbuminuriaCategory.A1): 0.15,
            (GFRStage.G3b, AlbuminuriaCategory.A2): 0.30,
            (GFRStage.G3b, AlbuminuriaCategory.A3): 0.50,
            (GFRStage.G4, AlbuminuriaCategory.A1): 0.40,
            (GFRStage.G4, AlbuminuriaCategory.A2): 0.55,
            (GFRStage.G4, AlbuminuriaCategory.A3): 0.70,
            (GFRStage.G5, AlbuminuriaCategory.A1): 0.80,
            (GFRStage.G5, AlbuminuriaCategory.A2): 0.90,
            (GFRStage.G5, AlbuminuriaCategory.A3): 0.95,
        }
        
        if alb_category:
            key = (gfr_stage, alb_category)
            risk = base_risks.get(key, 0.1)
        else:
            # Estimate risk based on GFR stage alone
            stage_risks = {
                GFRStage.G1: 0.02,
                GFRStage.G2: 0.04,
                GFRStage.G3a: 0.10,
                GFRStage.G3b: 0.25,
                GFRStage.G4: 0.50,
                GFRStage.G5: 0.85,
            }
            risk = stage_risks.get(gfr_stage, 0.1)
        
        # Adjust for age (higher age = higher risk)
        if age > 70:
            risk *= 1.2
        elif age > 60:
            risk *= 1.1
        
        risk = min(risk, 0.99)  # Cap at 99%
        
        # Estimate time to ESRD
        if egfr < 15:
            time_estimate = "فوري - يحتاج علاج"
        elif egfr < 30:
            time_estimate = "1-3 سنوات"
        elif egfr < 45:
            time_estimate = "3-5 سنوات"
        elif egfr < 60:
            time_estimate = "5-10 سنوات"
        else:
            time_estimate = "غير متوقع في المدى القريب"
        
        return ProgressionRisk(
            risk_percentage=round(risk * 100, 1),
            time_to_esrd=time_estimate,
            confidence=0.75
        )
    
    def generate_alerts(
        self,
        egfr: float,
        acr: Optional[float],
        creatinine: float,
        other_values: Dict[str, float] = None
    ) -> List[str]:
        """Generate clinical alerts based on lab values."""
        alerts = []
        
        # Critical eGFR
        if egfr < 15:
            alerts.append(" تحذير: فشل كلوي! يحتاج تدخل فوري")
        elif egfr < 30:
            alerts.append("[WARN] تحذير: وظائف الكلى منخفضة جداً")
        
        # High creatinine
        if creatinine > 3.0:
            alerts.append(" كرياتينين مرتفع جداً")
        elif creatinine > 2.0:
            alerts.append(" كرياتينين مرتفع")
        
        # High ACR
        if acr:
            if acr >= 300:
                alerts.append(" بروتين في البول مرتفع جداً (A3)")
            elif acr >= 30:
                alerts.append(" بروتين في البول مرتفع (A2)")
        
        # Check other values
        if other_values:
            # Potassium
            if other_values.get('potassium', 0) > 5.5:
                alerts.append("[WARN] بوتاسيوم مرتفع - خطر على القلب")
            
            # Hemoglobin
            if other_values.get('hemoglobin', 15) < 10:
                alerts.append("[WARN] أنيميا - هيموجلوبين منخفض")
            
            # Blood Pressure
            bp_sys = other_values.get('bp_systolic', 120)
            if bp_sys > 140:
                alerts.append("[WARN] ضغط الدم مرتفع")
            
            # HbA1c
            hba1c = other_values.get('hba1c', 0)
            if hba1c > 9.0:
                alerts.append(" HbA1c مرتفع جداً - سيطرة ضعيفة على السكر")
            elif hba1c > 7.0:
                alerts.append(" HbA1c مرتفع - يجب تحسين السيطرة على السكر")
            
            # Uric Acid
            uric_acid = other_values.get('uric_acid', 0)
            if uric_acid > 8.0:
                alerts.append("[WARN] حمض اليوريك مرتفع - خطر تلف الكلى")
            
            # BMI
            bmi = other_values.get('bmi', 0)
            if bmi > 30:
                alerts.append("[WARN] سمنة - عامل خطر لتطور المرض")
        
        return alerts
    
    def calculate_enhanced_risk_score(
        self,
        egfr: float,
        acr: Optional[float] = None,
        hba1c: float = 5.5,
        creatinine: float = 1.0,
        uric_acid: float = 5.0,
        bmi: float = 25.0,
        smoking: bool = False,
        diabetes_duration: float = 0,
        age: int = 50
    ) -> float:
        """
        Calculate enhanced multi-factor risk score (0-100).
        تقييم شامل للمخاطر باستخدام عدة عوامل حيوية.
        
        Combines multiple biomarkers with clinical guidelines:
        - eGFR (30%): Primary kidney function indicator
        - UACR (20%): Early kidney damage marker
        - HbA1c (15%): Glycemic control
        - Creatinine (10%): Kidney filtration
        - Uric Acid (10%): Inflammation marker
        - Other factors (15%): BMI, smoking, age, diabetes duration
        
        Returns:
            Risk score from 0 (lowest) to 100 (highest risk)
        """
        score = 0.0
        
        # eGFR component (30 points max)
        if egfr < 15:
            score += 30
        elif egfr < 30:
            score += 25
        elif egfr < 45:
            score += 20
        elif egfr < 60:
            score += 15
        elif egfr < 90:
            score += 8
        else:
            score += 0
        
        # UACR component (20 points max)
        if acr is not None:
            if acr >= 300:
                score += 20
            elif acr >= 30:
                score += 12
            else:
                score += 0
        
        # HbA1c component (15 points max)
        if hba1c >= 9.0:
            score += 15
        elif hba1c >= 7.5:
            score += 10
        elif hba1c >= 6.5:
            score += 5
        
        # Creatinine component (10 points max)
        if creatinine > 3.0:
            score += 10
        elif creatinine > 2.0:
            score += 7
        elif creatinine > 1.2:
            score += 3
        
        # Uric acid component (10 points max)
        if uric_acid > 8.0:
            score += 10
        elif uric_acid > 7.0:
            score += 5
        
        # Other factors (15 points max)
        if bmi > 30:
            score += 4
        if smoking:
            score += 4
        if age > 65:
            score += 3
        if diabetes_duration > 10:
            score += 4
        elif diabetes_duration > 5:
            score += 2
        
        return min(100.0, round(score, 1))
    
    def get_lifestyle_recommendations(
        self,
        risk_level: RiskLevel,
        has_diabetes: bool = False,
        has_hypertension: bool = False
    ) -> List[str]:
        """Get lifestyle recommendations based on risk factors."""
        recommendations = []
        
        # General recommendations
        recommendations.extend([
            "Follow a low-sodium, healthy diet",
            "Drink adequate water (2-3 liters daily)",
            "Avoid NSAID pain relievers without consulting a doctor",
        ])
        
        if has_diabetes or risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]:
            recommendations.append("Monitor blood glucose level regularly")
        
        if has_hypertension or risk_level != RiskLevel.LOW:
            recommendations.append("Monitor blood pressure daily")
        
        if risk_level in [RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]:
            recommendations.extend([
                "Limit dietary protein intake",
                "Avoid foods rich in potassium",
                "Follow up with a nephrologist monthly",
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Follow up with a doctor every 3 months")
        else:
            recommendations.append("Routine screening every 6-12 months")
        
        return recommendations
    
    def complete_assessment(
        self,
        ckd_probability: float,
        creatinine: Optional[float] = None,
        egfr: Optional[float] = None,
        acr: Optional[float] = None,
        age: int = 50,
        is_female: bool = False,
        other_values: Dict[str, float] = None
    ) -> CompleteAssessment:
        """
        Generate complete patient assessment combining ML prediction and staging.
        
        Args:
            ckd_probability: ML model's CKD probability (0-1)
            creatinine: Serum creatinine in mg/dL
            egfr: Pre-calculated eGFR (optional)
            acr: Albumin-to-Creatinine Ratio
            age: Patient age
            is_female: Patient sex
            other_values: Other lab values for alerts
            
        Returns:
            CompleteAssessment with all results
        """
        # Calculate eGFR if needed
        if egfr is None and creatinine is not None:
            egfr = self.gfr_calculator.calculate_egfr_ckdepi(creatinine, age, is_female)
        elif egfr is None:
            egfr = 90  # Default to normal
        
        # Get staging
        staging = self.gfr_calculator.calculate_stage(
            egfr=egfr,
            acr=acr,
            age=age,
            is_female=is_female
        )
        
        # Get progression risk
        progression = self.assess_progression_risk(
            staging.gfr_stage,
            staging.albuminuria_category,
            egfr,
            age
        )
        
        # Generate alerts
        alerts = self.generate_alerts(
            egfr,
            acr,
            creatinine if creatinine else 1.0,
            other_values
        )
        
        # Get recommendations
        recommendations = self.get_lifestyle_recommendations(staging.risk_level)
        
        # Determine CKD prediction
        ckd_prediction = ckd_probability >= 0.5 or staging.gfr_stage not in [GFRStage.G1, GFRStage.G2]
        
        # Calculate enhanced risk score
        enhanced_score = self.calculate_enhanced_risk_score(
            egfr=egfr,
            acr=acr,
            hba1c=other_values.get('hba1c', 5.5) if other_values else 5.5,
            creatinine=creatinine if creatinine else 1.0,
            uric_acid=other_values.get('uric_acid', 5.0) if other_values else 5.0,
            bmi=other_values.get('bmi', 25.0) if other_values else 25.0,
            smoking=bool(other_values.get('smoking', 0)) if other_values else False,
            diabetes_duration=other_values.get('diabetes_duration', 0) if other_values else 0,
            age=age
        )
        
        return CompleteAssessment(
            ckd_prediction=ckd_prediction,
            ckd_probability=ckd_probability,
            gfr_stage=staging.gfr_stage,
            albuminuria_category=staging.albuminuria_category,
            risk_level=staging.risk_level,
            egfr_value=egfr,
            acr_value=acr,
            progression_risk=progression,
            recommendations=recommendations,
            alerts=alerts,
            enhanced_risk_score=enhanced_score
        )
    
    def format_assessment(self, assessment: CompleteAssessment) -> str:
        """Format assessment as readable string."""
        lines = [
            "=" * 60,
            "Complete Kidney Function Assessment Report",
            "=" * 60,
            "",
            f"  CKD Prediction: {'Positive' if assessment.ckd_prediction else 'Negative'}",
            f"  Probability: {assessment.ckd_probability * 100:.1f}%",
            "",
            f"  eGFR: {assessment.egfr_value} mL/min/1.73m2",
            f"  Disease Stage: {assessment.gfr_stage.value}",
        ]
        
        if assessment.albuminuria_category:
            lines.append(f"  Albuminuria Category: {assessment.albuminuria_category.value}")
        
        if assessment.acr_value:
            lines.append(f"   (ACR: {assessment.acr_value} mg/g)")
        
        lines.extend([
            "",
            f"  Risk Level: {assessment.risk_level.value}",
            f"  Progression Risk: {assessment.progression_risk.risk_percentage}%",
            f"  Estimated Time to ESRD: {assessment.progression_risk.time_to_esrd}",
        ])
        
        if assessment.alerts:
            lines.extend(["", "  Important Alerts:"])
            for alert in assessment.alerts:
                lines.append(f"  - {alert}")
        
        lines.extend(["", "  Recommendations:"])
        for rec in assessment.recommendations:
            lines.append(f"  - {rec}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test risk assessor
    assessor = RiskAssessor()
    
    assessment = assessor.complete_assessment(
        ckd_probability=0.85,
        creatinine=2.3,
        acr=44.44,
        age=70,
        is_female=False
    )
    
    print(assessor.format_assessment(assessment))
