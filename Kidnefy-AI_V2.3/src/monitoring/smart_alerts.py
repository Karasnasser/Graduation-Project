"""
Smart Alerts Engine
نظام التنبيهات الذكية — AI-powered alerts beyond simple rule-based thresholds.

Features:
    1. Anomaly Detection — Isolation Forest per-patient personalized baselines
    2. Predictive Analytics — Multi-biomarker trend scoring with future risk prediction
    3. NLP Symptom Analysis — Gemini-based correlation of symptoms with lab data
    4. Alert Aggregation — Unified priority-sorted alert system
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .longitudinal_monitor import LongitudinalMonitor


# =============================================================================
# Data Structures
# =============================================================================

class AlertPriority(str, Enum):
    """Priority levels for smart alerts."""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class AlertType(str, Enum):
    """Type of smart alert."""
    ANOMALY = "anomaly"
    PREDICTIVE = "predictive"
    SYMPTOM = "symptom"
    RULE_BASED = "rule_based"


@dataclass
class SmartAlert:
    """A single smart alert."""
    alert_type: str       # AlertType value
    priority: str         # AlertPriority value
    title: str            # Short title (Arabic)
    message: str          # Detailed message (Arabic)
    details: Dict[str, Any]  # Technical details
    timestamp: str        # ISO format


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single measurement."""
    is_anomaly: bool
    anomaly_score: float   # -1 to 0 (more negative = more anomalous)
    anomalous_features: List[Dict[str, Any]]  # Which features are anomalous
    severity: str          # "high", "medium", "low"


@dataclass
class PredictiveResult:
    """Result of predictive analytics."""
    overall_risk_score: float   # 0-100
    risk_classification: str    # "critical", "high", "moderate", "low", "stable"
    biomarker_trends: Dict[str, Dict[str, Any]]  # Per-biomarker trend info
    predicted_timeline: Optional[str]  # Human-readable timeline
    alert_message: str


@dataclass
class SymptomAnalysis:
    """Result of NLP symptom analysis."""
    urgency: str                # "emergency", "urgent", "routine"
    matched_conditions: List[Dict[str, Any]]  # Possible conditions
    recommendations: List[str]  # Actions to take
    correlation_with_labs: Optional[str]  # If patient data available
    raw_ai_response: Optional[str]  # Full Gemini response


# =============================================================================
# Symptom Keywords Database (for fallback when Gemini is unavailable)
# =============================================================================

SYMPTOM_KEYWORDS = {
    "emergency": {
        "keywords_ar": [
            "ضيق تنفس شديد", "لا أستطيع التنفس", "ألم صدر",
            "إغماء", "فقدان الوعي", "تشنجات", "دم في البول كثير",
            "لا أتبول", "انقطاع البول", "تورم شديد مفاجئ"
        ],
        "keywords_en": [
            "can't breathe", "severe chest pain", "unconscious",
            "seizure", "no urine output", "severe swelling",
            "blood in urine heavy", "fainting"
        ],
        "conditions": [
            {"name": "إصابة كلوية حادة (AKI)", "name_en": "Acute Kidney Injury"},
            {"name": "فشل كلوي حاد", "name_en": "Acute Renal Failure"},
            {"name": "فرط بوتاسيوم الدم", "name_en": "Hyperkalemia"},
        ]
    },
    "urgent": {
        "keywords_ar": [
            "تورم", "انتفاخ", "قلة البول", "بول داكن", "غثيان مستمر",
            "دوخة", "إرهاق شديد", "حكة شديدة", "ألم في الظهر",
            "ألم في الخاصرة", "دم في البول", "رغوة في البول",
            "ضغط مرتفع", "صداع شديد"
        ],
        "keywords_en": [
            "swelling", "edema", "dark urine", "foamy urine", "reduced urine",
            "persistent nausea", "severe fatigue", "flank pain",
            "blood in urine", "high blood pressure", "severe headache",
            "itching", "dizziness"
        ],
        "conditions": [
            {"name": "تدهور وظائف الكلى", "name_en": "Kidney Function Decline"},
            {"name": "احتباس السوائل", "name_en": "Fluid Retention"},
            {"name": "أنيميا مرتبطة بالكلى", "name_en": "Renal Anemia"},
        ]
    },
    "routine": {
        "keywords_ar": [
            "تعب", "عطش", "كثرة التبول", "بول فاتح", "تعب خفيف",
            "فقدان شهية", "مغص خفيف"
        ],
        "keywords_en": [
            "tired", "thirsty", "frequent urination", "mild fatigue",
            "loss of appetite", "mild pain"
        ],
        "conditions": [
            {"name": "مرحلة مبكرة من مرض الكلى", "name_en": "Early CKD"},
            {"name": "جفاف", "name_en": "Dehydration"},
        ]
    }
}


# =============================================================================
# Smart Alert Engine
# =============================================================================

class SmartAlertEngine:
    """
    AI-powered Smart Alerts Engine.
    
    Combines three AI techniques:
    1. Anomaly Detection (Isolation Forest) — personalized per-patient
    2. Predictive Analytics — multi-biomarker trend scoring
    3. NLP Symptom Analysis — Gemini-based + keyword fallback
    
    Usage:
        engine = SmartAlertEngine(monitor)
        
        # Anomaly detection
        anomalies = engine.detect_anomalies("patient_001")
        
        # Predictive analytics
        prediction = engine.predict_future_risk("patient_001")
        
        # Symptom analysis
        analysis = engine.analyze_symptoms("حاسس بتورم في رجلي ودوخة")
        
        # All alerts combined
        alerts = engine.generate_smart_alerts("patient_001")
    """
    
    # Biomarker weights for predictive scoring
    BIOMARKER_WEIGHTS = {
        'egfr': 0.40,     # Primary kidney function
        'uacr': 0.25,     # Early damage marker
        'hba1c': 0.20,    # Glycemic control
        'bp_systolic': 0.15  # Blood pressure
    }
    
    # Minimum measurements for anomaly detection
    MIN_ANOMALY_DATA = 4
    
    def __init__(
        self,
        monitor: LongitudinalMonitor,
        gemini_rag=None
    ):
        """
        Initialize Smart Alert Engine.
        
        Args:
            monitor: LongitudinalMonitor instance (for patient data)
            gemini_rag: Optional GeminiRAG instance (for NLP symptom analysis)
        """
        self.monitor = monitor
        self.gemini_rag = gemini_rag
    
    # =========================================================================
    # 1. Anomaly Detection
    # =========================================================================
    
    def detect_anomalies(self, patient_id: str) -> AnomalyResult:
        """
        Detect anomalies in a patient's latest measurement compared to their history.
        
        Uses Isolation Forest on multi-dimensional patient data to find
        measurements that deviate from the patient's personal baseline.
        
        اكتشاف التغيرات غير الطبيعية في نتائج المريض مقارنة بتاريخه الشخصي.
        
        Args:
            patient_id: Unique patient identifier
            
        Returns:
            AnomalyResult with anomaly status and details
        """
        data = self.monitor.get_patient_history(patient_id)
        
        if len(data) < self.MIN_ANOMALY_DATA:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomalous_features=[],
                severity="low"
            )
        
        # Build feature matrix from patient history
        feature_names = ['egfr', 'creatinine', 'uacr', 'hba1c', 'bp_systolic']
        matrix = []
        
        for m in data:
            row = []
            for feat in feature_names:
                val = m.get(feat)
                if val is not None:
                    row.append(float(val))
                else:
                    row.append(np.nan)
            matrix.append(row)
        
        matrix = np.array(matrix, dtype=float)
        
        # Handle missing values: fill with column median
        for col_idx in range(matrix.shape[1]):
            col = matrix[:, col_idx]
            non_nan = col[~np.isnan(col)]
            if len(non_nan) > 0:
                median_val = np.median(non_nan)
                matrix[:, col_idx] = np.where(np.isnan(col), median_val, col)
            else:
                matrix[:, col_idx] = 0  # All NaN → fill with 0
        
        # Check if we have enough variance
        variances = np.var(matrix, axis=0)
        useful_cols = variances > 1e-10
        if not np.any(useful_cols):
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomalous_features=[],
                severity="low"
            )
        
        # Use only columns with variance
        X = matrix[:, useful_cols]
        used_features = [f for f, use in zip(feature_names, useful_cols) if use]
        
        if not SKLEARN_AVAILABLE or X.shape[0] < self.MIN_ANOMALY_DATA:
            # Fallback: simple z-score based anomaly detection
            return self._fallback_anomaly_detection(X, used_features, data)
        
        # Isolation Forest
        contamination = min(0.3, 1.0 / len(data))
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=50
        )
        
        predictions = iso_forest.fit_predict(X)
        scores = iso_forest.decision_function(X)
        
        # Check latest measurement
        latest_pred = predictions[-1]
        latest_score = scores[-1]
        is_anomaly = latest_pred == -1
        
        # Find which features are anomalous (z-score from mean)
        anomalous_features = []
        if is_anomaly:
            means = np.mean(X[:-1], axis=0)  # Mean of all except latest
            stds = np.std(X[:-1], axis=0)
            stds = np.where(stds < 1e-10, 1.0, stds)  # Avoid division by zero
            
            latest_row = X[-1]
            z_scores = np.abs((latest_row - means) / stds)
            
            for i, (feat, z) in enumerate(zip(used_features, z_scores)):
                if z > 1.5:  # Significant deviation
                    anomalous_features.append({
                        "feature": feat,
                        "feature_ar": self._feature_name_ar(feat),
                        "latest_value": float(latest_row[i]),
                        "historical_mean": float(means[i]),
                        "z_score": float(z),
                        "direction": "increased" if latest_row[i] > means[i] else "decreased"
                    })
            
            # Sort by z-score (most anomalous first)
            anomalous_features.sort(key=lambda x: x['z_score'], reverse=True)
        
        # Determine severity
        if latest_score < -0.3:
            severity = "high"
        elif latest_score < -0.1:
            severity = "medium"
        else:
            severity = "low"
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=float(latest_score),
            anomalous_features=anomalous_features,
            severity=severity
        )
    
    def _fallback_anomaly_detection(
        self,
        X: np.ndarray,
        feature_names: List[str],
        data: List[Dict]
    ) -> AnomalyResult:
        """Fallback anomaly detection using simple z-scores."""
        if X.shape[0] < 2:
            return AnomalyResult(False, 0.0, [], "low")
        
        means = np.mean(X[:-1], axis=0)
        stds = np.std(X[:-1], axis=0)
        stds = np.where(stds < 1e-10, 1.0, stds)
        
        latest = X[-1]
        z_scores = np.abs((latest - means) / stds)
        max_z = float(np.max(z_scores))
        
        is_anomaly = max_z > 2.0
        
        anomalous_features = []
        for i, (feat, z) in enumerate(zip(feature_names, z_scores)):
            if z > 1.5:
                anomalous_features.append({
                    "feature": feat,
                    "feature_ar": self._feature_name_ar(feat),
                    "latest_value": float(latest[i]),
                    "historical_mean": float(means[i]),
                    "z_score": float(z),
                    "direction": "increased" if latest[i] > means[i] else "decreased"
                })
        
        anomalous_features.sort(key=lambda x: x['z_score'], reverse=True)
        
        severity = "high" if max_z > 3.0 else ("medium" if max_z > 2.0 else "low")
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=-max_z / 5.0,  # Normalize to similar scale
            anomalous_features=anomalous_features,
            severity=severity
        )
    
    # =========================================================================
    # 2. Predictive Analytics
    # =========================================================================
    
    def predict_future_risk(self, patient_id: str) -> PredictiveResult:
        """
        Predict future kidney disease risk using multi-biomarker trend analysis.
        
        Analyzes trends in eGFR, UACR, HbA1c, and blood pressure to calculate
        a weighted risk score and predict disease trajectory.
        
        تحليل اتجاهات متعددة للتنبؤ بتطور مرض الكلى.
        
        Args:
            patient_id: Unique patient identifier
            
        Returns:
            PredictiveResult with risk score, classification, and timeline
        """
        data = self.monitor.get_patient_history(patient_id)
        
        if len(data) < 2:
            return PredictiveResult(
                overall_risk_score=0.0,
                risk_classification="insufficient_data",
                biomarker_trends={},
                predicted_timeline=None,
                alert_message="بيانات غير كافية للتحليل التنبؤي. يحتاج على الأقل قياسين."
            )
        
        # Calculate individual biomarker trends
        biomarker_trends = {}
        risk_components = {}
        
        # eGFR trend (most important — 40%)
        egfr_trend = self._calculate_biomarker_trend(data, 'egfr', lower_is_worse=True)
        biomarker_trends['egfr'] = egfr_trend
        risk_components['egfr'] = egfr_trend.get('risk_contribution', 0)
        
        # UACR trend (25%)
        uacr_trend = self._calculate_biomarker_trend(data, 'uacr', lower_is_worse=False)
        biomarker_trends['uacr'] = uacr_trend
        risk_components['uacr'] = uacr_trend.get('risk_contribution', 0)
        
        # HbA1c trend (20%)
        hba1c_trend = self._calculate_biomarker_trend(data, 'hba1c', lower_is_worse=False)
        biomarker_trends['hba1c'] = hba1c_trend
        risk_components['hba1c'] = hba1c_trend.get('risk_contribution', 0)
        
        # Blood pressure trend (15%)
        bp_trend = self._calculate_biomarker_trend(data, 'bp_systolic', lower_is_worse=False)
        biomarker_trends['bp_systolic'] = bp_trend
        risk_components['bp_systolic'] = bp_trend.get('risk_contribution', 0)
        
        # Calculate weighted overall risk score (0-100)
        overall_score = 0.0
        total_weight = 0.0
        for biomarker, weight in self.BIOMARKER_WEIGHTS.items():
            if biomarker in risk_components and risk_components[biomarker] is not None:
                overall_score += risk_components[biomarker] * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_score = (overall_score / total_weight)
        
        overall_score = min(100.0, max(0.0, overall_score))
        
        # Classify risk
        if overall_score >= 75:
            classification = "critical"
        elif overall_score >= 50:
            classification = "high"
        elif overall_score >= 30:
            classification = "moderate"
        elif overall_score >= 10:
            classification = "low"
        else:
            classification = "stable"
        
        # Generate timeline prediction
        timeline = self._predict_timeline(data, egfr_trend)
        
        # Generate alert message
        alert_message = self._generate_predictive_alert(
            classification, overall_score, biomarker_trends, timeline
        )
        
        return PredictiveResult(
            overall_risk_score=round(overall_score, 1),
            risk_classification=classification,
            biomarker_trends=biomarker_trends,
            predicted_timeline=timeline,
            alert_message=alert_message
        )
    
    def _calculate_biomarker_trend(
        self,
        data: List[Dict],
        biomarker: str,
        lower_is_worse: bool = True
    ) -> Dict[str, Any]:
        """Calculate trend for a single biomarker."""
        # Extract values and dates
        values = []
        dates = []
        for m in data:
            val = m.get(biomarker)
            if val is not None:
                values.append(float(val))
                dates.append(datetime.strptime(m['date'], '%Y-%m-%d'))
        
        if len(values) < 2:
            return {
                "available": False,
                "trend": "insufficient_data",
                "risk_contribution": None
            }
        
        # Convert dates to days from first measurement
        days = np.array([(d - dates[0]).days for d in dates], dtype=float)
        vals = np.array(values, dtype=float)
        
        if days[-1] == 0:
            return {
                "available": True,
                "trend": "stable",
                "slope_per_year": 0.0,
                "latest_value": float(vals[-1]),
                "change_percent": 0.0,
                "risk_contribution": 0.0
            }
        
        # Linear regression
        coefficients = np.polyfit(days, vals, 1)
        slope_per_day = coefficients[0]
        slope_per_year = slope_per_day * 365.25
        
        # Percentage change from first to last
        change_pct = ((vals[-1] - vals[0]) / abs(vals[0]) * 100) if vals[0] != 0 else 0
        
        # Classify trend
        if lower_is_worse:
            # For eGFR: declining is bad
            if slope_per_year <= -5:
                trend = "rapid_decline"
                risk_score = 90.0
            elif slope_per_year <= -3:
                trend = "moderate_decline"
                risk_score = 60.0
            elif slope_per_year <= -1:
                trend = "slow_decline"
                risk_score = 30.0
            elif slope_per_year >= 1:
                trend = "improving"
                risk_score = 5.0
            else:
                trend = "stable"
                risk_score = 10.0
        else:
            # For UACR, HbA1c, BP: increasing is bad
            if slope_per_year >= 50 and biomarker == 'uacr':
                trend = "rapid_increase"
                risk_score = 85.0
            elif slope_per_year >= 0.5 and biomarker == 'hba1c':
                trend = "increasing"
                risk_score = 70.0
            elif slope_per_year >= 5 and biomarker == 'bp_systolic':
                trend = "increasing"
                risk_score = 60.0
            elif slope_per_year > 0:
                trend = "slight_increase"
                risk_score = 25.0
            elif slope_per_year < 0:
                trend = "improving"
                risk_score = 5.0
            else:
                trend = "stable"
                risk_score = 10.0
        
        return {
            "available": True,
            "trend": trend,
            "slope_per_year": round(float(slope_per_year), 2),
            "latest_value": float(vals[-1]),
            "earliest_value": float(vals[0]),
            "change_percent": round(float(change_pct), 1),
            "measurements_count": len(values),
            "risk_contribution": risk_score
        }
    
    def _predict_timeline(
        self,
        data: List[Dict],
        egfr_trend: Dict[str, Any]
    ) -> Optional[str]:
        """Predict timeline to critical stages."""
        if not egfr_trend.get('available') or egfr_trend.get('trend') in ['stable', 'improving', 'insufficient_data']:
            return None
        
        slope = egfr_trend.get('slope_per_year', 0)
        latest_egfr = egfr_trend.get('latest_value', 90)
        
        if slope >= 0 or latest_egfr <= 15:
            return None
        
        years_to_esrd = (latest_egfr - 15) / abs(slope)
        
        if years_to_esrd < 1:
            return f"[WARN] خطر الوصول لفشل كلوي خلال أقل من سنة ({years_to_esrd:.1f} سنة)"
        elif years_to_esrd < 3:
            return f" متوقع الوصول لفشل كلوي خلال {years_to_esrd:.1f} سنة"
        elif years_to_esrd < 5:
            return f" متوقع الوصول لمرحلة حرجة خلال {years_to_esrd:.1f} سنة"
        else:
            return f" معدل التدهور الحالي: وصول لمرحلة حرجة خلال ~{years_to_esrd:.0f} سنة"
    
    def _generate_predictive_alert(
        self,
        classification: str,
        score: float,
        trends: Dict[str, Dict],
        timeline: Optional[str]
    ) -> str:
        """Generate human-readable predictive alert in Arabic."""
        messages = {
            "critical": f" تحذير حرج: مؤشرات التنبؤ تشير لتدهور سريع (درجة الخطر: {score:.0f}/100)",
            "high": f" مستوى خطر مرتفع: عدة مؤشرات حيوية في تدهور (درجة الخطر: {score:.0f}/100)",
            "moderate": f" مستوى خطر متوسط: بعض المؤشرات تحتاج متابعة (درجة الخطر: {score:.0f}/100)",
            "low": f" مستوى خطر منخفض: تغيرات طفيفة ملحوظة (درجة الخطر: {score:.0f}/100)",
            "stable": f"[OK] مستقر: المؤشرات الحيوية في المعدل الطبيعي (درجة الخطر: {score:.0f}/100)",
            "insufficient_data": "[WARN] بيانات غير كافية للتحليل التنبؤي"
        }
        
        msg = messages.get(classification, messages["insufficient_data"])
        
        # Add worsening biomarkers
        worsening = []
        for name, trend in trends.items():
            if trend.get('trend') in ['rapid_decline', 'moderate_decline', 'rapid_increase', 'increasing']:
                worsening.append(self._feature_name_ar(name))
        
        if worsening:
            msg += f"\nالمؤشرات المتدهورة: {', '.join(worsening)}"
        
        if timeline:
            msg += f"\n{timeline}"
        
        return msg
    
    # =========================================================================
    # 3. NLP Symptom Analysis
    # =========================================================================
    
    def analyze_symptoms(
        self,
        text: str,
        patient_id: str = None
    ) -> SymptomAnalysis:
        """
        Analyze patient symptoms using NLP.
        
        Uses Gemini (if available) to understand symptoms in Arabic or English,
        and correlates with patient's lab data if patient_id is provided.
        Falls back to keyword matching when Gemini is unavailable.
        
        تحليل شكوى المريض باللغة العربية أو الإنجليزية.
        
        Args:
            text: Patient's symptom description
            patient_id: Optional patient ID to correlate with lab data
            
        Returns:
            SymptomAnalysis with urgency, conditions, and recommendations
        """
        # Get patient context if available
        patient_context = None
        if patient_id:
            patient_context = self._get_patient_context(patient_id)
        
        # Try Gemini first
        if self.gemini_rag and hasattr(self.gemini_rag, 'model') and self.gemini_rag.model:
            return self._analyze_symptoms_gemini(text, patient_context)
        
        # Fallback to keyword matching
        return self._analyze_symptoms_keywords(text, patient_context)
    
    def _analyze_symptoms_gemini(
        self,
        text: str,
        patient_context: Optional[Dict]
    ) -> SymptomAnalysis:
        """Analyze symptoms using Gemini LLM."""
        context_str = ""
        if patient_context:
            context_str = f"""
بيانات المريض الأخيرة:
- eGFR: {patient_context.get('egfr', 'غير متاح')} mL/min/1.73m²
- كرياتينين: {patient_context.get('creatinine', 'غير متاح')} mg/dL
- UACR: {patient_context.get('uacr', 'غير متاح')} mg/g
- HbA1c: {patient_context.get('hba1c', 'غير متاح')}%
- اتجاه eGFR: {patient_context.get('egfr_trend', 'غير متاح')}
"""
        
        prompt = f"""أنت طبيب كلى متخصص. المريض يصف الأعراض التالية:

"{text}"

{context_str}

حلل هذه الأعراض وأجب بالتنسيق التالي بالضبط:
مستوى_الطوارئ: [emergency/urgent/routine]
الحالات_المحتملة: [قائمة مفصولة بفاصلة]
التوصيات: [قائمة مرقمة]
ارتباط_بالتحاليل: [تحليل الارتباط مع نتائج المريض إن وجدت]"""

        try:
            response = self.gemini_rag.model.generate_content(prompt)
            ai_text = response.text
            
            # Parse response
            urgency = "routine"
            if "emergency" in ai_text.lower() or "طوارئ" in ai_text:
                urgency = "emergency"
            elif "urgent" in ai_text.lower() or "عاجل" in ai_text:
                urgency = "urgent"
            
            return SymptomAnalysis(
                urgency=urgency,
                matched_conditions=[{"name": "تم التحليل بواسطة AI", "source": "Gemini"}],
                recommendations=["راجع الطبيب لتأكيد التشخيص والعلاج"],
                correlation_with_labs=context_str if patient_context else None,
                raw_ai_response=ai_text
            )
            
        except Exception as e:
            # Fall back to keywords
            return self._analyze_symptoms_keywords(text, patient_context)
    
    def _analyze_symptoms_keywords(
        self,
        text: str,
        patient_context: Optional[Dict]
    ) -> SymptomAnalysis:
        """Fallback: analyze symptoms using keyword matching."""
        text_lower = text.lower()
        
        # Check each urgency level
        matched_urgency = "routine"
        matched_conditions = []
        
        for urgency_level in ["emergency", "urgent", "routine"]:
            category = SYMPTOM_KEYWORDS[urgency_level]
            all_keywords = category["keywords_ar"] + category["keywords_en"]
            
            for keyword in all_keywords:
                if keyword.lower() in text_lower or keyword in text:
                    matched_urgency = urgency_level
                    matched_conditions = category["conditions"]
                    break
            
            if matched_urgency != "routine" or matched_conditions:
                break
        
        # Generate recommendations based on urgency
        recommendations = {
            "emergency": [
                " توجه للطوارئ فوراً",
                "اتصل بطبيبك المعالج بشكل عاجل",
                "لا تنتظر — الأعراض تشير لحالة حرجة"
            ],
            "urgent": [
                " تواصل مع طبيبك في أقرب وقت",
                "راقب الأعراض بدقة واكتب تفاصيلها",
                "اذهب للمستشفى إذا زادت الأعراض"
            ],
            "routine": [
                " احجز موعد مع الطبيب للمتابعة",
                "استمر في أدويتك بانتظام",
                "اشرب كمية كافية من المياه"
            ]
        }
        
        # Correlate with labs if available
        correlation = None
        if patient_context:
            egfr = patient_context.get('egfr')
            if egfr and egfr < 30:
                correlation = (
                    f"[WARN] تنبيه: الأعراض المذكورة قد تكون مرتبطة بانخفاض وظائف الكلى "
                    f"(eGFR = {egfr:.1f}). يجب مراجعة الطبيب."
                )
            elif egfr and egfr < 60:
                correlation = (
                    f" ملاحظة: وظائف الكلى منخفضة (eGFR = {egfr:.1f}). "
                    f"الأعراض المذكورة قد تكون مؤشر على تطور المرض."
                )
        
        return SymptomAnalysis(
            urgency=matched_urgency,
            matched_conditions=matched_conditions,
            recommendations=recommendations.get(matched_urgency, recommendations["routine"]),
            correlation_with_labs=correlation,
            raw_ai_response=None
        )
    
    def _get_patient_context(self, patient_id: str) -> Optional[Dict]:
        """Get latest patient data for context."""
        data = self.monitor.get_patient_history(patient_id)
        if not data:
            return None
        
        latest = data[-1]
        context = {
            'egfr': latest.get('egfr'),
            'creatinine': latest.get('creatinine'),
            'uacr': latest.get('uacr'),
            'hba1c': latest.get('hba1c'),
        }
        
        # Add trend info if available
        if len(data) >= 2:
            trend = self.monitor.calculate_trend(patient_id)
            context['egfr_trend'] = f"{trend.egfr_slope:+.1f} mL/min/year"
            context['is_fast_progressor'] = trend.is_fast_progressor
        
        return context
    
    # =========================================================================
    # 4. Alert Aggregation
    # =========================================================================
    
    def generate_smart_alerts(
        self,
        patient_id: str,
        include_rule_based: bool = True,
        rule_based_alerts: List[str] = None
    ) -> List[SmartAlert]:
        """
        Generate all smart alerts for a patient.
        
        Combines anomaly detection, predictive analytics, and existing
        rule-based alerts into a unified, priority-sorted list.
        
        Args:
            patient_id: Unique patient identifier
            include_rule_based: Include existing rule-based alerts
            rule_based_alerts: Pre-computed rule-based alerts (optional)
            
        Returns:
            List of SmartAlert sorted by priority (CRITICAL first)
        """
        alerts = []
        now = datetime.now().isoformat()
        
        # 1. Anomaly Detection Alerts
        try:
            anomaly = self.detect_anomalies(patient_id)
            if anomaly.is_anomaly:
                priority = (
                    AlertPriority.CRITICAL if anomaly.severity == "high"
                    else AlertPriority.WARNING if anomaly.severity == "medium"
                    else AlertPriority.INFO
                )
                
                feature_names = [f['feature_ar'] for f in anomaly.anomalous_features[:3]]
                details_text = ", ".join(feature_names) if feature_names else "تغيرات غير طبيعية"
                
                alerts.append(SmartAlert(
                    alert_type=AlertType.ANOMALY.value,
                    priority=priority.value,
                    title=" اكتشاف تغيرات غير طبيعية",
                    message=f"تم اكتشاف تغيرات مفاجئة في: {details_text}. "
                            f"هذه القيم تختلف بشكل ملحوظ عن المعدل الطبيعي للمريض.",
                    details={
                        "anomaly_score": anomaly.anomaly_score,
                        "severity": anomaly.severity,
                        "anomalous_features": anomaly.anomalous_features
                    },
                    timestamp=now
                ))
        except Exception:
            pass  # Don't let anomaly detection failure break everything
        
        # 2. Predictive Analytics Alerts
        try:
            prediction = self.predict_future_risk(patient_id)
            if prediction.risk_classification not in ["stable", "insufficient_data"]:
                priority = (
                    AlertPriority.CRITICAL if prediction.risk_classification == "critical"
                    else AlertPriority.WARNING if prediction.risk_classification in ["high", "moderate"]
                    else AlertPriority.INFO
                )
                
                alerts.append(SmartAlert(
                    alert_type=AlertType.PREDICTIVE.value,
                    priority=priority.value,
                    title=" تحليل تنبؤي",
                    message=prediction.alert_message,
                    details={
                        "risk_score": prediction.overall_risk_score,
                        "classification": prediction.risk_classification,
                        "timeline": prediction.predicted_timeline,
                        "trends": prediction.biomarker_trends
                    },
                    timestamp=now
                ))
        except Exception:
            pass
        
        # 3. Include rule-based alerts
        if include_rule_based and rule_based_alerts:
            for rule_alert in rule_based_alerts:
                priority = AlertPriority.CRITICAL if "" in rule_alert else (
                    AlertPriority.WARNING if "" in rule_alert or "[WARN]" in rule_alert
                    else AlertPriority.INFO
                )
                
                alerts.append(SmartAlert(
                    alert_type=AlertType.RULE_BASED.value,
                    priority=priority.value,
                    title=" تنبيه طبي",
                    message=rule_alert,
                    details={},
                    timestamp=now
                ))
        
        # Sort by priority (CRITICAL > WARNING > INFO)
        priority_order = {
            AlertPriority.CRITICAL.value: 0,
            AlertPriority.WARNING.value: 1,
            AlertPriority.INFO.value: 2
        }
        alerts.sort(key=lambda a: priority_order.get(a.priority, 3))
        
        return alerts
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    @staticmethod
    def _feature_name_ar(feature: str) -> str:
        """Get Arabic name for a biomarker feature."""
        names = {
            'egfr': 'معدل الترشيح الكلوي (eGFR)',
            'creatinine': 'الكرياتينين',
            'uacr': 'نسبة الألبومين للكرياتينين (UACR)',
            'hba1c': 'السكر التراكمي (HbA1c)',
            'bp_systolic': 'ضغط الدم الانقباضي',
            'bp_diastolic': 'ضغط الدم الانبساطي',
            'hemoglobin': 'الهيموجلوبين',
            'potassium': 'البوتاسيوم',
            'uric_acid': 'حمض اليوريك',
        }
        return names.get(feature, feature)
    
    def alerts_to_dict(self, alerts: List[SmartAlert]) -> List[Dict]:
        """Convert alerts to serializable dictionaries."""
        return [asdict(alert) for alert in alerts]
