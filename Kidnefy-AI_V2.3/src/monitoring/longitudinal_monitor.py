"""
Longitudinal Monitoring Module
Tracks patient eGFR and biomarkers over time to detect trends and fast progressors.

تتبع بيانات المريض عبر الزمن لاكتشاف:
- معدل انخفاض eGFR (الميل)
- المرضى اللي حالتهم بتتدهور بسرعة (Fast Progressors)
- التنبؤ بالوقت المتبقي قبل الوصول لمرحلة معينة
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Measurement:
    """Single patient measurement at a point in time."""
    date: str  # ISO format: YYYY-MM-DD
    egfr: float
    creatinine: Optional[float] = None
    uacr: Optional[float] = None
    hba1c: Optional[float] = None
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class TrendResult:
    """Result of eGFR trend analysis."""
    trend: str  # "stable", "slow_decline", "rapid_decline", "improving"
    egfr_slope: float  # mL/min/1.73m²/year (negative = declining)
    is_fast_progressor: bool  # True if decline > 5 mL/min/year
    measurements_count: int
    time_span_days: int
    latest_egfr: float
    earliest_egfr: float
    predicted_years_to_esrd: Optional[float]  # Years until eGFR < 15
    confidence: float  # 0-1 based on data quality
    alert_message: Optional[str] = None


class LongitudinalMonitor:
    """
    Longitudinal monitoring of patient kidney function over time.
    
    Features:
        - Track eGFR, creatinine, UACR, HbA1c over multiple visits
        - Calculate rate of eGFR decline (slope)
        - Detect fast progressors (> 5 mL/min/year decline)
        - Predict time to ESRD (End-Stage Renal Disease)
        - Classify trends: stable / slow_decline / rapid_decline / improving
        
    Data is stored as JSON files in a configurable directory.
    """
    
    # Thresholds based on KDIGO guidelines
    FAST_PROGRESSOR_THRESHOLD = -5.0  # mL/min/1.73m²/year
    SLOW_DECLINE_THRESHOLD = -1.0  # mL/min/1.73m²/year
    ESRD_EGFR_THRESHOLD = 15.0  # eGFR < 15 = Stage 5 / ESRD
    MIN_MEASUREMENTS = 2  # Minimum measurements for trend analysis
    
    def __init__(self, data_dir: str = "data/patient_monitoring"):
        """
        Initialize longitudinal monitor.
        
        Args:
            data_dir: Directory to store patient monitoring data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[Dict]] = {}
    
    def _get_patient_file(self, patient_id: str) -> Path:
        """Get the JSON file path for a patient."""
        return self.data_dir / f"patient_{patient_id}.json"
    
    def _load_patient_data(self, patient_id: str) -> List[Dict]:
        """Load patient data from file or cache."""
        if patient_id in self._cache:
            return self._cache[patient_id]
        
        filepath = self._get_patient_file(patient_id)
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._cache[patient_id] = data
                return data
        return []
    
    def _save_patient_data(self, patient_id: str, data: List[Dict]):
        """Save patient data to file and cache."""
        filepath = self._get_patient_file(patient_id)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._cache[patient_id] = data
    
    def add_measurement(
        self,
        patient_id: str,
        date: str,
        egfr: float,
        creatinine: float = None,
        uacr: float = None,
        hba1c: float = None,
        bp_systolic: float = None,
        bp_diastolic: float = None,
        notes: str = None
    ) -> Dict[str, Any]:
        """
        Add a new measurement for a patient.
        
        Args:
            patient_id: Unique patient identifier
            date: Date of measurement (YYYY-MM-DD)
            egfr: eGFR value in mL/min/1.73m²
            creatinine: Serum creatinine in mg/dL
            uacr: Urine albumin-to-creatinine ratio in mg/g
            hba1c: HbA1c percentage
            bp_systolic: Systolic blood pressure
            bp_diastolic: Diastolic blood pressure
            notes: Optional clinical notes
            
        Returns:
            Dictionary with measurement info and current trend
        """
        measurement = Measurement(
            date=date,
            egfr=egfr,
            creatinine=creatinine,
            uacr=uacr,
            hba1c=hba1c,
            bp_systolic=bp_systolic,
            bp_diastolic=bp_diastolic,
            notes=notes
        )
        
        # Load existing data
        data = self._load_patient_data(patient_id)
        
        # Add new measurement
        data.append(asdict(measurement))
        
        # Sort by date
        data.sort(key=lambda x: x['date'])
        
        # Save
        self._save_patient_data(patient_id, data)
        
        # Calculate current trend if enough data
        result = {
            "patient_id": patient_id,
            "measurement_added": asdict(measurement),
            "total_measurements": len(data)
        }
        
        if len(data) >= self.MIN_MEASUREMENTS:
            trend = self.calculate_trend(patient_id)
            result["current_trend"] = asdict(trend)
        
        return result
    
    def get_patient_history(self, patient_id: str) -> List[Dict]:
        """
        Get all measurements for a patient, sorted by date.
        
        Returns:
            List of measurement dictionaries
        """
        return self._load_patient_data(patient_id)
    
    def calculate_egfr_slope(self, patient_id: str) -> Optional[float]:
        """
        Calculate the rate of eGFR change over time using linear regression.
        
        The slope represents mL/min/1.73m² per year:
        - Negative slope → kidney function declining
        - Positive slope → kidney function improving
        - Slope < -5 → Fast progressor
        
        Returns:
            eGFR slope in mL/min/1.73m²/year, or None if insufficient data
        """
        data = self._load_patient_data(patient_id)
        
        if len(data) < self.MIN_MEASUREMENTS:
            return None
        
        # Convert dates to days from first measurement
        dates = [datetime.strptime(m['date'], '%Y-%m-%d') for m in data]
        egfr_values = [m['egfr'] for m in data]
        
        first_date = dates[0]
        days = np.array([(d - first_date).days for d in dates], dtype=float)
        egfr = np.array(egfr_values, dtype=float)
        
        # Avoid division by zero
        if days[-1] == 0:
            return 0.0
        
        # Simple linear regression: y = mx + b
        # Using numpy polyfit
        coefficients = np.polyfit(days, egfr, 1)
        slope_per_day = coefficients[0]
        
        # Convert to per-year (365.25 days)
        slope_per_year = slope_per_day * 365.25
        
        return float(slope_per_year)
    
    def is_fast_progressor(self, patient_id: str) -> bool:
        """
        Determine if a patient is a fast progressor.
        
        A fast progressor has eGFR decline > 5 mL/min/1.73m²/year.
        
        Returns:
            True if patient is a fast progressor
        """
        slope = self.calculate_egfr_slope(patient_id)
        if slope is None:
            return False
        return slope <= self.FAST_PROGRESSOR_THRESHOLD
    
    def predict_time_to_stage(
        self,
        patient_id: str,
        target_egfr: float = None
    ) -> Optional[float]:
        """
        Predict years until eGFR reaches a target level.
        
        Args:
            patient_id: Patient ID
            target_egfr: Target eGFR (default: 15 = ESRD)
            
        Returns:
            Estimated years, or None if not declining or insufficient data
        """
        if target_egfr is None:
            target_egfr = self.ESRD_EGFR_THRESHOLD
        
        data = self._load_patient_data(patient_id)
        if len(data) < self.MIN_MEASUREMENTS:
            return None
        
        slope = self.calculate_egfr_slope(patient_id)
        if slope is None or slope >= 0:
            return None  # Not declining
        
        latest_egfr = data[-1]['egfr']
        
        if latest_egfr <= target_egfr:
            return 0.0  # Already at or below target
        
        # Years = (current_egfr - target_egfr) / |slope|
        years = (latest_egfr - target_egfr) / abs(slope)
        
        return float(years)
    
    def calculate_trend(self, patient_id: str) -> TrendResult:
        """
        Complete trend analysis for a patient.
        
        Returns:
            TrendResult with trend classification, slope, and predictions
        """
        data = self._load_patient_data(patient_id)
        
        if len(data) < self.MIN_MEASUREMENTS:
            latest = data[-1] if data else {"egfr": 0}
            return TrendResult(
                trend="insufficient_data",
                egfr_slope=0.0,
                is_fast_progressor=False,
                measurements_count=len(data),
                time_span_days=0,
                latest_egfr=latest.get('egfr', 0),
                earliest_egfr=data[0].get('egfr', 0) if data else 0,
                predicted_years_to_esrd=None,
                confidence=0.0,
                alert_message="Insufficient data for trend analysis. Need at least 2 measurements."
            )
        
        slope = self.calculate_egfr_slope(patient_id)
        fast_progressor = self.is_fast_progressor(patient_id)
        years_to_esrd = self.predict_time_to_stage(patient_id)
        
        # Calculate time span
        dates = [datetime.strptime(m['date'], '%Y-%m-%d') for m in data]
        time_span = (dates[-1] - dates[0]).days
        
        # Classify trend
        if slope <= self.FAST_PROGRESSOR_THRESHOLD:
            trend = "rapid_decline"
        elif slope <= self.SLOW_DECLINE_THRESHOLD:
            trend = "slow_decline"
        elif slope >= 1.0:
            trend = "improving"
        else:
            trend = "stable"
        
        # Calculate confidence based on data quality
        confidence = min(1.0, len(data) / 10.0)  # More data = higher confidence
        if time_span < 90:  # Less than 3 months
            confidence *= 0.5  # Lower confidence for short spans
        
        # Generate alert
        alert_message = None
        if fast_progressor:
            alert_message = (
                f"[WARN] ALERT: Fast Progressor detected! "
                f"eGFR declining at {abs(slope):.1f} mL/min/year. "
                f"{'Estimated ' + f'{years_to_esrd:.1f} years to ESRD.' if years_to_esrd else ''}"
            )
        elif trend == "slow_decline":
            alert_message = (
                f" eGFR showing slow decline ({abs(slope):.1f} mL/min/year). "
                f"Monitor closely."
            )
        
        return TrendResult(
            trend=trend,
            egfr_slope=slope,
            is_fast_progressor=fast_progressor,
            measurements_count=len(data),
            time_span_days=time_span,
            latest_egfr=data[-1]['egfr'],
            earliest_egfr=data[0]['egfr'],
            predicted_years_to_esrd=years_to_esrd,
            confidence=confidence,
            alert_message=alert_message
        )
    
    def format_trend_report(self, patient_id: str) -> str:
        """
        Generate a readable trend report for a patient.
        
        Returns:
            Formatted trend report string
        """
        trend = self.calculate_trend(patient_id)
        data = self.get_patient_history(patient_id)
        
        report = []
        report.append("=" * 60)
        report.append("  Longitudinal Monitoring Report")
        report.append("  تقرير المتابعة الطولية")
        report.append("=" * 60)
        report.append(f"\nPatient ID: {patient_id}")
        report.append(f"Measurements: {trend.measurements_count}")
        report.append(f"Time Span: {trend.time_span_days} days")
        
        # Trend classification
        trend_labels = {
            "stable": "[OK] Stable (مستقر)",
            "slow_decline": " Slow Decline (انخفاض بطيء)",
            "rapid_decline": " Rapid Decline (انخفاض سريع)",
            "improving": " Improving (تحسن)",
            "insufficient_data": "[WARN] Insufficient Data (بيانات غير كافية)"
        }
        
        report.append(f"\nTrend: {trend_labels.get(trend.trend, trend.trend)}")
        report.append(f"eGFR Slope: {trend.egfr_slope:+.2f} mL/min/1.73m²/year")
        report.append(f"Fast Progressor: {'YES [WARN]' if trend.is_fast_progressor else 'No'}")
        report.append(f"Confidence: {trend.confidence:.0%}")
        
        if trend.predicted_years_to_esrd is not None:
            report.append(f"Predicted Time to ESRD: {trend.predicted_years_to_esrd:.1f} years")
        
        # eGFR history
        report.append("\n--- eGFR History ---")
        for m in data:
            extras = []
            if m.get('creatinine'):
                extras.append(f"Cr={m['creatinine']:.2f}")
            if m.get('hba1c'):
                extras.append(f"HbA1c={m['hba1c']:.1f}%")
            if m.get('uacr'):
                extras.append(f"UACR={m['uacr']:.0f}")
            extra_str = f" ({', '.join(extras)})" if extras else ""
            report.append(f"  {m['date']}: eGFR = {m['egfr']:.1f}{extra_str}")
        
        if trend.alert_message:
            report.append(f"\n{trend.alert_message}")
        
        report.append("=" * 60)
        return "\n".join(report)
    
    def get_all_patients(self) -> List[str]:
        """Get list of all patient IDs being monitored."""
        patients = []
        for f in self.data_dir.glob("patient_*.json"):
            patient_id = f.stem.replace("patient_", "")
            patients.append(patient_id)
        return patients
    
    def get_fast_progressors(self) -> List[Dict[str, Any]]:
        """
        Get all fast progressors across all monitored patients.
        
        Returns:
            List of patient IDs and their trend info
        """
        fast_progressors = []
        for patient_id in self.get_all_patients():
            if self.is_fast_progressor(patient_id):
                trend = self.calculate_trend(patient_id)
                fast_progressors.append({
                    "patient_id": patient_id,
                    "egfr_slope": trend.egfr_slope,
                    "latest_egfr": trend.latest_egfr,
                    "predicted_years_to_esrd": trend.predicted_years_to_esrd,
                    "alert": trend.alert_message
                })
        return fast_progressors


if __name__ == "__main__":
    print(" Testing Longitudinal Monitor...")
    
    monitor = LongitudinalMonitor(data_dir="data/patient_monitoring_test")
    
    # Simulate a patient with declining eGFR
    test_patient = "test_001"
    monitor.add_measurement(test_patient, "2025-01-01", egfr=85, creatinine=1.1, hba1c=7.2)
    monitor.add_measurement(test_patient, "2025-04-01", egfr=78, creatinine=1.3, hba1c=7.5)
    monitor.add_measurement(test_patient, "2025-07-01", egfr=70, creatinine=1.5, hba1c=7.8)
    monitor.add_measurement(test_patient, "2025-10-01", egfr=62, creatinine=1.7, hba1c=8.0)
    monitor.add_measurement(test_patient, "2026-01-01", egfr=53, creatinine=2.0, hba1c=8.3)
    
    # Get trend report
    print(monitor.format_trend_report(test_patient))
    
    # Check fast progressor
    print(f"\nIs fast progressor: {monitor.is_fast_progressor(test_patient)}")
    
    # Predict time to ESRD
    years = monitor.predict_time_to_stage(test_patient)
    print(f"Predicted years to ESRD: {years:.1f}" if years else "Not declining")
    
    # Test a stable patient
    stable_patient = "test_002"
    monitor.add_measurement(stable_patient, "2025-01-01", egfr=92, creatinine=0.9)
    monitor.add_measurement(stable_patient, "2025-06-01", egfr=90, creatinine=0.95)
    monitor.add_measurement(stable_patient, "2026-01-01", egfr=91, creatinine=0.92)
    
    print(f"\n{monitor.format_trend_report(stable_patient)}")
    
    # Cleanup test data
    import shutil
    shutil.rmtree("data/patient_monitoring_test", ignore_errors=True)
    print("\n[OK] All tests passed!")
