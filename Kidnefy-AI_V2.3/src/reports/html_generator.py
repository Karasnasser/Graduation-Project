"""
HTML Report Generator — بديل عربي كامل لـ PDF
Arabic RTL is handled natively by the browser via CSS `dir='rtl'`.
The backend can serve this as a file download or convert via browser print.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class PatientInfo:
    name: str
    age: int
    sex: str
    date: str
    lab_no: str = ""
    doctor_name: str = ""


@dataclass
class TestResult:
    name: str
    value: float
    unit: str
    reference_range: str
    is_abnormal: bool


STAGE_COLORS = {
    'G1': '#43A047', 'G2': '#8BC34A', 'G3a': '#FFA726',
    'G3b': '#F4511E', 'G4': '#E53935', 'G5': '#7B1FA2',
}

RISK_COLORS = {
    'Low Risk': '#43A047',
    'Moderate Risk': '#FFA726',
    'High Risk': '#F4511E',
    'Very High Risk': '#B71C1C',
    'Critical': '#6A1B9A',
}


def _stage_bar(current: str) -> str:
    stages = ['G1', 'G2', 'G3a', 'G3b', 'G4', 'G5']
    cells = ''
    for s in stages:
        bg = STAGE_COLORS.get(s, '#999')
        border = '3px solid #000' if s == current else '1px solid rgba(255,255,255,0.3)'
        cells += (f'<div style="background:{bg};border:{border};padding:8px 4px;'
                  f'text-align:center;color:#fff;font-weight:bold;font-size:13px;'
                  f'flex:1;border-radius:4px;">{s}</div>')
    return f'<div style="display:flex;gap:4px;margin:12px 0;">{cells}</div>'


def generate_html_report(
    patient: PatientInfo,
    prediction: bool,
    probability: float,
    risk_level: str,
    gfr_stage: str,
    egfr: float,
    alb_category: str = None,
    acr: float = None,
    lab_results: List[TestResult] = None,
    recommendations: List[str] = None,
    alerts: List[str] = None,
    output_dir: str = "generated_reports",
    filename: str = None,
) -> str:
    """Generate a bilingual Arabic/English HTML medical report."""

    if not filename:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe = patient.name.replace(' ', '_')[:20]
        filename = f"kidney_report_{safe}_{ts}.html"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result_color = '#B71C1C' if prediction else '#2E7D32'
    result_bg    = '#FFEBEE' if prediction else '#E8F5E9'
    result_icon  = '⚠' if prediction else '✓'
    result_en    = 'CKD DETECTED' if prediction else 'NO CKD DETECTED'
    result_ar    = 'تم اكتشاف مرض الكلى المزمن' if prediction else 'لا يوجد مرض الكلى المزمن'
    pct          = int(probability * 100)
    risk_color   = RISK_COLORS.get(risk_level, '#F4511E')
    stage_color  = STAGE_COLORS.get(gfr_stage, '#1565C0')

    # ── Alerts HTML ────────────────────────────────────────────────────────
    alerts_html = ''
    if alerts:
        items = ''.join(
            f'<div style="background:#FFEBEE;border:1px solid #E53935;border-radius:6px;'
            f'padding:10px 14px;margin:6px 0;color:#B71C1C;font-size:14px;">'
            f'⚠ {a}</div>' for a in alerts
        )
        alerts_html = f'''
        <div class="section">
          <div class="section-header">
            <span>⚠ Clinical Alerts</span>
            <span>تنبيهات طبية</span>
          </div>
          {items}
        </div>'''

    # ── Recommendations HTML ───────────────────────────────────────────────
    recs_html = ''
    if recommendations:
        items = ''.join(
            f'<div style="display:flex;gap:12px;align-items:flex-start;'
            f'padding:10px 0;border-bottom:1px solid #ECEFF1;">'
            f'<span style="background:#1565C0;color:#fff;min-width:28px;height:28px;'
            f'border-radius:50%;display:flex;align-items:center;justify-content:center;'
            f'font-weight:bold;font-size:13px;">{i+1}</span>'
            f'<span style="font-size:14px;line-height:1.6;">{r}</span></div>'
            for i, r in enumerate(recommendations)
        )
        recs_html = f'''
        <div class="section">
          <div class="section-header">
            <span>Recommendations</span>
            <span>التوصيات الطبية</span>
          </div>
          {items}
        </div>'''

    # ── Lab Results HTML ───────────────────────────────────────────────────
    labs_html = ''
    if lab_results:
        rows = ''.join(
            f'<tr style="background:{"#FFEBEE" if r.is_abnormal else "transparent"};">'
            f'<td>{r.name}</td><td style="text-align:center;">{r.value}</td>'
            f'<td style="text-align:center;">{r.unit}</td>'
            f'<td style="text-align:center;">{r.reference_range}</td>'
            f'<td style="text-align:center;font-weight:bold;'
            f'color:{"#B71C1C" if r.is_abnormal else "#2E7D32"};">'
            f'{"✗ Abnormal" if r.is_abnormal else "✓ Normal"}</td></tr>'
            for r in lab_results
        )
        labs_html = f'''
        <div class="section">
          <div class="section-header">
            <span>Lab Results</span><span>نتائج التحاليل</span>
          </div>
          <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead>
              <tr style="background:#1565C0;color:#fff;">
                <th style="padding:10px;text-align:left;">Test</th>
                <th style="padding:10px;">Result</th>
                <th style="padding:10px;">Unit</th>
                <th style="padding:10px;">Ref. Range</th>
                <th style="padding:10px;">Status</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>'''

    # ── ACR row ────────────────────────────────────────────────────────────
    acr_row = ''
    if alb_category and acr:
        acr_row = f'''
        <tr>
          <td style="background:#ECEFF1;padding:8px 12px;font-weight:600;">ACR / نسبة الألبومين</td>
          <td style="padding:8px 12px;">{acr} mg/g</td>
          <td style="background:#ECEFF1;padding:8px 12px;font-weight:600;">Albuminuria Category / فئة البروتين</td>
          <td style="padding:8px 12px;">{alb_category}</td>
        </tr>'''

    html = f'''<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kidney Disease Report — {patient.name}</title>
  <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Cairo', 'Inter', sans-serif;
      background: #F5F7FA;
      color: #263238;
      padding: 24px;
    }}
    .page {{
      max-width: 900px;
      margin: 0 auto;
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.10);
      overflow: hidden;
    }}
    .page-header {{
      background: linear-gradient(135deg, #1565C0 0%, #1976D2 100%);
      color: #fff;
      padding: 28px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .page-header h1 {{ font-size: 22px; font-weight: 700; }}
    .page-header .ar-title {{ font-size: 18px; font-weight: 700; opacity: 0.9; }}
    .page-header .meta {{ font-size: 12px; opacity: 0.75; margin-top: 4px; }}
    .body {{ padding: 28px 32px; }}
    .section {{ margin-bottom: 24px; }}
    .section-header {{
      background: #1565C0;
      color: #fff;
      padding: 10px 16px;
      border-radius: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 700;
      font-size: 14px;
      margin-bottom: 12px;
    }}
    table.info-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    table.info-table td {{ padding: 9px 12px; border: 1px solid #ECEFF1; }}
    table.info-table td:nth-child(odd) {{ background: #ECEFF1; font-weight: 600; }}
    .result-box {{
      background: {result_bg};
      border: 2px solid {result_color};
      border-radius: 8px;
      padding: 16px 20px;
      display: flex;
      align-items: center;
      gap: 16px;
      margin-bottom: 14px;
    }}
    .result-icon {{ font-size: 32px; color: {result_color}; }}
    .result-text-en {{ font-size: 18px; font-weight: 700; color: {result_color}; }}
    .result-text-ar {{ font-size: 16px; font-weight: 600; color: {result_color}; }}
    .risk-badge {{
      display: inline-block;
      background: {risk_color};
      color: #fff;
      padding: 4px 14px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 600;
      margin-top: 6px;
    }}
    .prob-bar-wrap {{
      background: #ECEFF1;
      border-radius: 6px;
      height: 12px;
      margin: 10px 0;
      overflow: hidden;
    }}
    .prob-bar-fill {{
      height: 100%;
      width: {pct}%;
      background: {result_color};
      border-radius: 6px;
      transition: width 0.8s ease;
    }}
    .footer {{
      text-align: center;
      font-size: 11px;
      color: #90A4AE;
      padding: 18px 32px;
      border-top: 1px solid #ECEFF1;
    }}
    @media print {{
      body {{ background: #fff; padding: 0; }}
      .page {{ box-shadow: none; border-radius: 0; }}
    }}
  </style>
</head>
<body>
  <div class="page">

    <!-- Header -->
    <div class="page-header">
      <div>
        <div style="font-size:28px;">⚕</div>
        <h1>Kidney Disease Prediction Report</h1>
        <div class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
      </div>
      <div class="ar-title">تقرير التنبؤ بأمراض الكلى</div>
    </div>

    <div class="body">

      <!-- Patient Info -->
      <div class="section">
        <div class="section-header">
          <span>Patient Information</span>
          <span>بيانات المريض</span>
        </div>
        <table class="info-table">
          <tr>
            <td>Patient Name / الاسم</td><td>{patient.name}</td>
            <td>Date / التاريخ</td><td>{patient.date}</td>
          </tr>
          <tr>
            <td>Age / العمر</td><td>{patient.age} سنة</td>
            <td>Sex / الجنس</td><td>{patient.sex}</td>
          </tr>
          <tr>
            <td>Lab No. / رقم المعمل</td><td>{patient.lab_no or '—'}</td>
            <td>Doctor / الطبيب</td><td>{patient.doctor_name or '—'}</td>
          </tr>
        </table>
      </div>

      <!-- Prediction -->
      <div class="section">
        <div class="section-header">
          <span>Prediction Result</span><span>نتيجة التنبؤ</span>
        </div>
        <div class="result-box">
          <div class="result-icon">{result_icon}</div>
          <div>
            <div class="result-text-en">{result_en}</div>
            <div class="result-text-ar">{result_ar}</div>
            <div class="risk-badge">{risk_level}</div>
          </div>
          <div style="margin-right:auto;text-align:left;">
            <div style="font-size:13px;color:#546E7A;">Risk Probability / نسبة الاحتمال</div>
            <div style="font-size:28px;font-weight:700;color:{result_color};">{pct}%</div>
            <div class="prob-bar-wrap"><div class="prob-bar-fill"></div></div>
          </div>
        </div>
      </div>

      <!-- Staging -->
      <div class="section">
        <div class="section-header">
          <span>CKD Staging (KDIGO)</span><span>مراحل الفشل الكلوي</span>
        </div>
        {_stage_bar(gfr_stage)}
        <table class="info-table" style="margin-top:8px;">
          <tr>
            <td>eGFR Value / معدل الترشيح</td>
            <td style="font-weight:700;">{egfr:.1f} mL/min/1.73m²</td>
            <td>Current Stage / المرحلة الحالية</td>
            <td style="font-weight:700;color:{stage_color};">{gfr_stage}</td>
          </tr>
          {acr_row}
        </table>
      </div>

      {labs_html}
      {alerts_html}
      {recs_html}

    </div>

    <!-- Footer -->
    <div class="footer">
      AI-Powered Kidney Disease Prediction System &nbsp;|&nbsp;
      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
      This report is NOT a substitute for professional medical advice.
      هذا التقرير لا يُغني عن استشارة الطبيب المختص.
    </div>

  </div>
</body>
</html>'''

    filepath = out_path / filename
    filepath.write_text(html, encoding='utf-8')
    print(f"[OK] HTML report generated: {filepath}")
    return str(filepath)


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = PatientInfo(name="محمد أحمد", age=70, sex="Male",
                    date=datetime.now().strftime('%Y-%m-%d'),
                    lab_no="1806", doctor_name="د. ولاء")

    labs = [
        TestResult("Serum Creatinine", 2.3,  "mg/dL", "0.5–1.5", True),
        TestResult("Blood Urea",        61,   "mg/dL", "10–50",   True),
        TestResult("eGFR",              28.5, "mL/min/1.73m²", ">90", True),
        TestResult("ACR",               44.4, "mg/g",  "<30",    True),
    ]
    recs = [
        "مراجعة دورية مع طبيب الكلى كل شهر",
        "الحفاظ على ضغط الدم أقل من 130/80 mmHg",
        "التحكم في مستوى السكر (HbA1c < 7%)",
        "تقليل الملح والبروتين في الطعام",
        "شرب 1.5–2 لتر ماء يومياً",
    ]
    alrts = [
        "وظائف الكلى منخفضة جداً — المرحلة G4",
        "كرياتينين مرتفع — خطر الفشل الكلوي",
    ]
    path = generate_html_report(
        patient=p, prediction=True, probability=0.85,
        risk_level="Very High Risk", gfr_stage="G4", egfr=28.5,
        alb_category="A2", acr=44.4, lab_results=labs,
        recommendations=recs, alerts=alrts,
    )
    print(f"Open in browser: {path}")
