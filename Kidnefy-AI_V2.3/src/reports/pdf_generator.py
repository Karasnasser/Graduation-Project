"""
PDF Report Generator Module
Generate professional, bilingual (Arabic/English) medical reports.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Arabic text support ───────────────────────────────────────────────
try:
    import arabic_reshaper
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False

# ── Arabic font registration ───────────────────────────────────────────
_ARABIC_FONT = "Helvetica"   # fallback (no Arabic)

def _try_register_arabic_font():
    global _ARABIC_FONT
    candidates = [
        ("C:/Windows/Fonts/arial.ttf",   "Arial"),
        ("C:/Windows/Fonts/tahoma.ttf",  "Tahoma"),
        ("C:/Windows/Fonts/calibri.ttf", "Calibri"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans"),
        ("/usr/share/fonts/truetype/fonts-arabeyes/ae_AlArabiya.ttf", "AeAlArabiya"),
    ]
    for path, name in candidates:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                _ARABIC_FONT = name
                print(f"[PDF] Arabic font registered: {name}")
                return
            except Exception:
                continue
    print("[PDF] Warning: No Arabic font found. Arabic text may not render correctly.")
    print("      Install Arial/Tahoma on Windows, or run: pip install arabic-reshaper")

_try_register_arabic_font()


def ar(text: str) -> str:
    """Reshape Arabic text for ReportLab PDF rendering.

    IMPORTANT: We use arabic_reshaper.reshape() ONLY (no bidi reversal).
    - reshape() connects Arabic letters into their correct contextual forms.
    - bidi get_display() reverses character order for visual LTR rendering,
      which causes double-reversal in ReportLab and shows text backwards.
    - Instead we use TA_RIGHT alignment on all Arabic paragraphs.
    """
    if not ARABIC_SUPPORT or not text:
        return text
    try:
        return arabic_reshaper.reshape(text)
    except Exception:
        return text


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


class PDFReportGenerator:
    """Generate professional bilingual PDF medical reports."""

    # ── Palette ────────────────────────────────────────────────────────────
    C = {
        'primary':    colors.HexColor('#1565C0'),
        'primary_lt': colors.HexColor('#E3F2FD'),
        'success':    colors.HexColor('#2E7D32'),
        'success_lt': colors.HexColor('#E8F5E9'),
        'warning':    colors.HexColor('#E65100'),
        'warning_lt': colors.HexColor('#FFF3E0'),
        'danger':     colors.HexColor('#B71C1C'),
        'danger_lt':  colors.HexColor('#FFEBEE'),
        'critical':   colors.HexColor('#6A1B9A'),
        'gray':       colors.HexColor('#546E7A'),
        'gray_lt':    colors.HexColor('#ECEFF1'),
        'white':      colors.white,
        'black':      colors.black,
    }

    # ── Stage colour map ───────────────────────────────────────────────────
    STAGE_COLORS = {
        'G1':  colors.HexColor('#43A047'),
        'G2':  colors.HexColor('#8BC34A'),
        'G3a': colors.HexColor('#FFA726'),
        'G3b': colors.HexColor('#F4511E'),
        'G4':  colors.HexColor('#E53935'),
        'G5':  colors.HexColor('#7B1FA2'),
    }

    def __init__(self, output_dir: str = "generated_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._build_styles()

    # ── Styles ─────────────────────────────────────────────────────────────
    def _build_styles(self):
        def add(name, parent='Normal', **kw):
            if name not in self.styles:
                self.styles.add(ParagraphStyle(name=name,
                                               parent=self.styles[parent],
                                               fontName=_ARABIC_FONT,
                                               **kw))

        add('RTitle',   fontSize=20, textColor=self.C['primary'],
            alignment=TA_CENTER, spaceAfter=4, leading=26)
        add('RSubtitle',fontSize=12, textColor=self.C['gray'],
            alignment=TA_CENTER, spaceAfter=16)
        add('RSection', fontSize=13, textColor=self.C['white'],
            spaceBefore=14, spaceAfter=6, leading=18)
        add('RBody',    fontSize=10, leading=15, spaceAfter=6)
        add('RSmall',   fontSize=9,  textColor=self.C['gray'], leading=13)
        add('RCenter',  fontSize=10, alignment=TA_CENTER, leading=14)
        add('RRight',   fontSize=10, alignment=TA_RIGHT,  leading=14)

    # ── Helpers ────────────────────────────────────────────────────────────
    def _section_header(self, en: str, ar_text: str) -> List:
        """Render a full-width coloured section banner."""
        data = [[Paragraph(f'<b>{en}</b>', self.styles['RSection']),
                 Paragraph(f'<b>{ar(ar_text)}</b>', self.styles['RSection'])]]
        t = Table(data, colWidths=[9*cm, 9*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.C['primary']),
            ('TEXTCOLOR',  (0, 0), (-1, -1), self.C['white']),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING',    (0, 0), (-1, -1), 8),
            ('LEFTPADDING',   (0, 0), (-1, -1), 10),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ('ROUNDEDCORNERS', [4, 4, 4, 4]),
        ]))
        return [t, Spacer(1, 8)]

    def _kv_table(self, rows: List[List], col_w=None) -> Table:
        """Two-column label/value table."""
        col_w = col_w or [4.5*cm, 4*cm, 4.5*cm, 5*cm]
        t = Table(rows, colWidths=col_w)
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (0, -1), self.C['gray_lt']),
            ('BACKGROUND',    (2, 0), (2, -1), self.C['gray_lt']),
            ('FONTNAME',      (0, 0), (-1, -1), _ARABIC_FONT),
            ('FONTSIZE',      (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING',    (0, 0), (-1, -1), 7),
            ('LEFTPADDING',   (0, 0), (-1, -1), 8),
            ('GRID',          (0, 0), (-1, -1), 0.4, self.C['gray']),
        ]))
        return t

    # ── Sections ───────────────────────────────────────────────────────────
    def _header(self, patient: PatientInfo) -> List:
        out = []

        # Logo row
        logo_data = [[
            Paragraph('⚕', ParagraphStyle('logo', fontSize=32,
                       textColor=self.C['primary'], alignment=TA_CENTER)),
            Paragraph('<b>Kidney Disease Prediction Report</b><br/>'
                      f'<font size="10" color="gray">'
                      f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</font>',
                      self.styles['RTitle']),
            Paragraph(ar('تقرير التنبؤ بأمراض الكلى'),
                      ParagraphStyle('arTitle', fontName=_ARABIC_FONT,
                                     fontSize=14, alignment=TA_RIGHT,
                                     textColor=self.C['primary']))
        ]]
        lt = Table(logo_data, colWidths=[1.5*cm, 12*cm, 5*cm])
        lt.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ]))
        out.append(lt)
        out.append(HRFlowable(width='100%', thickness=2,
                               color=self.C['primary'], spaceAfter=12))

        # Patient info
        out += self._section_header('Patient Information', 'بيانات المريض')
        rows = [
            ['Patient Name', patient.name,
             ar('الاسم'), patient.name],
            ['Age / Sex', f"{patient.age} y  |  {patient.sex}",
             ar('العمر / الجنس'), f"{patient.age} {ar('سنة')}  |  {patient.sex}"],
            ['Date', patient.date,
             ar('التاريخ'), patient.date],
            ['Lab No.', patient.lab_no or '—',
             ar('رقم المعمل'), patient.lab_no or '—'],
        ]
        if patient.doctor_name:
            rows.append(['Doctor', patient.doctor_name,
                         ar('الطبيب'), patient.doctor_name])
        out.append(self._kv_table(rows))
        out.append(Spacer(1, 14))
        return out

    def _prediction_section(self, prediction: bool,
                             probability: float, risk_level: str) -> List:
        out = []
        out += self._section_header('Prediction Result', 'نتيجة التنبؤ')

        if prediction:
            icon, en_txt, ar_txt = '⚠', 'CKD DETECTED', ar('تم اكتشاف مرض الكلى المزمن')
            bg = self.C['danger_lt']
            border = self.C['danger']
            txt_c = self.C['danger']
        else:
            icon, en_txt, ar_txt = '✓', 'NO CKD DETECTED', ar('لا يوجد مرض الكلى المزمن')
            bg = self.C['success_lt']
            border = self.C['success']
            txt_c = self.C['success']

        result_data = [[
            Paragraph(icon,
                      ParagraphStyle('icon', fontSize=22, alignment=TA_CENTER,
                                     textColor=txt_c)),
            Paragraph(f'<b>{en_txt}</b>',
                      ParagraphStyle('res_en', fontSize=13, textColor=txt_c,
                                     fontName=_ARABIC_FONT)),
            Paragraph(f'<b>{ar_txt}</b>',
                      ParagraphStyle('res_ar', fontSize=13, textColor=txt_c,
                                     fontName=_ARABIC_FONT, alignment=TA_RIGHT)),
        ]]
        rt = Table(result_data, colWidths=[1.5*cm, 8*cm, 8.5*cm])
        rt.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), bg),
            ('BOX',           (0, 0), (-1, -1), 1.5, border),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING',    (0, 0), (-1, -1), 10),
            ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ]))
        out.append(rt)
        out.append(Spacer(1, 8))

        # Probability bar
        pct = int(probability * 100)
        bar_w = 16 * cm
        fill_w = bar_w * probability
        bar_data = [[
            Paragraph(f'Risk Probability: <b>{pct}%</b>',
                      self.styles['RBody']),
            Paragraph(f'<b>{risk_level}</b>',
                      ParagraphStyle('rl', fontSize=10, alignment=TA_RIGHT,
                                     textColor=txt_c, fontName=_ARABIC_FONT)),
        ]]
        bt = Table(bar_data, colWidths=[9*cm, 9*cm])
        out.append(bt)

        # Visual bar
        bar_cells = [['', '']]
        bar_t = Table(bar_cells, colWidths=[fill_w, bar_w - fill_w],
                      rowHeights=[0.5*cm])
        bar_t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), border),
            ('BACKGROUND', (1, 0), (1, 0), self.C['gray_lt']),
            ('BOX', (0, 0), (-1, -1), 0.5, self.C['gray']),
        ]))
        out.append(bar_t)
        out.append(Spacer(1, 14))
        return out

    def _staging_section(self, gfr_stage: str, egfr: float,
                          alb_category: str = None, acr: float = None) -> List:
        out = []
        out += self._section_header('CKD Staging (KDIGO)', 'مراحل الفشل الكلوي')

        stages = ['G1', 'G2', 'G3a', 'G3b', 'G4', 'G5']
        ranges = ['>90', '60-89', '45-59', '30-44', '15-29', '<15']
        stage_cells = [stages, ranges]
        col_w = [18*cm / 6] * 6
        st = Table(stage_cells, colWidths=col_w, rowHeights=[0.8*cm, 0.55*cm])

        style_cmds = [
            ('FONTNAME',      (0, 0), (-1, -1), _ARABIC_FONT),
            ('FONTSIZE',      (0, 0), (-1, 0), 11),
            ('FONTSIZE',      (0, 1), (-1, 1), 8),
            ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
            ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR',     (0, 1), (-1, 1), self.C['gray']),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ]
        for i, s in enumerate(stages):
            c = self.STAGE_COLORS.get(s, self.C['primary'])
            style_cmds.append(('BACKGROUND', (i, 0), (i, 0), c))
            if s == gfr_stage:
                style_cmds += [
                    ('BOX',      (i, 0), (i, 1), 3, colors.black),
                    ('FONTSIZE', (i, 0), (i, 0), 13),
                ]
        st.setStyle(TableStyle(style_cmds))
        out.append(st)
        out.append(Spacer(1, 8))

        info_rows = [
            ['eGFR Value', f'{egfr:.1f} mL/min/1.73m²',
             ar('معدل الترشيح'), f'{egfr:.1f}'],
            ['Current Stage', gfr_stage,
             ar('المرحلة الحالية'), gfr_stage],
        ]
        if alb_category and acr:
            info_rows.append(['ACR', f'{acr} mg/g',
                               ar('نسبة الألبومين'), alb_category])
        out.append(self._kv_table(info_rows))
        out.append(Spacer(1, 14))
        return out

    def _lab_section(self, results: List[TestResult]) -> List:
        out = []
        out += self._section_header('Lab Results', 'نتائج التحاليل')

        header = [
            Paragraph('<b>Test</b>', self.styles['RBody']),
            Paragraph('<b>Result</b>', self.styles['RCenter']),
            Paragraph('<b>Unit</b>', self.styles['RCenter']),
            Paragraph('<b>Ref. Range</b>', self.styles['RCenter']),
            Paragraph('<b>Status</b>', self.styles['RCenter']),
        ]
        rows = [header]
        style_cmds = [
            ('BACKGROUND',    (0, 0), (-1, 0), self.C['primary']),
            ('TEXTCOLOR',     (0, 0), (-1, 0), self.C['white']),
            ('FONTNAME',      (0, 0), (-1, -1), _ARABIC_FONT),
            ('FONTSIZE',      (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING',    (0, 0), (-1, -1), 7),
            ('GRID',          (0, 0), (-1, -1), 0.4, self.C['gray']),
            ('ALIGN',         (1, 1), (-1, -1), 'CENTER'),
        ]
        for i, r in enumerate(results, start=1):
            status_txt = '✗  Abnormal' if r.is_abnormal else '✓  Normal'
            status_c = self.C['danger'] if r.is_abnormal else self.C['success']
            rows.append([
                r.name,
                str(r.value),
                r.unit,
                r.reference_range,
                Paragraph(f'<font color="{status_c.hexval()}">'
                           f'<b>{status_txt}</b></font>',
                           self.styles['RCenter']),
            ])
            if r.is_abnormal:
                style_cmds.append(
                    ('BACKGROUND', (0, i), (-1, i), self.C['danger_lt']))

        t = Table(rows, colWidths=[5*cm, 2.5*cm, 2.5*cm, 3.5*cm, 4.5*cm])
        t.setStyle(TableStyle(style_cmds))
        out.append(t)
        out.append(Spacer(1, 14))
        return out

    def _recommendations_section(self, recommendations: List[str],
                                   alerts: List[str] = None) -> List:
        out = []

        if alerts:
            out += self._section_header('⚠  Clinical Alerts', 'تنبيهات طبية')
            for alert in alerts:
                a_data = [[
                    Paragraph('⚠', ParagraphStyle('aw', fontSize=14,
                               textColor=self.C['danger'], alignment=TA_CENTER)),
                    Paragraph(alert, ParagraphStyle('ab', fontSize=10,
                               textColor=self.C['danger'], fontName=_ARABIC_FONT,
                               leading=14)),
                ]]
                at = Table(a_data, colWidths=[1*cm, 17*cm])
                at.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), self.C['danger_lt']),
                    ('BOX',        (0, 0), (-1, -1), 1, self.C['danger']),
                    ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING',    (0, 0), (-1, -1), 8),
                ]))
                out.append(at)
                out.append(Spacer(1, 5))
            out.append(Spacer(1, 8))

        out += self._section_header('Recommendations', 'التوصيات الطبية')
        for i, rec in enumerate(recommendations, 1):
            rec_data = [[
                Paragraph(f'<b>{i}</b>',
                           ParagraphStyle('num', fontSize=11,
                                          textColor=self.C['primary'],
                                          alignment=TA_CENTER)),
                Paragraph(rec, ParagraphStyle('rec', fontSize=10,
                           fontName=_ARABIC_FONT, leading=15)),
            ]]
            rt = Table(rec_data, colWidths=[0.8*cm, 17.2*cm])
            rt.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), self.C['primary_lt']),
                ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING',    (0, 0), (-1, -1), 6),
                ('LINEBELOW', (0, 0), (-1, -1), 0.3, self.C['gray']),
            ]))
            out.append(rt)
        out.append(Spacer(1, 14))
        return out

    def _footer(self) -> List:
        out = [HRFlowable(width='100%', thickness=1,
                           color=self.C['gray'], spaceBefore=10)]
        ft = (
            '<para align="center">'
            '<font size="8" color="gray">'
            'AI-Powered Kidney Disease Prediction System &nbsp;|&nbsp; '
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br/>'
            'This report is NOT a substitute for professional medical advice.'
            '</font></para>'
        )
        out.append(Paragraph(ft, self.styles['Normal']))
        return out

    # ── Public API ─────────────────────────────────────────────────────────
    def generate_report(
        self,
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
        filename: str = None,
    ) -> str:
        if not filename:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe = patient.name.replace(' ', '_')[:20]
            filename = f"kidney_report_{safe}_{ts}.pdf"

        filepath = self.output_dir / filename
        doc = SimpleDocTemplate(
            str(filepath), pagesize=A4,
            rightMargin=1.5*cm, leftMargin=1.5*cm,
            topMargin=1.5*cm, bottomMargin=1.5*cm,
        )

        elems = []
        elems += self._header(patient)
        elems += self._prediction_section(prediction, probability, risk_level)
        elems += self._staging_section(gfr_stage, egfr, alb_category, acr)
        if lab_results:
            elems += self._lab_section(lab_results)
        if recommendations:
            elems += self._recommendations_section(recommendations, alerts)
        elems += self._footer()

        doc.build(elems)
        print(f"[OK] Report generated: {filepath}")
        return str(filepath)


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    g = PDFReportGenerator()
    p = PatientInfo(name="محمد أحمد", age=70, sex="Male",
                    date=datetime.now().strftime('%Y-%m-%d'),
                    lab_no="1806", doctor_name="Dr. Walaa")

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
    path = g.generate_report(
        patient=p, prediction=True, probability=0.85,
        risk_level="Very High Risk", gfr_stage="G4", egfr=28.5,
        alb_category="A2", acr=44.4, lab_results=labs,
        recommendations=recs, alerts=alrts,
    )
    print(f"Report saved -> {path}")
