# API Contract — Kidney Disease Prediction System

> **Version:** 1.0.0 | **Base URL:** `http://<SERVER_IP>:8000` | **Docs:** `/docs`

> **IMPORTANT:** This document is the single source of truth for Frontend integration. All endpoints accept and return **JSON** (except `/predict/image` which uses `multipart/form-data` and `/report/download` which returns a PDF file).

---

## Quick Reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `POST` | `/egfr` | Calculate eGFR |
| `POST` | `/stage` | CKD Staging |
| `POST` | `/predict` | Full AI Prediction |
| `POST` | `/predict/image` | Predict from Lab Image (OCR) |
| `POST` | `/predict/ct` | Predict from CT Scan Image (MobileNetV2) |
| `POST` | `/report` | Generate PDF Report |
| `GET` | `/report/download/{filename}` | Download PDF |
| `POST` | `/chat` | Medical Q/A Chatbot |
| `POST` | `/diet/plan` | 7-Day Smart Diet Planner |
| `POST` | `/explain` | Explain Test Results |
| `GET` | `/chat/status` | RAG Service Status |
| `POST` | `/alerts/analyze` | Smart Alert Analysis |
| `POST` | `/alerts/symptoms` | Symptom NLP Analysis |
| `GET` | `/alerts/patient/{id}` | Patient Alert History |

---

## Error Codes

All endpoints may return these errors:

| Code | Meaning | Frontend Action |
|---|---|---|
| `200` | Success | Parse the JSON response |
| `400` | Bad Request (invalid input) | Show validation error to user |
| `404` | Not Found | Show "not found" message |
| `422` | Validation Error (wrong types) | Highlight the invalid fields |
| `500` | Server Error | Show "try again later" |
| `503` | Service Unavailable (OCR/RAG off) | Disable that feature in UI |

---

## 1. Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-03-18T01:00:00",
  "model_loaded": true,
  "ocr_available": true
}
```

**TIP:** Call this on app startup. If `model_loaded` is `false`, disable the Predict button. If `ocr_available` is `false`, hide the image upload feature.

---

## 2. Calculate eGFR

```
POST /egfr?creatinine=2.1&age=65&is_female=false
```

All parameters are **query parameters** (not JSON body).

| Param | Type | Required | Validation |
|---|---|---|---|
| `creatinine` | float | Yes | > 0 |
| `age` | int | Yes | > 0 |
| `is_female` | bool | No | Default: `false` |

**Response:**
```json
{
  "egfr": 32.5,
  "unit": "mL/min/1.73m2",
  "gfr_stage": "G3b",
  "interpretation": "Moderately to severely decreased"
}
```

---

## 3. CKD Staging

```
POST /stage
Content-Type: application/json
```

**Request Body:**
```json
{
  "creatinine": 2.3,
  "age": 70,
  "acr": 44.44,
  "is_female": false
}
```

| Field | Type | Required | Validation |
|---|---|---|---|
| `creatinine` | float | Yes | > 0 |
| `age` | int | Yes | 1 -- 120 |
| `acr` | float | No | >= 0 |
| `is_female` | bool | No | Default: `false` |

**Response:**
```json
{
  "egfr": 28.4,
  "gfr_stage": "G4",
  "albuminuria_category": "A2",
  "risk_level": "Very High Risk",
  "description": "Severely decreased kidney function",
  "recommendations": ["Follow up with nephrologist monthly"],
  "urgency_color": "red"
}
```

**TIP:** Use `urgency_color` directly for UI styling: `"green"`, `"yellow"`, `"orange"`, `"red"`.

---

## 4. Full AI Prediction (Main Endpoint)

```
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "patient": {
    "name": "Ahmed Mohamed",
    "age": 65,
    "sex": "male"
  },
  "lab_values": {
    "creatinine": 2.1,
    "acr": 44.44,
    "blood_urea": 61,
    "hemoglobin": 11.0,
    "blood_pressure": 150,
    "blood_glucose": 180,
    "sodium": 138,
    "potassium": 5.2,
    "hba1c": 8.5,
    "uacr": 120,
    "uric_acid": 7.5,
    "serum_albumin": 3.2,
    "bmi": 28.5,
    "blood_pressure_diastolic": 95,
    "diabetes_duration": 12,
    "smoking": 1,
    "dyslipidemia": 1,
    "diabetes_type": 2
  }
}
```

### Patient Fields

| Field | Type | Required | Validation | Default |
|---|---|---|---|---|
| `name` | string | No | -- | `"Patient"` |
| `age` | int | **Yes** | 1 -- 120 | -- |
| `sex` | string | No | `"male"` or `"female"` | `"male"` |

### Lab Values Fields

| Field | Type | Required | Unit | Default |
|---|---|---|---|---|
| `creatinine` | float | **Yes** | mg/dL | -- |
| `acr` | float | No | mg/g | `null` |
| `blood_urea` | float | No | mg/dL | `null` |
| `hemoglobin` | float | No | g/dL | `null` |
| `blood_pressure` | int | No | mmHg (systolic) | `null` |
| `blood_glucose` | float | No | mg/dL | `null` |
| `sodium` | float | No | mmol/L | `null` |
| `potassium` | float | No | mmol/L | `null` |
| `hba1c` | float | No | % | `null` |
| `uacr` | float | No | mg/g | `null` |
| `uric_acid` | float | No | mg/dL | `null` |
| `serum_albumin` | float | No | g/dL | `null` |
| `bmi` | float | No | kg/m2 | `null` |
| `blood_pressure_diastolic` | int | No | mmHg | `null` |
| `diabetes_duration` | int | No | years | `null` |
| `smoking` | int | No | 0 or 1 | `null` |
| `dyslipidemia` | int | No | 0 or 1 | `null` |
| `diabetes_type` | int | No | 0/1/2 | `null` |

**Response:**
```json
{
  "prediction": true,
  "probability": 0.8734,
  "confidence": 0.9201,
  "egfr": 28.4,
  "gfr_stage": "G4",
  "albuminuria_category": "A2",
  "risk_level": "Very High Risk",
  "progression_risk_percent": 55.0,
  "recommendations": [
    "Follow a healthy low-salt diet",
    "Monitor blood pressure daily",
    "Follow up with nephrologist monthly"
  ],
  "alerts": [
    "Warning: Very low kidney function",
    "High creatinine"
  ],
  "ai_staging": {
    "predicted_stage": 4,
    "stage_label": "Stage 4",
    "confidence": 0.82
  },
  "xai_explanation": {
    "top_risk_factors": [
      {"feature": "Serum Creatinine", "shap_value": 0.34, "feature_value": 2.1},
      {"feature": "eGFR", "shap_value": 0.28, "feature_value": 28.4}
    ],
    "top_protective_factors": [
      {"feature": "Sodium", "shap_value": -0.05, "feature_value": 138.0}
    ],
    "explanation_text": "The most significant risk factor is Serum Creatinine (value: 2.10)."
  }
}
```

### UI Mapping Guide

| Response Field | How to Display It |
|---|---|
| `prediction` | Badge: `true` = "CKD Detected" (red), `false` = "No CKD" (green) |
| `probability` | Progress bar (0% to 100%), color gradient green to red |
| `confidence` | Small text "Confidence: 92%" |
| `gfr_stage` | Badge: G1-G2 (green), G3a-G3b (orange), G4-G5 (red) |
| `risk_level` | Card header with matching color |
| `alerts` | Red notification box, each alert as a list item |
| `recommendations` | Checklist or accordion |
| `xai_explanation.top_risk_factors` | Horizontal bar chart sorted by `shap_value` |

---

## 5. Predict from Lab Image (OCR)

```
POST /predict/image
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | File | **Yes** | Lab report image (JPG/PNG) |
| `patient_age` | int | No | Default: `50` |
| `patient_sex` | string | No | Default: `"male"` |

**JavaScript Example:**
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('patient_age', 65);
formData.append('patient_sex', 'male');

const response = await fetch('http://SERVER:8000/predict/image', {
  method: 'POST',
  body: formData
});
const data = await response.json();
```

**Response:**
```json
{
  "extracted_data": {
    "lab_values": {"creatinine": 2.1, "hemoglobin": 11.0},
    "raw_text_extracted": "..."
  },
  "egfr": 28.4,
  "gfr_stage": "G4",
  "risk_level": "Very High Risk",
  "recommendations": ["..."],
  "warning": null
}
```

**WARNING:** If `warning` is not `null`, display it prominently. It means creatinine was not found in the image.

---

## 6. Predict from CT Scan (Deep Learning)

```
POST /predict/ct
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | **Yes** | CT scan image (JPG/PNG) |

**Response:**
```json
{
  "prediction": "Tumor",
  "confidence_percentage": 98.5,
  "clinical_note": "AI detected a tumor in the CT scan. Please refer to a specialist immediately."
}
```

---

## 7. Generate PDF Report

```
POST /report
Content-Type: application/json
```

**Request:** Same as `/predict` (same PredictionRequest schema).

**Response:**
```json
{
  "message": "Report generated successfully",
  "filename": "kidney_report_20260318_010000.pdf",
  "download_url": "/report/download/kidney_report_20260318_010000.pdf"
}
```

**Download the PDF:**
```
GET /report/download/kidney_report_20260318_010000.pdf
```
Returns: `application/pdf` file. Open in new tab or trigger browser download.

---

## 8. Smart Diet Planner (7-Day Plan)

```
POST /diet/plan
Content-Type: application/json
```

**Request:**
```json
{
  "age": 60,
  "weight_kg": 85,
  "ckd_stage": "G3b",
  "potassium_level": 5.2,
  "phosphorus_level": 4.5,
  "has_diabetes": true
}
```

**Response:**
```json
{
  "diet_plan": {
    "days": [
      {
        "day": 1,
        "meals": [
          {"type": "Breakfast", "food": "Oatmeal with apples", "reason": "Low potassium"}
        ]
      }
    ],
    "nutritional_targets": {
      "daily_calories": 2100,
      "protein_limit_grams": 55,
      "sodium_limit_mg": 2000
    }
  }
}
```

---

## 9. Medical Chatbot (RAG)

```
POST /chat
Content-Type: application/json
```

**Request:**
```json
{
  "question": "What does eGFR 28 mean?",
  "patient_context": {
    "egfr": 28,
    "gfr_stage": "G4"
  }
}
```

| Field | Type | Required |
|---|---|---|
| `question` | string | **Yes** (min 3 chars) |
| `patient_context` | object | No |

**Response:**
```json
{
  "answer": "eGFR 28 means your kidneys are working at about 28% of normal...",
  "sources": ["KDIGO Guidelines 2024"],
  "disclaimer": "This information is for educational purposes only."
}
```

---

## 8. Explain Test Results

```
POST /explain
Content-Type: application/json
```

**Request:**
```json
{
  "egfr": 28.4,
  "gfr_stage": "G4",
  "acr": 44.44,
  "risk_level": "Very High Risk"
}
```

**Response:**
```json
{
  "explanation": "Detailed AI-generated explanation of the results...",
  "egfr": 28.4,
  "stage": "G4",
  "source": "AI-Generated with Medical Knowledge Base"
}
```

---

## 9. Smart Alerts -- Analyze Patient

```
POST /alerts/analyze
Content-Type: application/json
```

**Request:**
```json
{
  "patient_id": "patient_001",
  "include_rule_based": true
}
```

**Response:**
```json
{
  "patient_id": "patient_001",
  "total_alerts": 3,
  "alerts": [
    {
      "type": "anomaly",
      "severity": "high",
      "message": "Abnormal creatinine trend detected",
      "timestamp": "2026-03-18T01:00:00"
    }
  ],
  "anomaly_detection": {
    "is_anomaly": true,
    "severity": "high",
    "anomaly_score": -0.82,
    "anomalous_features": ["creatinine", "egfr"]
  },
  "predictive_analytics": {
    "risk_score": 72.5,
    "risk_classification": "High Risk",
    "timeline": "3-6 months",
    "biomarker_trends": {}
  }
}
```

---

## 10. Smart Alerts -- Symptom Analysis

```
POST /alerts/symptoms
Content-Type: application/json
```

**Request:**
```json
{
  "text": "I feel swelling in my legs, dizziness, and reduced urination",
  "patient_id": "patient_001"
}
```

Supports both **Arabic** and **English** input.

**Response:**
```json
{
  "urgency": "high",
  "matched_conditions": ["Edema", "Oliguria"],
  "recommendations": ["Urgent nephrology consultation"],
  "correlation_with_labs": "...",
  "ai_analysis": "..."
}
```

---

## 11. Get Patient Alert History

```
GET /alerts/patient/patient_001
```

**Response:**
```json
{
  "patient_id": "patient_001",
  "total_measurements": 5,
  "latest_measurement": {"egfr": 32, "creatinine": 2.1},
  "total_alerts": 2,
  "alerts": [...]
}
```

---

## CORS Configuration

The API allows all origins by default. No CORS issues expected:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: *
Access-Control-Allow-Headers: *
```

---

## Running the Server

```bash
# Development
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Production
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

## Interactive Documentation

Once the server is running:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
