# Backend Integration Documentation
# دليل ربط الـ Backend

##  Quick Start

### 1. Install Dependencies
```bash
cd kidney_disease_prediction
pip install -r requirements.txt
```

### 2. Run the API Server
```bash
python api.py
# or
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. API will be available at:
- **API Base URL**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

##  API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "model_loaded": true,
    "ocr_available": true
}
```

---

### Calculate eGFR
```http
POST /egfr?creatinine=2.3&age=70&is_female=false
```

**Response:**
```json
{
    "egfr": 28.5,
    "unit": "mL/min/1.73m²",
    "gfr_stage": "G4",
    "interpretation": "انخفاض شديد في وظائف الكلى"
}
```

---

### Calculate Stage
```http
POST /stage
Content-Type: application/json

{
    "creatinine": 2.3,
    "age": 70,
    "acr": 44.44,
    "is_female": false
}
```

**Response:**
```json
{
    "egfr": 28.5,
    "gfr_stage": "G4",
    "albuminuria_category": "A2",
    "risk_level": "Very High Risk",
    "description": "انخفاض شديد في وظائف الكلى",
    "recommendations": [...],
    "urgency_color": "#F44336"
}
```

---

### Predict CKD
```http
POST /predict
Content-Type: application/json

{
    "patient": {
        "name": "محمد أحمد",
        "age": 70,
        "sex": "male"
    },
    "lab_values": {
        "creatinine": 2.3,
        "acr": 44.44,
        "blood_urea": 61,
        "hemoglobin": 12.5
    }
}
```

**Response:**
```json
{
    "prediction": true,
    "probability": 0.85,
    "confidence": 0.70,
    "egfr": 28.5,
    "gfr_stage": "G4",
    "albuminuria_category": "A2",
    "risk_level": "Very High Risk",
    "progression_risk_percent": 55.0,
    "recommendations": [
        "المتابعة الدورية مع طبيب الكلى",
        "الحفاظ على ضغط الدم في المعدل الطبيعي",
        ...
    ],
    "alerts": [
        "⚠️ تحذير: وظائف الكلى منخفضة جداً"
    ]
}
```

---

### Predict from Image (OCR)
```http
POST /predict/image
Content-Type: multipart/form-data

image: [file]
patient_age: 70
patient_sex: male
```

**Response:**
```json
{
    "extracted_values": {
        "serum_creatinine": {"value": 2.3, "unit": "mg/dL"},
        "acr": {"value": 44.44, "unit": "mg/g"}
    },
    "patient_info": {"age": 70, "sex": "male"},
    "egfr": 28.5,
    "gfr_stage": "G4",
    "risk_level": "Very High Risk",
    "recommendations": [...]
}
```

---

### Predict from CT Scan (MobileNetV2)
```http
POST /predict/ct
Content-Type: multipart/form-data

file: [image.jpg]
```

**Response:**
```json
{
    "prediction": "Tumor",
    "confidence_percentage": 98.5,
    "clinical_note": "AI detected a tumor in the CT scan."
}
```

---

### Smart Diet Planner
```http
POST /diet/plan
Content-Type: application/json

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
                "meals": [{"type": "Breakfast", "food": "Oatmeal", "reason": "Low potassium"}]
            }
        ]
    }
}
```

---

### Generate PDF Report
```http
POST /report
Content-Type: application/json

{
    "patient": {
        "name": "محمد أحمد",
        "age": 70,
        "sex": "male"
    },
    "lab_values": {
        "creatinine": 2.3,
        "acr": 44.44
    }
}
```

**Response:**
```json
{
    "message": "Report generated successfully",
    "filename": "kidney_report_20250208_223000.pdf",
    "download_url": "/report/download/kidney_report_20250208_223000.pdf"
}
```

### Download Report
```http
GET /report/download/{filename}
```
Returns PDF file.

---

### Chat with AI (RAG)
```http
POST /chat
Content-Type: application/json

{
    "question": "What does eGFR 28 mean?",
    "patient_context": {
        "egfr": 28,
        "gfr_stage": "G4"
    }
}
```

**Response:**
```json
{
    "answer": "eGFR 28 indicates Stage 4 CKD (Severe)...",
    "sources": [
        {"source": "KDIGO Guidelines", "relevance": 0.92}
    ],
    "disclaimer": "..."
}
```

---

### Explain Results
```http
POST /explain
Content-Type: application/json

{
    "egfr": 28,
    "gfr_stage": "G4",
    "acr": 44.44
}
```

**Response:**
```json
{
    "explanation": "Based on your results...",
    "source": "AI-Generated with Medical Knowledge Base"
}
```

---

##  Configuration

### Environment Variables
You typically need to set the Gemini API Key for RAG features:
```bash
set GEMINI_API_KEY=your_api_key_here
```

### CORS
The API allows all origins by default. For production, update in `api.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    ...
)
```

### Port
Default port is 8000. To change:
```bash
uvicorn api:app --port 5000
```

---

##  Files for Backend Team

| File | Purpose |
|------|---------|
| `api.py` | **Main API file** - run this |
| `src/` | All model code |
| `requirements.txt` | Dependencies |
| `models/` | Trained models (after training) |

---

##  Testing with cURL

```bash
# Health check
curl http://localhost:8000/health

# Calculate eGFR
curl -X POST "http://localhost:8000/egfr?creatinine=2.3&age=70"

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"patient":{"name":"Test","age":70,"sex":"male"},"lab_values":{"creatinine":2.3}}'
```
