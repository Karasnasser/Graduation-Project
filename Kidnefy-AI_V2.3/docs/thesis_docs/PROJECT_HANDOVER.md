# Kidney Disease Prediction System - Project Handover

## Overview
AI-powered system for predicting Chronic Kidney Disease (CKD) and Diabetic Nephropathy:
1.  **ML Ensemble**: XGBoost, Random Forest, SVM (weighted voting)
2.  **Deep Learning**: Neural Network for complex pattern recognition
3.  **OCR Module**: Extract lab values from images (EasyOCR/Tesseract)
4.  **RAG System**: Gemini 1.5 Flash for medical Q&A
5.  **Staging & Risk**: eGFR (CKD-EPI 2021) + KDIGO staging
6.  **AI Staging (New)**: XGBoost Classifier for 6-class prediction (0-5) learned from 4000 patient records.
7.  **CT Vision Classifier (New)**: MobileNetV2 CNN for detecting 4 classes (Tumor, Cyst, Stone, Normal).
8.  **Smart Diet Planner (New)**: Gemini-powered 7-day personalized meal planner.

---

## Directory Structure
```
kidney_disease_prediction/
├── api.py              # FastAPI application
├── main.py             # CLI entry point (train/predict/stage)
├── train_diabetes.py   # Diabetes training pipeline
├── config.py           # Centralized settings (paths, defaults)
├── requirements.txt    # Pinned dependencies
├── .gitignore          # Files excluded from version control
│
├── src/
│   ├── models/         # ML + DL model implementations
│   ├── preprocessing/  # Data loading, merging, feature engineering
│   │   └── diabetes_preprocessing.py  # Diabetes-specific preprocessor
│   ├── ocr/            # Image processing + text extraction
│   ├── rag/            # Gemini RAG + ChromaDB vector DB
│   ├── reports/        # PDF generation logic
│   ├── imaging/        # CT image classification (MobileNetV2)
│   └── staging/        # eGFR calc + KDIGO risk assessment
│
├── models/             # Trained CKD models (.joblib, .keras)
│   └── diabetes/       # Trained diabetes models (.pkl)
│
├── data/raw/           # Datasets
│   ├── kidney_disease.csv
│   ├── Diabetic_Nephropathy_v1.xlsx
│   └── diabetes_prediction_dataset.csv
│
├── test_simple.py      # Staging logic verification
└── test_client.py      # API endpoint test client
```

---

## Setup & Installation

1.  **Python**: 3.10+ recommended
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables**:
    ```bash
    set GEMINI_API_KEY=your_key    # Required for RAG/Chat features
    ```
4.  **System Dependencies (for OCR)**:
    - **Tesseract**: Install `tesseract-ocr` binary at OS level
    - **EasyOCR**: Requires PyTorch (auto-installed, CPU or CUDA)

---

## Two Training Pipelines

### 1. CKD Models (main pipeline)
```bash
python main.py train --epochs 50
```
- Loads `kidney_disease.csv` + `Diabetic_Nephropathy_v1.xlsx`
- Merges, preprocesses, and trains Ensemble (ML + DL)
- Saves to `models/` directory

### 2. Diabetes Models
```bash
python train_diabetes.py
```
- Loads `diabetes_prediction_dataset.csv`
- Uses `DiabetesPreprocessor` (in `src/preprocessing/diabetes_preprocessing.py`)
- Saves to `models/diabetes/` directory

### 3. AI Staging Model (New)
```bash
python train_staging.py
```
- Loads `updated_ckd_dataset_with_stages.csv` (4000 rows)
- Trains XGBoost Multi-class Classifier (Stages 0-5)
- Saves to `models/staging/xgb_staging.json`

---

## Key Files for Backend Integration

| File | Purpose |
|------|---------|
| `api.py` | **Main API — run this** |
| `config.py` | All configurable paths, ports, and defaults |
| `src/` | Source code (all modules) |
| `models/` | Trained model files (must exist before prediction) |
| `requirements.txt` | Dependencies |
| `API_DOCUMENTATION.md` | Full endpoint reference with examples |

---

## Recent Fixes (Verified)
1.  **`rag_engine` Bug**: Fixed missing `global` keyword — RAG/Chat endpoints now work properly
2.  **Data Leakage**: Removed `id` column from training
3.  **XLSX Support**: Diabetic Nephropathy dataset loads from `.xlsx`
4.  **Crash Fixes**: Fixed `AttributeError` in staging report generation
5.  **XGBoost Compatibility**: Removed deprecated parameters
6.  **RAG Updates**: Updated to modern `chromadb` API and `gemini-1.5-flash`
7.  **AI Staging**: Added support for 5-stage prediction using new dataset (4000 records)
8.  **CT Scan**: Added `MobileNetV2` integration for `.keras` inference.
9.  **Diet Planner**: Added strict medical prompt injection for KDIGO guidelines.
10. **Cleanup**: Moved old training scripts to `scripts/`, archived old docs to `docs/thesis_docs/`.

---

## Notes for Backend Team
-   **Config**: All model paths and settings are centralized in `config.py`
-   **RAG Persistence**: Vector DB saved in `knowledge_base/` — ensure write permissions
-   **OCR**: EasyOCR needs PyTorch. Tesseract needs OS-level binary
-   **Model Loading**: Verify `models/` has trained files before running inference
-   **CORS**: Default allows all origins — restrict in production via `api.py`

---

**Status**: ✅ Ready for Integration
