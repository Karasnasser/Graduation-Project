"""
Kidney Disease Prediction API
FastAPI web service for kidney disease prediction and staging.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import pandas as pd
import numpy as np

# Import our modules (all wrapped to prevent server crash if a dependency is missing)
from src.preprocessing import DataLoader, calculate_egfr, FeatureEngineer
from src.staging import GFRCalculator, RiskAssessor, GFRStage, RiskLevel
from config import CORS_ORIGINS
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
try:
    from src.models import EnsembleModel
    ENSEMBLE_AVAILABLE = True
except Exception as e:
    ENSEMBLE_AVAILABLE = False
    EnsembleModel = None
    print(f"Warning: ML/DL models not available ({e}). Prediction will use staging fallback.")

try:
    from src.models.staging_model import StagingModel
    STAGING_MODEL_AVAILABLE = True
except Exception as e:
    STAGING_MODEL_AVAILABLE = False
    StagingModel = None
    print(f"Warning: Staging model not available ({e}).")

try:
    from src.reports import PDFReportGenerator, PatientInfo, TestResult
    REPORTS_AVAILABLE = True
except Exception as e:
    REPORTS_AVAILABLE = False
    print(f"Warning: PDF report generation not available ({e}).")

try:
    from src.monitoring import LongitudinalMonitor, SmartAlertEngine
    MONITORING_AVAILABLE = True
except Exception as e:
    MONITORING_AVAILABLE = False
    LongitudinalMonitor = None
    SmartAlertEngine = None
    print(f"Warning: Monitoring not available ({e}).")

try:
    from src.explainability import SHAPExplainer
    SHAP_AVAILABLE = True
except Exception as e:
    SHAP_AVAILABLE = False
    SHAPExplainer = None
    print(f"Warning: SHAP explainability not available ({e}).")



# Try to import RAG
try:
    from src.rag import GeminiRAG
    RAG_AVAILABLE = True
except Exception as e:
    RAG_AVAILABLE = False
    print(f"Warning: RAG not available ({e}). Chat feature will be disabled.")

# Try to import Smart Diet Planner
try:
    from src.rag.diet_planner import SmartDietPlanner
    DIET_PLANNER_AVAILABLE = True
except Exception as e:
    DIET_PLANNER_AVAILABLE = False
    print(f"Warning: Diet Planner not available ({e}).")

# Try to import CT Kidney Image Classifier
try:
    from src.imaging.kidney_image_classifier import KidneyImageClassifier
    CT_CLASSIFIER_AVAILABLE = True
except Exception as e:
    CT_CLASSIFIER_AVAILABLE = False
    print(f"Warning: CT Classifier not available ({e}).")


# =============================================================================
# Pydantic Models (Request/Response Schemas)
# =============================================================================

class PatientData(BaseModel):
    """Patient information for predictions."""
    name: str = Field(default="Patient", description="Patient name")
    age: int = Field(..., ge=1, le=120, description="Patient age in years")
    sex: str = Field(default="male", description="Patient sex (male/female)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "محمد أحمد",
                "age": 70,
                "sex": "male"
            }
        }


class LabValues(BaseModel):
    """Laboratory test values for prediction."""
    creatinine: float = Field(..., gt=0, description="Serum creatinine in mg/dL")
    acr: Optional[float] = Field(None, ge=0, description="Albumin/Creatinine Ratio in mg/g")
    blood_urea: Optional[float] = Field(None, ge=0, description="Blood urea in mg/dL")
    blood_urea_nitrogen: Optional[float] = Field(None, ge=0, description="Blood urea nitrogen in mg/dL")
    sodium: Optional[float] = Field(None, description="Sodium in mmol/L")
    potassium: Optional[float] = Field(None, description="Potassium in mmol/L")
    calcium: Optional[float] = Field(None, description="Calcium in mg/dL")
    magnesium: Optional[float] = Field(None, description="Magnesium in mg/dL")
    hba1c: Optional[float] = Field(None, ge=0, description="HbA1c percentage")
    uacr: Optional[float] = Field(None, ge=0, description="Urine Albumin-Creatinine Ratio")
    
    class Config:
        json_schema_extra = {
            "example": {
                "creatinine": 2.3,
                "acr": 44.44,
                "blood_urea": 61,
                "blood_urea_nitrogen": 28,
                "sodium": 140,
                "potassium": 4.5,
                "calcium": 9.0,
                "magnesium": 2.0,
                "hba1c": 5.5,
                "uacr": 40.0
            }
        }


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""
    patient: PatientData
    lab_values: LabValues


class WhatIfPatientData(BaseModel):
    """Single scenario for the What-If Simulator.
    
    Mirrors the lightweight payload the dashboard sends:
    {age, sex, sc (creatinine), bp, al (albumin level 0-4), dm ("yes"/"no")}
    """
    age: int = Field(default=60, ge=1, le=120, description="Patient age")
    sex: str = Field(default="male", description="Patient sex (male/female)")
    sc: float = Field(..., gt=0, description="Serum creatinine in mg/dL")
    bp: int = Field(default=120, ge=0, description="Systolic blood pressure")
    al: int = Field(default=0, ge=0, le=5, description="Albumin level (0-5 scale)")
    dm: str = Field(default="no", description="Diabetes mellitus (yes/no)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 60, "sex": "male",
                "sc": 2.5, "bp": 160, "al": 2, "dm": "no"
            }
        }


class WhatIfRequest(BaseModel):
    """Request body for the What-If Treatment Simulator.
    
    Compares a *baseline* clinical state against a hypothetical
    *modified* state to quantify the impact of treatment changes.
    """
    baseline: WhatIfPatientData
    modified: WhatIfPatientData

    class Config:
        json_schema_extra = {
            "example": {
                "baseline": {"age": 60, "sex": "male", "sc": 2.5, "bp": 160, "al": 2, "dm": "no"},
                "modified": {"age": 60, "sex": "male", "sc": 1.8, "bp": 125, "al": 0, "dm": "no"}
            }
        }


class ChatRequest(BaseModel):
    """Request body for the medical chatbot endpoint."""
    question: str = Field(..., description="The user's question")
    patient_id: Optional[str] = Field(None, description="Optional patient ID to include clinical context")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "ما هي الأطعمة الممنوعة لمريض الكلى في المرحلة الثالثة؟"
            }
        }


class DietRequest(BaseModel):
    """Request body for the Smart Diet Planner endpoint."""
    age: int = Field(..., ge=1, le=120, description="Patient age in years")
    weight: float = Field(..., gt=0, description="Patient weight in kg")
    egfr: Optional[float] = Field(None, ge=0, description="eGFR value (will auto-calculate stage)")
    stage: Optional[str] = Field(None, description="CKD stage (G1–G5) if eGFR not available")
    potassium: Optional[float] = Field(4.5, description="Serum potassium in mEq/L (normal: 3.5-5.0)")
    sodium: Optional[float] = Field(140.0, description="Serum sodium in mEq/L (normal: 135-145)")
    diabetes: Optional[str] = Field("no", description="Diabetes status (yes/no)")
    hypertension: Optional[str] = Field("no", description="Hypertension status (yes/no)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 58,
                "weight": 80,
                "egfr": 28,
                "potassium": 5.8,
                "sodium": 145,
                "diabetes": "yes",
                "hypertension": "yes"
            }
        }


class StageRequest(BaseModel):
    """Request body for staging endpoint."""
    creatinine: float = Field(..., gt=0)
    age: int = Field(..., ge=1, le=120)
    acr: Optional[float] = Field(None)
    is_female: bool = Field(default=False)


class GFRStageEnum(str, Enum):
    G1 = "G1"
    G2 = "G2"
    G3a = "G3a"
    G3b = "G3b"
    G4 = "G4"
    G5 = "G5"


class RiskLevelEnum(str, Enum):
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"
    VERY_HIGH = "Very High Risk"
    CRITICAL = "Critical - Kidney Failure"


class StagingResult(BaseModel):
    """Response for staging endpoint."""
    egfr: float
    gfr_stage: str
    albuminuria_category: Optional[str]
    risk_level: str
    description: str
    recommendations: List[str]
    urgency_color: str


class PredictionResult(BaseModel):
    """Response for prediction endpoint."""
    prediction: bool
    probability: float
    confidence: float
    egfr: float
    gfr_stage: str
    albuminuria_category: Optional[str]
    risk_level: str
    progression_risk_percent: float
    recommendations: List[str]
    alerts: List[str]
    ai_staging: Optional[Dict[str, Any]] = None
    xai_explanation: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    model_loaded: bool


# =============================================================================
# FastAPI Application
# =============================================================================

# --- Global instances (module-level, available before lifespan runs) ---
gfr_calculator = GFRCalculator()
risk_assessor = RiskAssessor()
report_generator = PDFReportGenerator(output_dir="generated_reports") if REPORTS_AVAILABLE else None
ensemble_model = None  # Will be loaded on startup
staging_model = None   # AI Staging Model
rag_engine = None  # RAG for medical Q&A
diet_planner = None  # Smart Diet Planner
longitudinal_monitor = LongitudinalMonitor() if MONITORING_AVAILABLE and LongitudinalMonitor else None
smart_alert_engine = None  # Will be initialized on startup
feature_engineer = FeatureEngineer()
shap_explainer = SHAPExplainer() if SHAP_AVAILABLE and SHAPExplainer else None
xai_ready = False
ct_classifier = None  # CT Kidney Image Classifier


# =============================================================================
# Lifespan (replaces deprecated @app.on_event — FastAPI >= 0.93)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup, clean up on shutdown."""
    global ensemble_model, rag_engine, staging_model, xai_ready, smart_alert_engine, diet_planner, ct_classifier

    print(" Starting Kidney Disease Prediction API...")

    # Create directories
    Path("generated_reports").mkdir(exist_ok=True)
    Path("uploads").mkdir(exist_ok=True)

    # Load ML ensemble (metadata + joblib checkpoints + optional DL)
    model_path = Path("models")
    if ENSEMBLE_AVAILABLE and model_path.exists():
        try:
            ensemble_model = EnsembleModel(str(model_path))
            ensemble_model.load()
            if ensemble_model.is_trained:
                print("ML models loaded successfully")
            else:
                print("No trained ML checkpoints found under models/ (train or add .joblib files)")
        except Exception as e:
            print(f"Could not load ML models: {e}")

    xai_ready = False
    if shap_explainer and ensemble_model and ensemble_model.is_trained:
        xgb_m = ensemble_model.ml_models.models.get("XGBoost")
        if xgb_m is not None:
            try:
                n_feat = getattr(
                    xgb_m, "n_features_in_", None
                ) or (
                    len(ensemble_model.feature_names) if ensemble_model.feature_names else 1
                )
                n_feat = max(1, int(n_feat))
                rng = np.random.default_rng(42)
                bg = rng.normal(0.0, 0.01, size=(min(100, max(10, n_feat)), n_feat)).astype(
                    np.float64
                )
                shap_explainer.fit(xgb_m, bg, model_type="tree")
                xai_ready = shap_explainer.explainer is not None
                if xai_ready:
                    print("SHAP explainer initialized")
            except Exception as e:
                print(f"SHAP initialization failed: {e}")

    # Load Staging Model
    if STAGING_MODEL_AVAILABLE:
        try:
            staging_model = StagingModel()
        except Exception as e:
            print(f"Could not load Staging model: {e}")



    # Initialize RAG
    if RAG_AVAILABLE:
        print("RAG module is available. It will be initialized on first use to prevent ChromaDB startup crashes.")

    # Initialize Smart Alerts
    if MONITORING_AVAILABLE and SmartAlertEngine:
        try:
            smart_alert_engine = SmartAlertEngine(
                monitor=longitudinal_monitor,
                gemini_rag=rag_engine
            )
            print("Smart Alert Engine initialized")
        except Exception as e:
            print(f"Smart Alert Engine failed: {e}")

    # Initialize Smart Diet Planner
    if DIET_PLANNER_AVAILABLE:
        try:
            diet_planner = SmartDietPlanner()
            if diet_planner.is_active:
                print("[OK] Smart Diet Planner initialized")
            else:
                print("[WARN] Smart Diet Planner: GEMINI_API_KEY missing, feature disabled.")
        except Exception as e:
            print(f"[FAIL] Diet Planner initialization failed: {e}")

    # Initialize CT Kidney Image Classifier
    if CT_CLASSIFIER_AVAILABLE:
        try:
            ct_classifier = KidneyImageClassifier()
            if ct_classifier.is_ready:
                print("[OK] CT Kidney Classifier loaded successfully")
            else:
                print("[WARN] CT Classifier: No trained model found. Run train_ultrasound.py first.")
        except Exception as e:
            print(f"[FAIL] CT Classifier initialization failed: {e}")

    print("[OK] API ready!")
    yield
    # Shutdown cleanup (if any) goes here


app = FastAPI(
    title="Kidney Disease Prediction API",
    description="""
    ## نظام التنبؤ بأمراض الكلى
    
    API for predicting chronic kidney disease (CKD) and diabetic nephropathy.
    
    ### Features:
    - **Prediction**: ML/DL-based CKD prediction
    - **Staging**: GFR-based kidney disease staging (G1-G5)
    - **Reports**: Generate PDF reports
    
    ### Endpoints:
    - `POST /predict` - Predict CKD from lab values
    - `POST /stage` - Calculate kidney disease stage
    - `POST /egfr` - Calculate eGFR value
    - `POST /report` - Generate PDF report
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS: wildcard origin cannot use credentials (browser + spec); use CORS_ORIGINS for production
_cors_raw = (CORS_ORIGINS or "*").strip()
if _cors_raw == "*":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    _origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
    if not _origins:
        _origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Kidney Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        model_loaded=ensemble_model is not None
    )


@app.post("/egfr", tags=["Calculations"])
async def calculate_egfr_endpoint(
    creatinine: float,
    age: int,
    is_female: bool = False
):
    """
    Calculate eGFR (estimated Glomerular Filtration Rate).
    
    Uses CKD-EPI 2021 equation.
    """
    if creatinine <= 0 or age <= 0:
        raise HTTPException(status_code=400, detail="Invalid input values")
    
    egfr = gfr_calculator.calculate_egfr_ckdepi(creatinine, age, is_female)
    stage = gfr_calculator.get_gfr_stage(egfr)
    
    return {
        "egfr": egfr,
        "unit": "mL/min/1.73m²",
        "gfr_stage": stage.value,
        "interpretation": gfr_calculator.STAGE_DESCRIPTIONS.get(stage.value, "Unknown")
    }


@app.post("/stage", response_model=StagingResult, tags=["Staging"])
async def calculate_stage(request: StageRequest):
    """
    Calculate complete kidney disease staging.
    
    Based on KDIGO guidelines using eGFR and ACR.
    """
    try:
        result = gfr_calculator.calculate_stage(
            creatinine=request.creatinine,
            acr=request.acr,
            age=request.age,
            is_female=request.is_female
        )
        
        return StagingResult(
            egfr=result.egfr_value,
            gfr_stage=result.gfr_stage.value,
            albuminuria_category=result.albuminuria_category.value if result.albuminuria_category else None,
            risk_level=result.risk_level.value,
            description=result.description,
            recommendations=result.recommendations,
            urgency_color=result.urgency_color
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict CKD from patient data and lab values.
    
    Combines ML/DL predictions with clinical staging.
    """
    try:
        patient = request.patient
        labs = request.lab_values
        
        # Calculate eGFR
        egfr = calculate_egfr(
            labs.creatinine,
            patient.age,
            patient.sex.lower() == "female"
        )
        
        # Get staging
        staging = gfr_calculator.calculate_stage(
            creatinine=labs.creatinine,
            acr=labs.acr,
            age=patient.age,
            is_female=patient.sex.lower() == "female"
        )
        
        # Get complete assessment
        # Use ML ensemble model for probability calculation if trained, else fallback to staging estimates
        probability = 0.5
        if ensemble_model and ensemble_model.is_trained:
            from config import CKD_FEATURE_ORDER, CKD_FEATURE_DEFAULTS
            
            # Prepare feature dict
            feature_dict = {}
            for f in CKD_FEATURE_ORDER:
                val = None

                if f == "sc":
                    val = labs.creatinine
                elif f == "age":
                    val = patient.age
                elif f == "gender":
                    val = 1 if patient.sex.lower() == "male" else 0
                elif f == "bu":
                    val = labs.blood_urea
                elif f == "bun":
                    val = labs.blood_urea_nitrogen
                elif f == "sod":
                    val = labs.sodium
                elif f == "pot":
                    val = labs.potassium
                elif f == "cal":
                    val = labs.calcium
                elif f == "mag":
                    val = labs.magnesium
                elif f == "hba1c":
                    val = labs.hba1c
                elif f == "uacr":
                    val = labs.uacr if labs.uacr is not None else labs.acr

                if val is None:
                    val = CKD_FEATURE_DEFAULTS.get(f, 0)

                feature_dict[f] = [val]
            
            # Transform and align features
            df_features = pd.DataFrame(feature_dict)
            df_features = feature_engineer.create_categorical_bins(df_features)
            
            trained_features = None
            if ensemble_model.feature_names:
                trained_features = ensemble_model.feature_names
            elif (
                getattr(ensemble_model.ml_models, "feature_names", None)
            ):
                trained_features = ensemble_model.ml_models.feature_names
            if trained_features:
                for col in trained_features:
                    if col not in df_features.columns:
                        df_features[col] = 0
                df_features = df_features[trained_features]
            
            # Get probability
            df_features = df_features.astype(float)  # Prevent XGBoost object dtype crash
            feature_vector = df_features.values
            _, _, details = ensemble_model.predict_with_confidence(feature_vector)
            probability = float(details['ensemble_proba'][0])
            
        else:
            # Fallback staging-based probability
            if staging.gfr_stage in [GFRStage.G4, GFRStage.G5]:
                probability = 0.95
            elif staging.gfr_stage == GFRStage.G3b:
                probability = 0.80
            elif staging.gfr_stage == GFRStage.G3a:
                probability = 0.60
            elif staging.gfr_stage == GFRStage.G2:
                probability = 0.30
            else:
                probability = 0.10
            
            # Adjust for ACR
            if labs.acr and labs.acr >= 300:
                probability = min(probability + 0.2, 0.99)
            elif labs.acr and labs.acr >= 30:
                probability = min(probability + 0.1, 0.99)
        
        
        # Get risk assessment
        assessment = risk_assessor.complete_assessment(
            ckd_probability=probability,
            creatinine=labs.creatinine,
            egfr=egfr,
            acr=labs.acr,
            age=patient.age,
            is_female=patient.sex.lower() == "female"
        )
        
        # AI Staging Prediction
        ai_stage_result = None
        if staging_model:
            # Map input to features expected by model (creatinine, age, bp, etc)
            # We pass all available labs + patient info
            stage_input = {
                "age": patient.age,
                "blood_pressure": labs.blood_pressure if labs.blood_pressure else 120, # Default BP
                "serum_creatinine": labs.creatinine,
                "gfr": egfr,
                "bun": labs.blood_urea if labs.blood_urea else 40, # Default BUN
                "serum_calcium": 9.0, # Default Ca (not in input)
                "hemoglobin": labs.hemoglobin if labs.hemoglobin else 14.0,
            }
            ai_stage_result = staging_model.predict_stage(stage_input)

        # Improve confidence using model agreement when ensemble is trained
        if ensemble_model and ensemble_model.is_trained:
            # Safely extract confidence and agreement (might be scalar or array)
            conf_val = details.get('confidence', abs(probability - 0.5) * 2)
            confidence_score = float(conf_val[0] if isinstance(conf_val, (list, np.ndarray)) and len(conf_val) > 0 else conf_val)
            
            agr_val = details.get('model_agreement', 1.0)
            agreement = float(agr_val[0] if isinstance(agr_val, (list, np.ndarray)) and len(agr_val) > 0 else agr_val)
            if np.isnan(agreement):
                agreement = 0.5
            # Blend distance-from-threshold with model agreement
            confidence_val = round((confidence_score * 0.6) + (agreement * 0.4), 4)
        else:
            confidence_val = round(abs(probability - 0.5) * 2, 4)

        # SHAP Explanation (if available)
        xai_result = None
        if xai_ready and ensemble_model and ensemble_model.is_trained:
            try:
                explanation = shap_explainer.explain_prediction(
                    feature_vector, ensemble_model.feature_names, top_k=5
                )
                if 'error' not in explanation:
                    xai_result = {
                        'top_risk_factors': explanation.get('top_risk_factors', [])[:5],
                        'top_protective_factors': explanation.get('top_protective_factors', [])[:5],
                        'explanation_text': explanation.get('explanation_text', '')
                    }
            except Exception:
                xai_result = None

        return PredictionResult(
            prediction=assessment.ckd_prediction,
            probability=round(probability, 4),
            confidence=confidence_val,
            egfr=egfr,
            gfr_stage=assessment.gfr_stage.value,
            albuminuria_category=assessment.albuminuria_category.value if assessment.albuminuria_category else None,
            risk_level=assessment.risk_level.value,
            progression_risk_percent=assessment.progression_risk.risk_percentage,
            recommendations=assessment.recommendations,
            alerts=assessment.alerts,
            ai_staging=ai_stage_result,
            xai_explanation=xai_result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# What-If Treatment Simulator
# ─────────────────────────────────────────────────────────────────────────────

def _whatif_evaluate_scenario(scenario: WhatIfPatientData) -> Dict[str, Any]:
    """Evaluate a single What-If scenario.
    
    Returns eGFR, GFR stage, CKD probability, risk level,
    and progression_risk_percent for one set of clinical values.
    """
    is_female = scenario.sex.lower() == "female"

    # 1. eGFR via CKD-EPI
    egfr = gfr_calculator.calculate_egfr_ckdepi(
        scenario.sc, scenario.age, is_female
    )

    # 2. Map albumin level (0-5) → approximate ACR for staging
    al_to_acr = {0: 10, 1: 20, 2: 80, 3: 200, 4: 500, 5: 1000}
    acr_estimate = al_to_acr.get(scenario.al, 10)

    # 3. KDIGO staging
    staging = gfr_calculator.calculate_stage(
        creatinine=scenario.sc,
        acr=acr_estimate,
        age=scenario.age,
        is_female=is_female
    )

    # 4. ML probability (if ensemble loaded) or staging-based fallback
    probability = 0.5
    if ensemble_model and ensemble_model.is_trained:
        from config import CKD_FEATURE_ORDER, CKD_FEATURE_DEFAULTS
        feature_dict = {}
        for f in CKD_FEATURE_ORDER:
            if f == "age":
                val = scenario.age
            elif f == "gender":
                val = 1 if scenario.sex.lower() == "male" else 0
            elif f == "sc":
                val = scenario.sc
            elif f == "uacr":
                val = acr_estimate
            else:
                val = CKD_FEATURE_DEFAULTS.get(f, 0)
            feature_dict[f] = [val]

        df_features = pd.DataFrame(feature_dict)
        df_features = feature_engineer.create_categorical_bins(df_features)

        trained_features = None
        if ensemble_model.feature_names:
            trained_features = ensemble_model.feature_names
        elif getattr(ensemble_model.ml_models, "feature_names", None):
            trained_features = ensemble_model.ml_models.feature_names
        if trained_features:
            for col in trained_features:
                if col not in df_features.columns:
                    df_features[col] = 0
            df_features = df_features[trained_features]

        feature_vector = df_features.values
        _, _, details = ensemble_model.predict_with_confidence(feature_vector)
        probability = float(details['ensemble_proba'][0])
    else:
        # Staging-based fallback (same logic as /predict)
        stage_prob = {
            GFRStage.G1: 0.10, GFRStage.G2: 0.30,
            GFRStage.G3a: 0.60, GFRStage.G3b: 0.80,
            GFRStage.G4: 0.95, GFRStage.G5: 0.95,
        }
        probability = stage_prob.get(staging.gfr_stage, 0.5)
        # Adjust for ACR
        if acr_estimate >= 300:
            probability = min(probability + 0.2, 0.99)
        elif acr_estimate >= 30:
            probability = min(probability + 0.1, 0.99)

    # 5. Full risk assessment
    assessment = risk_assessor.complete_assessment(
        ckd_probability=probability,
        creatinine=scenario.sc,
        egfr=egfr,
        acr=acr_estimate,
        age=scenario.age,
        is_female=is_female
    )

    return {
        "egfr": round(egfr, 2),
        "gfr_stage": assessment.gfr_stage.value,
        "probability": round(probability, 4),
        "risk_level": assessment.risk_level.value,
        "progression_risk_percent": assessment.progression_risk.risk_percentage,
        "recommendations": assessment.recommendations,
        "alerts": assessment.alerts,
    }


@app.post("/predict/whatif", tags=["Prediction"])
async def predict_whatif(request: WhatIfRequest):
    """
    What-If Treatment Simulator.

    Compare two clinical scenarios (baseline vs. modified) to quantify
    the impact of treatment changes on kidney disease progression risk.

    The response includes full results for both states plus computed deltas
    so the frontend can visualise improvements (or deterioration) instantly.
    """
    try:
        baseline = _whatif_evaluate_scenario(request.baseline)
        modified = _whatif_evaluate_scenario(request.modified)

        # Compute deltas
        delta_prob  = round(modified["probability"] - baseline["probability"], 4)
        delta_egfr  = round(modified["egfr"] - baseline["egfr"], 2)
        delta_risk  = round(
            modified["progression_risk_percent"] - baseline["progression_risk_percent"], 2
        )

        # Stage comparison (e.g. "G4 → G3a")
        stage_change = (
            f"{baseline['gfr_stage']} → {modified['gfr_stage']}"
            if baseline["gfr_stage"] != modified["gfr_stage"]
            else "No change"
        )

        return {
            "baseline": baseline,
            "modified": modified,
            "deltas": {
                "probability": delta_prob,
                "egfr": delta_egfr,
                "progression_risk": delta_risk,
                "stage_change": stage_change,
                "risk_improved": delta_prob < 0,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/report", tags=["Reports"])
async def generate_report(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate PDF report for patient.
    
    Returns download link for the generated report.
    """
    if not REPORTS_AVAILABLE or report_generator is None:
        raise HTTPException(status_code=503, detail="PDF report generation is not available. Ensure reportlab is installed.")
    try:
        patient = request.patient
        labs = request.lab_values
        
        # First get prediction
        egfr = calculate_egfr(
            labs.creatinine,
            patient.age,
            patient.sex.lower() == "female"
        )
        
        staging = gfr_calculator.calculate_stage(
            creatinine=labs.creatinine,
            acr=labs.acr,
            age=patient.age,
            is_female=patient.sex.lower() == "female"
        )
        
        # Create PatientInfo
        patient_info = PatientInfo(
            name=patient.name,
            age=patient.age,
            sex=patient.sex,
            date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Create lab results
        lab_results = [
            TestResult(
                name="Serum Creatinine",
                value=labs.creatinine,
                unit="mg/dL",
                reference_range="0.5 - 1.5",
                is_abnormal=labs.creatinine > 1.5
            ),
            TestResult(
                name="eGFR",
                value=egfr,
                unit="mL/min/1.73m²",
                reference_range="> 90",
                is_abnormal=egfr < 90
            )
        ]
        
        if labs.acr:
            lab_results.append(TestResult(
                name="ACR",
                value=labs.acr,
                unit="mg/g",
                reference_range="< 30",
                is_abnormal=labs.acr >= 30
            ))
        
        if labs.blood_urea:
            lab_results.append(TestResult(
                name="Blood Urea",
                value=labs.blood_urea,
                unit="mg/dL",
                reference_range="10 - 50",
                is_abnormal=labs.blood_urea > 50
            ))
        
        # Calculate probability based on staging
        probability = 0.5
        if staging.gfr_stage in [GFRStage.G4, GFRStage.G5]:
            probability = 0.9
        elif staging.gfr_stage == GFRStage.G3b:
            probability = 0.75
        elif staging.gfr_stage == GFRStage.G3a:
            probability = 0.5
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"kidney_report_{timestamp}.pdf"
        
        # Generate report
        filepath = report_generator.generate_report(
            patient=patient_info,
            prediction=staging.gfr_stage not in [GFRStage.G1, GFRStage.G2],
            probability=probability,
            risk_level=staging.risk_level.value,
            gfr_stage=staging.gfr_stage.value,
            egfr=egfr,
            alb_category=staging.albuminuria_category.value if staging.albuminuria_category else None,
            acr=labs.acr,
            lab_results=lab_results,
            recommendations=staging.recommendations,
            alerts=[],
            filename=filename
        )
        
        return {
            "message": "Report generated successfully",
            "filename": filename,
            "download_url": f"/report/download/{filename}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/download/{filename}", tags=["Reports"])
async def download_report(filename: str):
    """Download generated PDF report (only files inside generated_reports/)."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    base = Path("generated_reports").resolve()
    filepath = (base / filename).resolve()
    try:
        filepath.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid path")

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    if not filepath.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    return FileResponse(
        path=str(filepath),
        filename=filepath.name,
        media_type="application/pdf",
    )


# =============================================================================
# RAG / Chat Endpoints
# =============================================================================

# Note: ChatRequest is already defined above (line ~198). The duplicate was removed to prevent
# the second definition from silently shadowing the first and breaking the /chat endpoint.


class ExplainRequest(BaseModel):
    """Request for result explanation."""
    egfr: float
    gfr_stage: str
    acr: Optional[float] = None
    risk_level: Optional[str] = None


@app.post("/chat", tags=["Chat/RAG"])
async def chat(request: ChatRequest):
    """
    Ask a question about kidney disease.
    
    Uses RAG (Retrieval-Augmented Generation) with medical knowledge base
    and Google Gemini for intelligent responses.
    
    اسأل سؤال عن أمراض الكلى واحصل على إجابة ذكية من قاعدة المعرفة الطبية.
    """
    global rag_engine
    
    if RAG_AVAILABLE and rag_engine is None:
        try:
            from src.rag import GeminiRAG
            rag_engine = GeminiRAG()
        except Exception as e:
            print(f"RAG dynamic initialization failed: {e}")
            
    if not RAG_AVAILABLE or rag_engine is None:
        return {
            "answer": "عذراً دكتور، نظام الذكاء الاصطناعي للمحادثة (RAG) غير متصل حالياً. يرجى التأكد من إضافة مفتاح `GEMINI_API_KEY` في ملف `.env` وتثبيت المكتبات المطلوبة ليعمل النظام بشكل كامل.",
            "sources": [],
            "disclaimer": "النظام في وضع عدم الاتصال (Offline Mode)."
        }
    
    try:
        result = rag_engine.ask(
            question=request.question,
            patient_context=request.patient_context,
            include_sources=True
        )
        
        if result.get("error"):
             return {
                "answer": f"عذراً، حدث خطأ أثناء الاتصال بنموذج اللغة: {result.get('answer')}",
                "sources": [],
                "disclaimer": ""
            }
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "disclaimer": "هذه المعلومات للأغراض التعليمية فقط. يرجى استشارة طبيب للتشخيص والعلاج."
        }
        
    except Exception as e:
        return {
            "answer": f"عذراً، حدث خطأ غير متوقع: {str(e)}",
            "sources": [],
            "disclaimer": ""
        }


@app.post("/explain", tags=["Chat/RAG"])
async def explain_results(request: ExplainRequest):
    """
    Get AI explanation for kidney test results.
    
    احصل على شرح مفصل لنتائج تحاليل الكلى بلغة بسيطة.
    """
    global rag_engine
    
    if RAG_AVAILABLE and rag_engine is None:
        try:
            from src.rag import GeminiRAG
            rag_engine = GeminiRAG()
        except Exception as e:
            print(f"RAG dynamic initialization failed: {e}")
            
    if not RAG_AVAILABLE or rag_engine is None:
        # Provide basic explanation without RAG
        explanations = {
            "G1": "وظائف الكلى طبيعية. لا يوجد ما يدعو للقلق.",
            "G2": "انخفاض طفيف في وظائف الكلى. المتابعة الدورية مطلوبة.",
            "G3a": "انخفاض متوسط. يجب مراجعة طبيب الكلى.",
            "G3b": "انخفاض ملحوظ. المتابعة كل 3 أشهر مطلوبة.",
            "G4": "انخفاض شديد. يحتاج متابعة مكثفة مع طبيب متخصص.",
            "G5": "فشل كلوي. يحتاج غسيل كلى أو زراعة."
        }
        
        return {
            "explanation": explanations.get(request.gfr_stage, "Unknown stage"),
            "egfr": request.egfr,
            "stage": request.gfr_stage,
            "source": "Basic Rules (RAG not available)"
        }
    
    try:
        explanation = rag_engine.explain_result(
            egfr=request.egfr,
            gfr_stage=request.gfr_stage,
            acr=request.acr,
            risk_level=request.risk_level
        )
        
        return {
            "explanation": explanation,
            "egfr": request.egfr,
            "stage": request.gfr_stage,
            "source": "AI-Generated with Medical Knowledge Base"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.get("/chat/status", tags=["Chat/RAG"])
async def chat_status():
    """Check RAG/Chat service status."""
    return {
        "rag_available": RAG_AVAILABLE,
        "rag_initialized": rag_engine is not None,
        "gemini_configured": rag_engine.model is not None if rag_engine else False,
        "knowledge_base_loaded": rag_engine.knowledge_base.collection.count() > 0 if rag_engine else False,
        "message": "Set GEMINI_API_KEY environment variable to enable full RAG functionality"
    }


# =============================================================================
# PDF Report Generation Endpoint
# =============================================================================

class ReportRequest(BaseModel):
    """Request body for generating a PDF report."""
    patient_name: str = Field("Patient", description="Patient name")
    patient_age: int = Field(50, description="Patient age")
    patient_sex: str = Field("male", description="Patient sex")
    prediction: bool = Field(..., description="CKD prediction result")
    probability: float = Field(..., description="Prediction probability")
    risk_level: str = Field(..., description="Risk level string")
    gfr_stage: str = Field(..., description="GFR Stage (G1-G5)")
    egfr: float = Field(..., description="eGFR value")
    albuminuria_category: Optional[str] = Field(None, description="Albuminuria category")
    acr: Optional[float] = Field(None, description="ACR value")
    creatinine: Optional[float] = Field(None, description="Serum creatinine")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations list")
    alerts: List[str] = Field(default_factory=list, description="Alerts list")


@app.post("/report/generate", tags=["Reports"])
async def generate_pdf_report(request: ReportRequest):
    """
    Generate a professional PDF report from prediction results.

    Returns the PDF file as a downloadable attachment.
    """
    if not REPORTS_AVAILABLE or report_generator is None:
        raise HTTPException(status_code=503, detail="PDF report generation is not available. Ensure reportlab is installed.")
    try:
        patient = PatientInfo(
            name=request.patient_name,
            age=request.patient_age,
            sex=request.patient_sex,
            date=datetime.now().strftime("%Y-%m-%d"),
            lab_no="",
            doctor_name=""
        )

        # Build lab results if creatinine is available
        lab_results = []
        if request.creatinine:
            lab_results.append(TestResult(
                name="Serum Creatinine", value=request.creatinine,
                unit="mg/dL", reference_range="0.5 - 1.5",
                is_abnormal=request.creatinine > 1.5
            ))
        lab_results.append(TestResult(
            name="eGFR", value=round(request.egfr, 1),
            unit="mL/min/1.73m\u00b2", reference_range=">90",
            is_abnormal=request.egfr < 60
        ))
        if request.acr:
            lab_results.append(TestResult(
                name="ACR", value=request.acr,
                unit="mg/g", reference_range="<30",
                is_abnormal=request.acr >= 30
            ))

        filepath = report_generator.generate_report(
            patient=patient,
            prediction=request.prediction,
            probability=request.probability,
            risk_level=request.risk_level,
            gfr_stage=request.gfr_stage,
            egfr=request.egfr,
            alb_category=request.albuminuria_category,
            acr=request.acr,
            lab_results=lab_results if lab_results else None,
            recommendations=request.recommendations if request.recommendations else None,
            alerts=request.alerts if request.alerts else None
        )

        return FileResponse(
            path=filepath,
            filename=Path(filepath).name,
            media_type="application/pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


# =============================================================================
# Longitudinal Monitoring Endpoints
# =============================================================================


class MeasurementRequest(BaseModel):
    """Request body for adding a patient measurement."""
    patient_id: str = Field(..., description="Unique patient identifier")
    date: str = Field(..., description="Measurement date (YYYY-MM-DD)")
    egfr: float = Field(..., description="eGFR value")
    creatinine: Optional[float] = Field(None, description="Serum creatinine mg/dL")
    uacr: Optional[float] = Field(None, description="UACR mg/g")
    hba1c: Optional[float] = Field(None, description="HbA1c percentage")
    bp_systolic: Optional[int] = Field(None, description="Systolic blood pressure")
    hemoglobin: Optional[float] = Field(None, description="Hemoglobin g/dL")
    potassium: Optional[float] = Field(None, description="Potassium mmol/L")

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "patient_001",
                "date": "2026-03-18",
                "egfr": 45.0,
                "creatinine": 2.5,
                "uacr": 180.0,
                "hba1c": 8.5,
                "bp_systolic": 160
            }
        }


@app.post("/monitor/add", tags=["Monitoring"])
async def add_measurement(request: MeasurementRequest):
    """
    Record a patient measurement for longitudinal tracking.

    Used by Smart Alerts to detect anomalies and predict future risk.
    """
    if longitudinal_monitor is None:
        raise HTTPException(status_code=503, detail="Longitudinal monitoring is not available.")
    try:
        result = longitudinal_monitor.add_measurement(
            patient_id=request.patient_id,
            date=request.date,
            egfr=request.egfr,
            creatinine=request.creatinine,
            uacr=request.uacr,
            hba1c=request.hba1c
        )
        return {
            "status": "recorded",
            "patient_id": request.patient_id,
            "date": request.date,
            "total_measurements": result.get("total_measurements", 1) if isinstance(result, dict) else 1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record measurement: {str(e)}")


# =============================================================================
# Smart Alerts Endpoints
# =============================================================================


class SymptomRequest(BaseModel):
    """Request body for symptom analysis."""
    text: str = Field(..., min_length=3, description="Patient symptom description (Arabic or English)")
    patient_id: Optional[str] = Field(None, description="Patient ID for correlation with lab data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "حاسس بتورم في رجلي ودوخة وقلة في البول",
                "patient_id": "patient_001"
            }
        }


class AlertAnalyzeRequest(BaseModel):
    """Request body for alert analysis."""
    patient_id: str = Field(..., description="Patient ID")
    include_rule_based: bool = Field(default=True, description="Include rule-based alerts")


@app.post("/alerts/analyze", tags=["Smart Alerts"])
async def analyze_patient_alerts(request: AlertAnalyzeRequest):
    """
    Analyze patient data for anomalies and predictive alerts.
    
    تحليل بيانات المريض للكشف عن التغيرات غير الطبيعية والتنبؤ بالمخاطر.
    
    Combines:
    1. Anomaly Detection (Isolation Forest)
    2. Predictive Analytics (Multi-biomarker trends)
    3. Rule-based alerts (existing)
    """
    if smart_alert_engine is None:
        raise HTTPException(status_code=503, detail="Smart Alert Engine not initialized")
    
    try:
        # Get anomaly detection result
        anomaly = smart_alert_engine.detect_anomalies(request.patient_id)
        
        # Get predictive analytics
        prediction = smart_alert_engine.predict_future_risk(request.patient_id)
        
        # Get rule-based alerts if requested
        rule_alerts = []
        if request.include_rule_based:
            data = longitudinal_monitor.get_patient_history(request.patient_id)
            if data:
                latest = data[-1]
                egfr = latest.get('egfr', 90)
                creatinine = latest.get('creatinine', 1.0)
                uacr = latest.get('uacr')
                rule_alerts = risk_assessor.generate_alerts(
                    egfr=egfr, acr=uacr, creatinine=creatinine,
                    other_values={
                        'hba1c': latest.get('hba1c', 0),
                        'potassium': latest.get('potassium', 0),
                        'hemoglobin': latest.get('hemoglobin', 15),
                        'bp_systolic': latest.get('bp_systolic', 120),
                    }
                )
        
        # Generate combined smart alerts
        alerts = smart_alert_engine.generate_smart_alerts(
            patient_id=request.patient_id,
            include_rule_based=request.include_rule_based,
            rule_based_alerts=rule_alerts
        )
        
        return {
            "patient_id": request.patient_id,
            "total_alerts": len(alerts),
            "alerts": smart_alert_engine.alerts_to_dict(alerts),
            "anomaly_detection": {
                "is_anomaly": anomaly.is_anomaly,
                "severity": anomaly.severity,
                "anomaly_score": anomaly.anomaly_score,
                "anomalous_features": anomaly.anomalous_features
            },
            "predictive_analytics": {
                "risk_score": prediction.overall_risk_score,
                "risk_classification": prediction.risk_classification,
                "timeline": prediction.predicted_timeline,
                "biomarker_trends": prediction.biomarker_trends
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert analysis failed: {str(e)}")


@app.post("/alerts/symptoms", tags=["Smart Alerts"])
async def analyze_symptoms(request: SymptomRequest):
    """
    Analyze patient symptoms using NLP.
    
    تحليل شكوى المريض باللغة العربية أو الإنجليزية واقتراح التشخيص والتوصيات.
    
    Uses Gemini AI (if available) or keyword-based matching.
    """
    if smart_alert_engine is None:
        raise HTTPException(status_code=503, detail="Smart Alert Engine not initialized")
    
    try:
        result = smart_alert_engine.analyze_symptoms(
            text=request.text,
            patient_id=request.patient_id
        )
        
        return {
            "urgency": result.urgency,
            "matched_conditions": result.matched_conditions,
            "recommendations": result.recommendations,
            "correlation_with_labs": result.correlation_with_labs,
            "ai_analysis": result.raw_ai_response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symptom analysis failed: {str(e)}")


@app.get("/alerts/patient/{patient_id}", tags=["Smart Alerts"])
async def get_patient_alerts(patient_id: str):
    """
    Get all smart alerts for a monitored patient.
    
    عرض جميع التنبيهات الذكية لمريض معين مع ترتيبها حسب الأولوية.
    """
    if smart_alert_engine is None:
        raise HTTPException(status_code=503, detail="Smart Alert Engine not initialized")
    
    try:
        # Check if patient exists
        data = longitudinal_monitor.get_patient_history(patient_id)
        if not data:
            return {
                "patient_id": patient_id,
                "total_alerts": 0,
                "alerts": [],
                "message": "لا توجد بيانات مسجلة لهذا المريض"
            }
        
        # Get rule-based alerts from latest data
        latest = data[-1]
        rule_alerts = risk_assessor.generate_alerts(
            egfr=latest.get('egfr', 90),
            acr=latest.get('uacr'),
            creatinine=latest.get('creatinine', 1.0)
        )
        
        # Generate all alerts
        alerts = smart_alert_engine.generate_smart_alerts(
            patient_id=patient_id,
            rule_based_alerts=rule_alerts
        )
        
        return {
            "patient_id": patient_id,
            "total_measurements": len(data),
            "latest_measurement": data[-1],
            "total_alerts": len(alerts),
            "alerts": smart_alert_engine.alerts_to_dict(alerts)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


# Note: The /chat POST endpoint is already registered above (line ~1055). The duplicate
# was removed to prevent FastAPI from silently dropping the second handler.


# =============================================================================
# Diet Planner Endpoint
# =============================================================================

@app.post("/diet/plan", tags=["AI Features"])
async def generate_diet_plan(request: DietRequest):
    """
    🥗 Smart AI Diet Planner

    Generates a personalized, medically accurate 7-day meal plan for a kidney
    disease patient based on their eGFR stage and lab values (Potassium, Sodium).

    Rules are based on KDIGO 2012 clinical practice guidelines for nutritional
    management of CKD.
    """
    if not DIET_PLANNER_AVAILABLE or diet_planner is None or not diet_planner.is_active:
        raise HTTPException(
            status_code=503,
            detail="Smart Diet Planner is unavailable. Ensure GEMINI_API_KEY is configured in .env"
        )

    # Auto-derive stage label from eGFR if not provided
    stage_label = request.stage
    if not stage_label and request.egfr is not None:
        try:
            gfr_stage_obj = gfr_calculator.get_gfr_stage(request.egfr)
            stage_label = gfr_stage_obj.value
        except Exception:
            stage_label = "Unknown"

    patient_data = {
        "age":         request.age,
        "weight":      request.weight,
        "egfr":        request.egfr,
        "stage":       stage_label or "Unknown",
        "potassium":   request.potassium,
        "sodium":      request.sodium,
        "diabetes":    request.diabetes,
        "hypertension":request.hypertension,
    }

    try:
        plan_markdown = diet_planner.generate_diet_plan(patient_data)
        return {
            "status":       "success",
            "stage":        stage_label,
            "egfr":         request.egfr,
            "diet_plan":    plan_markdown,
            "disclaimer":   "تنبيه: هذا النظام الغذائي مقترح من الذكاء الاصطناعي بناءً على إرشادات KDIGO الطبية. يجب مراجعة طبيب التغذية المختص قبل التطبيق."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diet plan generation failed: {str(e)}")


# =============================================================================
# CT Kidney Image Analysis Endpoint
# =============================================================================

@app.post(
    "/predict/ct",
    summary="CT Kidney Image Analysis",
    tags=["Medical Imaging"],
    response_description="Classification result (Normal / Cyst / Stone / Tumor) with confidence score."
)
async def predict_ct_kidney(
    file: UploadFile = File(..., description="CT Scan image of the kidney (JPEG/PNG)")
):
    """
    ## تحليل صور الأشعة المقطعية للكلى 🔬

    يستقبل صورة أشعة مقطعية (CT Scan) للكلى ويُصنفها إلى:
    - **Normal**: كلى سليمة
    - **Cyst**: تكيس كلوي
    - **Stone**: حصوة كلوية
    - **Tumor**: كتلة / ورم

    **المُدخل:** ملف صورة (JPEG أو PNG).

    **المُخرج:** تصنيف المرض + نسبة الثقة + ملاحظة سريرية.

    > ⚠️ هذه الأداة مساعدة للطبيب فقط ولا تُغني عن التشخيص الطبي المتخصص.
    """
    if not CT_CLASSIFIER_AVAILABLE or ct_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="CT Classifier is not available. Please install TensorFlow and run train_ultrasound.py."
        )

    if not ct_classifier.is_ready:
        raise HTTPException(
            status_code=503,
            detail="CT Classifier model has not been trained yet. Run: python train_ultrasound.py"
        )

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Accepted: JPEG, PNG."
        )

    try:
        image_bytes = await file.read()
        result = ct_classifier.predict(image_bytes)

        if "error" in result and result["prediction"] is None:
            raise HTTPException(status_code=500, detail=result["error"])

        return {
            "status":            "success",
            "filename":         file.filename,
            "prediction":       result["prediction"],
            "confidence":       result["confidence"],
            "all_probabilities": result["all_probabilities"],
            "clinical_note":    result["clinical_note"],
            "disclaimer":       (
                "⚕️ هذه النتيجة مُقدَّمة من نموذج ذكاء اصطناعي مُدرَّب على بيانات أشعة مقطعية. "
                "لا تُعدّ بديلاً عن التشخيص الطبي المتخصص."
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CT analysis failed: {str(e)}")


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
