"""
Project Configuration
الإعدادات المركزية للمشروع - Central settings for all modules.

Usage:
    from config import settings
    model_path = settings.MODEL_DIR
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE any os.getenv calls
load_dotenv()

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "models"
DIABETES_MODEL_DIR = MODEL_DIR / "diabetes"
STAGING_MODEL_DIR = MODEL_DIR / "staging"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJECT_ROOT / "generated_reports"
UPLOADS_DIR = PROJECT_ROOT / "uploads"

# =============================================================================
# Dataset Filenames (edit these if your files have different names)
# =============================================================================
CKD_DATASET = "kidney_disease.csv"
DIABETIC_NEPHROPATHY_DATASET = "Diabetic_Nephropathy_v1.xlsx"
DIABETES_PREDICTION_DATASET = "diabetes_prediction_dataset.csv"

# =============================================================================
# API Settings
# =============================================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
# Comma-separated origins, or "*" for any origin (credentials disabled in api.py when "*")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

# =============================================================================
# Model Settings
# =============================================================================
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =============================================================================
# CKD Feature Defaults (normal values for prediction when missing)
# NOTE: Only EARLY SCREENING features are included.
#       CKD symptom features and direct biomarkers (sc, bu) were removed
#       to force the model to predict early-stage risk rather than 
#       just re-calculating medical diagnosis formulas.
# =============================================================================
CKD_FEATURE_ORDER = [
    'age', 'gender',
    'hba1c', 'sc', 'uacr', 'bu', 'bun',
    'sod', 'pot', 'cal', 'mag'
]

CKD_FEATURE_DEFAULTS = {
    'age': 50, 'gender': 1,
    'hba1c': 5.5, 'sc': 1.0, 'uacr': 10.0, 'bu': 40.0, 'bun': 15.0,
    'sod': 140.0, 'pot': 4.5, 'cal': 9.0, 'mag': 2.0
}

# =============================================================================
# External API Keys
# =============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
