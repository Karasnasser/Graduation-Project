"""
train_ultrasound.py
===================
Entry-point script to train the CT Kidney Image Classifier.

Usage:
    python train_ultrasound.py

This will:
    1. Load the 12,446-image CT Kidney Dataset from data/
    2. Train a MobileNetV2-based model in 2 phases (frozen + fine-tune)
    3. Save the best model to models/kidney_ct_classifier.keras
    4. Print final validation accuracy

Training time on CPU: ~30-60 minutes (depending on your machine).
"""
import sys
import os
import warnings
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

DATASET_PATH = (
    Path(__file__).parent
    / "data"
    / "archive (4)"
    / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
)


def main():
    print("=" * 60)
    print("  🏥 Kidnefy-AI: CT Kidney Classifier Training")
    print("=" * 60)
    print(f"\n  Dataset path: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        print(f"\n❌ ERROR: Dataset not found at:\n   {DATASET_PATH}")
        print("\n   Please ensure the dataset is at:")
        print("   data/archive (4)/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/")
        sys.exit(1)

    from src.imaging.kidney_image_classifier import train

    start = time.time()
    model = train(str(DATASET_PATH))
    elapsed = time.time() - start

    print(f"\n⏱️  Total training time: {elapsed/60:.1f} minutes")
    print("\n🎉 Model is ready. You can now start the API server:")
    print("   uvicorn api:app --reload")


if __name__ == "__main__":
    main()
