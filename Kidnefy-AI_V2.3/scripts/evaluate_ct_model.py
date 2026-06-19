"""
evaluate_ct_model.py
=====================
Generates a full professional evaluation report for the trained CT Kidney Classifier:
  - Confusion Matrix heatmap
  - Per-class Classification Report (Precision, Recall, F1)
  - Training history curves (Accuracy & Loss)
  - Saves all outputs to models/evaluation/

Usage:
    python evaluate_ct_model.py
"""

import sys
import os
import json
import warnings
import numpy as np
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')   # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

import tensorflow as tf
from src.imaging.kidney_image_classifier import (
    CLASSES, MODEL_SAVE_PATH, HISTORY_PATH, IMG_SIZE, BATCH_SIZE
)

OUTPUT_DIR = Path("models") / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = (
    Path("data")
    / "archive (4)"
    / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
)


# ── 1. Load Model ─────────────────────────────────────────────────────────────
def load_trained_model():
    if not MODEL_SAVE_PATH.exists():
        print(f"❌ Model not found at {MODEL_SAVE_PATH}. Run train_ultrasound.py first.")
        sys.exit(1)
    print(f"[INFO] Loading model from: {MODEL_SAVE_PATH}")
    return tf.keras.models.load_model(str(MODEL_SAVE_PATH))


# ── 2. Validation Dataset ─────────────────────────────────────────────────────
def load_validation_data(model):
    """
    Rebuilds the same stratified per-class 80/20 split used during training.
    Returns (val_dataset, y_true_array, y_pred_array) directly.
    """
    import random
    from src.imaging.kidney_image_classifier import CLASSES, IMG_SIZE, BATCH_SIZE

    val_files, val_labels = [], []
    for class_idx, class_name in enumerate(CLASSES):
        cls_path = DATASET_PATH / class_name
        if not cls_path.exists():
            continue
        files = sorted(cls_path.glob("*.jpg")) + sorted(cls_path.glob("*.png"))
        random.seed(42)
        random.shuffle(files)
        split_idx = int(len(files) * 0.8)
        val_part = files[split_idx:]
        val_files.extend([str(f) for f in val_part])
        val_labels.extend([class_idx] * len(val_part))
        print(f"   Validation {class_name}: {len(val_part)} images")

    print(f"\n[INFO] Total validation: {len(val_files)} images")

    # Build dataset
    def load_img_tf(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    AUTOTUNE = tf.data.AUTOTUNE
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_files, val_labels))
        .map(load_img_tf, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    return val_ds, np.array(val_labels)


# ── 3. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(model, val_ds, y_true):
    print("[INFO] Computing predictions for confusion matrix...")
    y_pred = []

    for images, _ in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))

    y_pred = np.array(y_pred)

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    print("\n" + "="*60)
    print("  Classification Report")
    print("="*60)
    print(report)

    report_path = OUTPUT_DIR / "classification_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"CT Kidney Classifier — Evaluation Report\n")
        f.write(f"Dataset: 12,446 CT images (4 classes)\n\n")
        f.write(report)
    print(f"[INFO] Classification report saved: {report_path}")

    # Confusion Matrix heatmap
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalized

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('CT Kidney Classifier — Confusion Matrix', fontsize=14, fontweight='bold')

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title('Raw Counts')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1])
    axes[1].set_title('Normalized (% per True Class)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved: {cm_path}")

    return y_true, y_pred

# ── 4. Training History Curves ────────────────────────────────────────────────
def plot_training_history():
    if not HISTORY_PATH.exists():
        print(f"[WARN] No training history found at {HISTORY_PATH}. Skipping curves.")
        return

    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['accuracy']) + 1)
    frozen_end = 10  # Phase 1 ended at epoch 10

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('CT Kidney Classifier — Training History', fontsize=14, fontweight='bold')

    colors = {'train': '#2196F3', 'val': '#FF5722'}

    for ax, metric, title in zip(axes, ['accuracy', 'loss'], ['Accuracy', 'Loss']):
        ax.plot(epochs, history[metric], color=colors['train'],
                label=f'Training {title}', linewidth=2)
        ax.plot(epochs, history[f'val_{metric}'], color=colors['val'],
                label=f'Validation {title}', linewidth=2, linestyle='--')

        # Mark Phase 2 start
        if frozen_end < len(epochs):
            ax.axvline(x=frozen_end + 0.5, color='gray', linestyle=':', linewidth=1.5)
            ax.text(frozen_end + 0.7, ax.get_ylim()[0], 'Fine-tune →',
                    color='gray', fontsize=9, va='bottom')

        ax.set_title(f'{title} Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if metric == 'accuracy':
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    history_path = OUTPUT_DIR / "training_history.png"
    plt.savefig(history_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Training history curves saved: {history_path}")


# ── 5. Per-Class Accuracy Bar Chart ──────────────────────────────────────────
def plot_per_class_accuracy(y_true, y_pred):
    per_class_acc = []
    for i in range(len(CLASSES)):
        mask = y_true == i
        if mask.sum() > 0:
            per_class_acc.append((y_pred[mask] == i).sum() / mask.sum())
        else:
            per_class_acc.append(0.0)

    colors = ['#4CAF50' if a >= 0.85 else '#FF9800' if a >= 0.70 else '#F44336'
              for a in per_class_acc]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CLASSES, [a * 100 for a in per_class_acc], color=colors, width=0.5)

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.5,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax.set_title('Per-Class Validation Accuracy', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    class_acc_path = OUTPUT_DIR / "per_class_accuracy.png"
    plt.savefig(class_acc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Per-class accuracy chart saved: {class_acc_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  📊 CT Kidney Classifier — Full Evaluation Report")
    print("=" * 60)

    model          = load_trained_model()
    val_ds, y_true = load_validation_data(model)

    y_true, y_pred = plot_confusion_matrix(model, val_ds, y_true)
    plot_training_history()
    plot_per_class_accuracy(y_true, y_pred)

    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print(f"   All outputs saved to: {OUTPUT_DIR.resolve()}")
    print("   Files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"     - {f.name}")
    print("=" * 60)
