"""
kidney_image_classifier.py
==========================
Professional-grade CT Kidney Image Classification Engine.

Architecture: Transfer Learning using MobileNetV2 (optimized for CPU training).
Dataset:      CT-KIDNEY-DATASET (Normal, Cyst, Tumor, Stone) — 12,446 images.
Technique:    Fine-tuning top layers + Class-weight balancing for Stone class.

Classes:
    - Normal (5,077 images)
    - Cyst   (3,709 images)
    - Tumor  (2,283 images)
    - Stone  (1,377 images)

This module is used both for training (train_ultrasound.py) and inference (api.py).
"""

import os
import json
import warnings
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Suppress verbose TF warnings for clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)

# ── Constants ───────────────────────────────────────────────────────────────
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 16        # CPU-friendly batch size
EPOCHS_FROZEN   = 10        # Train only the new head first
EPOCHS_FINETUNE = 10        # Then unfreeze top layers of base
CLASSES         = ['Cyst', 'Normal', 'Stone', 'Tumor']  # sorted alphabetically (keras convention)
MODEL_SAVE_PATH = Path(__file__).parent.parent.parent / "models" / "kidney_ct_classifier.keras"
HISTORY_PATH    = Path(__file__).parent.parent.parent / "models" / "training_history.json"


# ── Data Pipeline ────────────────────────────────────────────────────────────
def build_data_pipelines(dataset_dir: str):
    """
    Creates TensorFlow data pipelines with STRATIFIED per-class split.
    This ensures all 4 classes are represented in both train and val sets.
    """
    import glob, random
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    print(f"\n[INFO] Loading dataset from: {dataset_path}")

    # Collect files per class with stratified 80/20 split
    train_files, train_labels = [], []
    val_files,   val_labels   = [], []

    for class_idx, class_name in enumerate(CLASSES):
        cls_path = dataset_path / class_name
        if not cls_path.exists():
            print(f"[WARN] Class folder not found: {cls_path}")
            continue
        files = sorted(cls_path.glob("*.jpg")) + sorted(cls_path.glob("*.png"))
        random.seed(42)
        random.shuffle(files)
        split_idx = int(len(files) * 0.8)
        train_part = files[:split_idx]
        val_part   = files[split_idx:]
        train_files.extend([str(f) for f in train_part])
        train_labels.extend([class_idx] * len(train_part))
        val_files.extend([str(f) for f in val_part])
        val_labels.extend([class_idx] * len(val_part))
        print(f"       {class_name}: {len(train_part)} train | {len(val_part)} val")

    print(f"\n[INFO] Total: {len(train_files)} train | {len(val_files)} val")

    def load_and_preprocess(path, label, augment=False):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, 0.15)
            img = tf.image.random_contrast(img, 0.85, 1.15)
        label_oh = tf.one_hot(label, depth=len(CLASSES))
        return img, label_oh

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_files, train_labels))
        .shuffle(len(train_files), seed=42)
        .map(lambda p, l: load_and_preprocess(p, l, augment=True),
             num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_files, val_labels))
        .map(lambda p, l: load_and_preprocess(p, l, augment=False),
             num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds


def compute_class_weights(dataset_dir: str) -> dict:
    """Compute class weights to handle class imbalance (Stone has fewer images)."""
    from sklearn.utils.class_weight import compute_class_weight
    
    dataset_path = Path(dataset_dir)
    labels = []
    for i, cls in enumerate(CLASSES):
        cls_path = dataset_path / cls
        if cls_path.exists():
            n = len(list(cls_path.glob("*.*")))
            labels.extend([i] * n)
    
    labels = np.array(labels)
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return {i: w for i, w in enumerate(weights)}


# ── Model Architecture ───────────────────────────────────────────────────────
def build_model(num_classes: int = 4) -> Model:
    """
    Builds a Transfer Learning model using MobileNetV2.
    
    Architecture:
        MobileNetV2 (frozen) → GlobalAveragePooling2D → BatchNorm →
        Dense(256, relu) → Dropout(0.4) → Dense(num_classes, softmax)
    
    MobileNetV2 is chosen for optimal CPU training speed with high accuracy.
    """
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze in Phase 1

    inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x       = base_model(inputs, training=False)
    x       = GlobalAveragePooling2D()(x)
    x       = BatchNormalization()(x)
    x       = Dense(256, activation='relu')(x)
    x       = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="KidneyCT_Classifier")
    return model, base_model


# ── Training Pipeline ─────────────────────────────────────────────────────────
def train(dataset_dir: str):
    """
    Full professional training pipeline:
    Phase 1: Train only the new classification head (10 epochs).
    Phase 2: Fine-tune top layers of MobileNetV2 (10 epochs).
    """
    MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("  🧠 Kidnefy-AI: CT Image Classifier Training")
    print("="*60)

    # 1. Build data pipelines
    train_ds, val_ds = build_data_pipelines(dataset_dir)
    class_weights    = compute_class_weights(dataset_dir)
    
    print(f"\n[INFO] Class Weights (for Stone imbalance): {class_weights}")

    # 2. Build model
    model, base_model = build_model(num_classes=len(CLASSES))
    model.summary()

    # 3. ── PHASE 1: Train head only ─────────────────────────────────────────
    print("\n[PHASE 1] Training classification head (base model frozen)...")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_phase1 = [
        EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(str(MODEL_SAVE_PATH), monitor='val_accuracy',
                        save_best_only=True, verbose=1),
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FROZEN,
        class_weight=class_weights,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    # 4. ── PHASE 2: Fine-tune top 30 layers of base ──────────────────────────
    print("\n[PHASE 2] Fine-tuning top 30 layers of MobileNetV2...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # Much lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_phase2 = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(str(MODEL_SAVE_PATH), monitor='val_accuracy',
                        save_best_only=True, verbose=1),
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        class_weight=class_weights,
        callbacks=callbacks_phase2,
        verbose=1,
    )

    # 5. Save training history for later plotting
    combined_history = {
        'accuracy':     history1.history['accuracy']     + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss':         history1.history['loss']         + history2.history['loss'],
        'val_loss':     history1.history['val_loss']     + history2.history['val_loss'],
    }
    with open(HISTORY_PATH, 'w') as f:
        json.dump(combined_history, f, indent=2)

    print(f"\n✅ Training complete! Model saved to: {MODEL_SAVE_PATH}")
    print(f"   Best val_accuracy: {max(combined_history['val_accuracy']):.4f}")
    
    # 6. Final evaluation
    print("\n[INFO] Evaluating on validation set...")
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"   Final Validation Accuracy: {acc*100:.2f}%")
    print(f"   Final Validation Loss:     {loss:.4f}")
    
    return model


# ── Inference Engine ──────────────────────────────────────────────────────────
class KidneyImageClassifier:
    """
    Production-ready inference engine for CT kidney image classification.
    Loads the trained model and exposes a single `predict(image_bytes)` method.
    """
    _instance = None  # Singleton

    def __init__(self):
        self.model = None
        self.classes = CLASSES
        self.is_ready = False
        self._load_model()

    def _load_model(self):
        if MODEL_SAVE_PATH.exists():
            try:
                self.model = load_model(str(MODEL_SAVE_PATH))
                self.is_ready = True
                print(f"[INFO] CT Classifier loaded from: {MODEL_SAVE_PATH}")
            except Exception as e:
                print(f"[WARN] Failed to load CT Classifier: {e}")
                self.is_ready = False
        else:
            print(f"[WARN] No trained CT model found at {MODEL_SAVE_PATH}.")
            print("       Run 'python train_ultrasound.py' first to train the model.")
            self.is_ready = False

    def predict(self, image_bytes: bytes) -> dict:
        """
        Runs inference on raw image bytes (from API upload).
        
        Returns:
            {
                "prediction": "Stone",
                "confidence": 0.9423,
                "all_probabilities": {"Cyst": 0.02, "Normal": 0.01, "Stone": 0.94, "Tumor": 0.02},
                "clinical_note": "..."
            }
        """
        if not self.is_ready:
            return {
                "error": "CT Classifier model not trained yet. Run train_ultrasound.py first.",
                "prediction": None,
                "confidence": None
            }

        try:
            import io
            img = load_img(io.BytesIO(image_bytes), target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = self.model.predict(img_array, verbose=0)[0]
            predicted_class = self.classes[np.argmax(preds)]
            confidence      = float(np.max(preds))

            all_probs = {cls: float(round(p, 4)) for cls, p in zip(self.classes, preds)}

            clinical_notes = {
                "Cyst":   "كيس كلوي بسيط (Simple Renal Cyst) — يستدعي المتابعة الدورية.",
                "Normal": "لا توجد علامات واضحة لاعتلال الكلى في الأشعة.",
                "Stone":  "حصوة كلوية (Nephrolithiasis) — يُنصح بالتدخل الطبي الفوري.",
                "Tumor":  "كتلة مشتبه بها (Renal Mass) — يستدعي مراجعة طارئة لطبيب أورام.",
            }

            return {
                "prediction":       predicted_class,
                "confidence":       round(confidence, 4),
                "all_probabilities": all_probs,
                "clinical_note":    clinical_notes[predicted_class],
            }
        except Exception as e:
            return {"error": str(e), "prediction": None, "confidence": None}
