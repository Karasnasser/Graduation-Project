"""
Deep Learning Models Module
Implements TensorFlow/Keras neural network for kidney disease prediction.
TensorFlow is imported lazily to avoid crashing the entire package when TF
is unavailable or broken — the ML models (RF, XGBoost, SVM) still work.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── Lazy TensorFlow import ────────────────────────────────────────────────────
# TF is only needed at training/inference time, not at import time.
# A broken TF installation will raise an ImportError here; callers must handle it.
tf = None
keras = None
_TF_AVAILABLE = False
try:
    import tensorflow as _tf
    from tensorflow import keras as _keras
    tf = _tf
    keras = _keras
    _TF_AVAILABLE = True
except Exception as _tf_err:
    import warnings as _w
    _w.warn(
        f"[dl_models] TensorFlow unavailable ({type(_tf_err).__name__}: {_tf_err}). "
        "DeepLearningModel will raise RuntimeError when called. "
        "ML models (Random Forest, XGBoost, SVM) are unaffected.",
        ImportWarning,
        stacklevel=2,
    )

def _require_tf():
    """Raise a clear RuntimeError if TensorFlow is not available."""
    if not _TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is required for DeepLearningModel but is not available. "
            "Install a compatible version or use ML models only (Random Forest / XGBoost / SVM)."
        )


class DeepLearningModel:
    """
    Deep Learning model for kidney disease classification using TensorFlow/Keras.
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.history = None
        self.input_shape: Tuple = None
        
    def build_model(
        self,
        input_dim: int,
        hidden_layers: List[int] = None,
        dropout_rate: float = 0.4,
        learning_rate: float = 0.0005,
        output_classes: int = 2
    ):
        """
        Build a neural network model with strong anti-overfitting measures.

        Anti-Overfitting Design:
          - Smaller network (64->32->16)
          - Higher dropout (0.4)
          - Stronger L2 regularization (0.02)
          - Lower learning rate (0.0005)
        """
        _require_tf()
        Sequential = tf.keras.Sequential
        Dense = tf.keras.layers.Dense
        Dropout = tf.keras.layers.Dropout
        BatchNormalization = tf.keras.layers.BatchNormalization
        Input = tf.keras.layers.Input
        Adam = tf.keras.optimizers.Adam
        l2 = tf.keras.regularizers.l2

        if hidden_layers is None:
            hidden_layers = [64, 32, 16]
        self.input_shape = (input_dim,)
        
        model = Sequential()
        model.add(Input(shape=self.input_shape))

        model.add(Dense(hidden_layers[0], activation='relu', kernel_regularizer=l2(0.02)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu', kernel_regularizer=l2(0.02)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        if output_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
        else:
            model.add(Dense(output_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model
        return model

    
    def build_advanced_model(
        self,
        input_dim: int,
        output_classes: int = 2
    ):
        """Build a more advanced model with residual connections."""
        _require_tf()
        Input = tf.keras.layers.Input
        Dense = tf.keras.layers.Dense
        BatchNormalization = tf.keras.layers.BatchNormalization
        Dropout = tf.keras.layers.Dropout
        Model = tf.keras.Model
        Adam = tf.keras.optimizers.Adam
        l2 = tf.keras.regularizers.l2

        inputs = Input(shape=(input_dim,))
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        y = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        y = BatchNormalization()(y)
        y = Dropout(0.3)(y)
        z = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(y)
        z = BatchNormalization()(z)
        z = Dropout(0.2)(z)
        z = Dense(16, activation='relu')(z)
        
        if output_classes == 2:
            outputs = Dense(1, activation='sigmoid')(z)
            loss = 'binary_crossentropy'
        else:
            outputs = Dense(output_classes, activation='softmax')(z)
            loss = 'sparse_categorical_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        self.model = model
        return model
    
    def get_callbacks(
        self,
        patience: int = 10,
        model_checkpoint_path: str = None
    ) -> List:
        """Get training callbacks."""
        _require_tf()
        EarlyStopping = tf.keras.callbacks.EarlyStopping
        ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
        ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        if model_checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    model_checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        return callbacks
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        callbacks: List = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model(input_dim=X_train.shape[1])
        
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        # Determine validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_data = None
        
        print("Training Deep Learning Model...")
        print("=" * 50)
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split if validation_data is None else None,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics — key MUST be 'f1_weighted' to match ML models in _update_weights()
        f1_val = f1_score(y_test, y_pred, average='weighted')
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_val,  # consistent key with ML models → enables correct ensemble weight
            'f1_score': f1_val,     # alias kept for backward compatibility
        }
        
        try:
            metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            metrics['auc_roc'] = 0.0
        
        print("\nDeep Learning Model Evaluation:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_weighted']:.4f}")
        print(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
        
        return metrics

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_proba = self.model.predict(X, verbose=0)
        return (y_proba > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X, verbose=0)
    
    def save_model(self, filename: str = "dl_model.keras"):
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = self.model_dir / filename
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        _require_tf()
        self.model = tf.keras.models.load_model(filepath)
        return self.model
    
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()


class MultiTaskModel(DeepLearningModel):
    """
    Multi-task model for simultaneous CKD detection and stage prediction.
    """
    
    def build_multitask_model(
        self,
        input_dim: int,
        num_stages: int = 6  # G1-G5 + Normal
    ):
        """
        Build a multi-task model with two outputs:
        1. Binary classification (CKD / No CKD)
        2. Stage classification (G1-G5)
        """
        _require_tf()
        Input = tf.keras.layers.Input
        Dense = tf.keras.layers.Dense
        BatchNormalization = tf.keras.layers.BatchNormalization
        Dropout = tf.keras.layers.Dropout
        Model = tf.keras.Model
        Adam = tf.keras.optimizers.Adam
        l2 = tf.keras.regularizers.l2

        inputs = Input(shape=(input_dim,))
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        shared = Dense(32, activation='relu')(x)
        
        # Task 1: CKD Detection (binary)
        ckd_branch = Dense(16, activation='relu')(shared)
        ckd_output = Dense(1, activation='sigmoid', name='ckd_detection')(ckd_branch)
        
        # Task 2: Stage Classification (multiclass)
        stage_branch = Dense(16, activation='relu')(shared)
        stage_output = Dense(num_stages, activation='softmax', name='stage_classification')(stage_branch)
        
        model = Model(inputs=inputs, outputs=[ckd_output, stage_output])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'ckd_detection': 'binary_crossentropy',
                'stage_classification': 'sparse_categorical_crossentropy'
            },
            loss_weights={'ckd_detection': 1.0, 'stage_classification': 0.5},
            metrics={
                'ckd_detection': ['accuracy', tf.keras.metrics.AUC(name='auc')],
                'stage_classification': ['accuracy']
            }
        )
        self.model = model
        return model


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.preprocessing.data_loader import DataLoader

    DATA_DIR = "data/raw"

    loader = DataLoader(data_dir=DATA_DIR)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = loader.load_and_prepare_data()

    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Features: {feature_names}")

    # Create and train model
    dl_model = DeepLearningModel()
    dl_model.build_model(input_dim=X_train.shape[1])
    dl_model.summary()

    # Train with VALIDATION set (not test set)
    history = dl_model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=50,
        batch_size=32
    )

    # Evaluate on held-out TEST set
    metrics = dl_model.evaluate(X_test, y_test)

