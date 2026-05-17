"""
model.py — Neural network architecture for squat posture classification.

Builds a Conv1D model that processes sliding windows of pose features
to classify squat form quality into 7 classes.

Input:  (batch, WINDOW_SIZE, FEATURE_COUNT) e.g. (batch, 15, 15)
Output: (batch, NUM_CLASSES) softmax probabilities
"""

from src.config import LEARNING_RATE


def build_model(window_size, num_features, num_classes):
    """
    Build a Conv1D temporal classification model.

    Architecture:
      Conv1D(64) → BN → ReLU → MaxPool
      Conv1D(128) → BN → ReLU
      Conv1D(64) → BN → ReLU → GlobalAvgPool
      Dense(64) → Dropout(0.4) → Softmax

    Three conv layers (vs original two) for better temporal pattern recognition.
    """
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(window_size, num_features)),

        # Block 1: Initial temporal features
        keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),

        # Block 2: Higher-level patterns
        keras.layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),

        # Block 3: Refinement
        keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling1D(),

        # Classification head
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
