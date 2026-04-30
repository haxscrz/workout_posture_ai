"""
model.py — Neural Network Architecture for Pose Classification

Builds a Conv1D model that processes sliding windows of pose features
to classify exercise form quality.

Input:  (window_size, num_features)  e.g. (15, 12)
Output: (num_classes,) softmax probabilities
"""

import tensorflow as tf
from tensorflow import keras


def build_conv1d_model(window_size, num_features, num_classes):
    """
    Builds a Conv1D model for temporal pose classification.

    Architecture:
      Conv1D(64) → BN → MaxPool → Conv1D(128) → BN → GAP → Dense(64) → Dropout → Softmax

    This is deliberately lightweight for fast browser inference via TF.js.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(window_size, num_features)),

        # Temporal feature extraction
        keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),

        keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling1D(),

        # Classification head
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
