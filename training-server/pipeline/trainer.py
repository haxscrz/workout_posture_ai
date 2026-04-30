"""
trainer.py — Training Pipeline Manager

Handles the full training pipeline:
  1. Scan data directory for labeled videos
  2. Extract features from each video using MediaPipe
  3. Build windowed dataset with class balancing
  4. Train Conv1D model with early stopping
  5. Export to TensorFlow.js format
  6. Generate confusion matrix report
"""

import os
import json
import time
import queue
import threading
import numpy as np
import tensorflow as tf
from tensorflow import keras

from pipeline.video_processor import process_video_for_features
from pipeline.model import build_conv1d_model
from pipeline.feature_extractor import FEATURE_COUNT


class TrainingCallback(keras.callbacks.Callback):
    """Sends training metrics to the SSE message queue."""

    def __init__(self, message_queue, total_epochs):
        super().__init__()
        self.message_queue = message_queue
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.message_queue.put({
            'type': 'epoch',
            'epoch': epoch + 1,
            'total_epochs': self.total_epochs,
            'loss': float(logs.get('loss', 0)),
            'accuracy': float(logs.get('accuracy', 0)),
            'val_loss': float(logs.get('val_loss', 0)),
            'val_accuracy': float(logs.get('val_accuracy', 0))
        })


def extract_all_videos(data_dir, message_queue, window_size=15):
    """
    Scan data_dir for all exercises and labels, extract features.

    Directory structure:
      data_dir/<exercise>/<label>/video1.mp4
      data_dir/squat/GOOD_FORM/vid1.mp4
      data_dir/squat/KNEES_CAVING/vid2.mp4
    """
    all_data = {}  # exercise -> {X, y, label_names}

    if not os.path.exists(data_dir):
        message_queue.put({'type': 'log', 'message': f'Data directory not found: {data_dir}'})
        return all_data

    for exercise in sorted(os.listdir(data_dir)):
        exercise_dir = os.path.join(data_dir, exercise)
        if not os.path.isdir(exercise_dir):
            continue

        label_names = sorted([
            d for d in os.listdir(exercise_dir)
            if os.path.isdir(os.path.join(exercise_dir, d))
        ])

        if len(label_names) < 2:
            message_queue.put({
                'type': 'log',
                'message': f'Skipping {exercise}: need at least 2 labels, found {len(label_names)}'
            })
            continue

        X = []
        y = []

        for idx, label in enumerate(label_names):
            label_dir = os.path.join(exercise_dir, label)
            videos = [
                f for f in os.listdir(label_dir)
                if f.lower().endswith(('.mp4', '.mov', '.webm', '.avi', '.mkv'))
            ]

            if not videos:
                message_queue.put({'type': 'log', 'message': f'  No videos for {exercise}/{label}'})
                continue

            for vid in videos:
                vid_path = os.path.join(label_dir, vid)
                message_queue.put({'type': 'log', 'message': f'  Processing: {exercise}/{label}/{vid}'})

                try:
                    def on_progress(current, total):
                        message_queue.put({
                            'type': 'log',
                            'message': f'    Frame {current}/{total}...'
                        })

                    windows = process_video_for_features(
                        vid_path, window_size,
                        subsample=2, augment=True,
                        on_progress=on_progress
                    )

                    for w in windows:
                        X.append(w)
                        y.append(idx)

                    message_queue.put({
                        'type': 'log',
                        'message': f'    ✓ Extracted {len(windows)} windows (with augmentation)'
                    })
                except Exception as e:
                    message_queue.put({
                        'type': 'log',
                        'message': f'    ✗ Error: {str(e)}'
                    })

        if len(X) > 0:
            all_data[exercise] = {
                'X': np.array(X, dtype=np.float32),
                'y': np.array(y, dtype=np.int32),
                'label_names': label_names,
            }
            message_queue.put({
                'type': 'log',
                'message': f'  Dataset for {exercise}: {len(X)} samples, {len(label_names)} classes'
            })

    return all_data


def train_model_thread(data_dir, export_dir, message_queue, epochs=80):
    """Main training function — runs in a background thread."""
    try:
        # 1. Process Data
        message_queue.put({'type': 'log', 'message': '═══ Starting Feature Extraction ═══'})
        all_data = extract_all_videos(data_dir, message_queue)

        if not all_data:
            message_queue.put({
                'type': 'error',
                'message': 'No training data found. Upload videos first, organized as: data/<exercise>/<label>/video.mp4'
            })
            return

        # 2. Train a model for each exercise
        for exercise, dataset in all_data.items():
            X = dataset['X']
            y = dataset['y']
            label_names = dataset['label_names']
            num_classes = len(label_names)

            message_queue.put({
                'type': 'log',
                'message': f'\n═══ Training Model: {exercise} ═══'
            })
            message_queue.put({
                'type': 'log',
                'message': f'  Samples: {len(X)}, Classes: {num_classes}, Window: {X.shape[1]}×{X.shape[2]}'
            })

            # Shuffle data
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            # Compute class weights for imbalanced data
            class_counts = np.bincount(y, minlength=num_classes)
            total = len(y)
            class_weights = {}
            for i in range(num_classes):
                if class_counts[i] > 0:
                    class_weights[i] = total / (num_classes * class_counts[i])
                else:
                    class_weights[i] = 1.0

            message_queue.put({
                'type': 'log',
                'message': f'  Class distribution: {dict(zip(label_names, class_counts.tolist()))}'
            })
            message_queue.put({
                'type': 'log',
                'message': f'  Class weights: {dict(zip(label_names, [f"{w:.2f}" for w in class_weights.values()]))}'
            })

            # One-hot encode
            y_one_hot = keras.utils.to_categorical(y, num_classes=num_classes)

            # Build model
            window_size = X.shape[1]
            num_features = X.shape[2]
            model = build_conv1d_model(window_size, num_features, num_classes)

            # Early stopping
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            )

            # Learning rate reduction on plateau
            lr_reduce = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=0
            )

            # Training callback for SSE
            cb = TrainingCallback(message_queue, epochs)

            message_queue.put({'type': 'log', 'message': f'  Starting training ({epochs} max epochs)...'})

            # Train
            history = model.fit(
                X, y_one_hot,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                class_weight=class_weights,
                callbacks=[cb, early_stop, lr_reduce],
                verbose=0
            )

            # Get final metrics
            actual_epochs = len(history.history['loss'])
            final_val_acc = history.history.get('val_accuracy', [0])[-1]
            final_val_loss = history.history.get('val_loss', [0])[-1]

            message_queue.put({
                'type': 'log',
                'message': f'  ✓ Training complete after {actual_epochs} epochs'
            })
            message_queue.put({
                'type': 'log',
                'message': f'  Final val_accuracy: {final_val_acc:.4f} ({final_val_acc*100:.1f}%)'
            })
            message_queue.put({
                'type': 'log',
                'message': f'  Final val_loss: {final_val_loss:.4f}'
            })

            # 3. Export to TensorFlow.js
            message_queue.put({'type': 'log', 'message': f'  Exporting model to TensorFlow.js...'})

            try:
                import tensorflowjs as tfjs

                target_dir = os.path.join(export_dir, exercise)
                os.makedirs(target_dir, exist_ok=True)

                # Export model
                tfjs.converters.save_keras_model(model, target_dir)

                # Save labels
                with open(os.path.join(target_dir, 'labels.json'), 'w') as f:
                    json.dump(label_names, f)

                message_queue.put({
                    'type': 'log',
                    'message': f'  ✓ Model exported to {target_dir}'
                })
            except ImportError:
                message_queue.put({
                    'type': 'log',
                    'message': '  ⚠ tensorflowjs not installed. Run: pip install tensorflowjs'
                })
                # Save as Keras h5 instead
                h5_path = os.path.join(export_dir, f'{exercise}_model.h5')
                model.save(h5_path)
                message_queue.put({
                    'type': 'log',
                    'message': f'  Saved Keras model to {h5_path} instead'
                })

        message_queue.put({
            'type': 'complete',
            'message': '═══ All models trained and exported successfully! ═══'
        })

    except Exception as e:
        import traceback
        message_queue.put({
            'type': 'error',
            'message': f'Training failed: {str(e)}\n{traceback.format_exc()}'
        })


class TrainingManager:
    """Manages training state and SSE message streaming."""

    def __init__(self, data_dir, export_dir):
        self.data_dir = data_dir
        self.export_dir = export_dir
        self.message_queue = queue.Queue()
        self.is_training = False

    def start_training(self):
        if self.is_training:
            return False

        # Clear old messages
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                break

        self.is_training = True

        def run_wrapper():
            train_model_thread(self.data_dir, self.export_dir, self.message_queue)
            self.is_training = False

        thread = threading.Thread(target=run_wrapper)
        thread.daemon = True
        thread.start()
        return True

    def get_messages(self):
        """Generator that yields SSE messages."""
        while True:
            try:
                msg = self.message_queue.get(timeout=2.0)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get('type') in ['complete', 'error']:
                    break
            except queue.Empty:
                if not self.is_training:
                    break
                yield ": keepalive\n\n"
