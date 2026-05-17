"""
train.py — Training pipeline.

Processes squat videos from SQUAT_VIDEOS/, extracts features using MediaPipe,
creates sliding windows, and trains a Conv1D classifier.

Usage:
    python src/main.py --train
"""

import os
import sys
import json
import cv2
import numpy as np
import mediapipe as mp

from src.config import (
    SQUAT_VIDEOS_DIR, MODELS_DIR, MODEL_PATH, LABELS_PATH,
    CLASSES, NUM_CLASSES, VIDEO_LABEL_MAP, SKIP_FILES,
    FEATURE_COUNT, WINDOW_SIZE, SUBSAMPLE_RATE,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_VAL_SPLIT,
    EARLY_STOP_PATIENCE, LEARNING_RATE,
    LM_LEFT_HEEL, LM_RIGHT_HEEL,
)
from src.utils import landmarks_to_list, get_lm
from src.features import extract_features
from src.model import build_model

mp_pose = mp.solutions.pose


def scan_training_videos():
    """
    Walk SQUAT_VIDEOS/ and map each video file to a class label.
    Returns list of (filepath, label_str) tuples.
    """
    training_data = []

    for root, dirs, files in os.walk(SQUAT_VIDEOS_DIR):
        for f in files:
            if not f.lower().endswith((".mp4", ".mov", ".webm", ".avi")):
                continue

            # Skip blacklisted files
            if any(skip in f for skip in SKIP_FILES):
                continue

            filepath = os.path.join(root, f)
            path_str = filepath.replace("\\", "/")

            # Match against VIDEO_LABEL_MAP
            label = None
            for pattern, lbl in VIDEO_LABEL_MAP.items():
                if pattern in path_str:
                    label = lbl
                    break

            if label is not None:
                training_data.append((filepath, label))
            else:
                print(f"  WARNING: Skipping (no label match): {f}")

    return training_data


def process_video(video_path, augment=True):
    """
    Process a single video file into sliding windows of features.

    Returns:
        list of feature windows (each is list of FEATURE_COUNT-length lists)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    features_list = []
    features_mirrored = []
    baseline = {}
    baseline_mirror = {}
    frame_idx = 0
    valid_count = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # Max accuracy for training
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Subsample for speed
            if frame_idx % SUBSAMPLE_RATE != 0:
                frame_idx += 1
                continue

            # Process frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm_list = landmarks_to_list(results.pose_landmarks)

                # Set baseline from first valid frame
                if valid_count == 0:
                    l_heel = get_lm(lm_list, LM_LEFT_HEEL)
                    r_heel = get_lm(lm_list, LM_RIGHT_HEEL)
                    if l_heel:
                        baseline["left_heel_y"] = l_heel["y"]
                    if r_heel:
                        baseline["right_heel_y"] = r_heel["y"]

                feats = extract_features(lm_list, baseline)
                if feats is not None:
                    features_list.append(feats)
                    valid_count += 1

                # Augmentation: horizontal mirror
                if augment:
                    mirrored = cv2.flip(frame, 1)
                    mirrored_rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
                    mirrored_rgb.flags.writeable = False
                    mirror_results = pose.process(mirrored_rgb)

                    if mirror_results.pose_landmarks:
                        mlm = landmarks_to_list(mirror_results.pose_landmarks)

                        if valid_count == 1:  # First mirror frame
                            ml_heel = get_lm(mlm, LM_LEFT_HEEL)
                            mr_heel = get_lm(mlm, LM_RIGHT_HEEL)
                            if ml_heel:
                                baseline_mirror["left_heel_y"] = ml_heel["y"]
                            if mr_heel:
                                baseline_mirror["right_heel_y"] = mr_heel["y"]

                        mfeats = extract_features(mlm, baseline_mirror)
                        if mfeats is not None:
                            features_mirrored.append(mfeats)

            frame_idx += 1

            # Progress
            if frame_idx % 60 == 0:
                pct = int(100 * frame_idx / max(total_frames, 1))
                print(f"    Frame {frame_idx}/{total_frames} ({pct}%)")

    cap.release()

    # Create sliding windows (50% overlap)
    windows = _create_windows(features_list, WINDOW_SIZE)
    if augment and len(features_mirrored) >= WINDOW_SIZE:
        windows.extend(_create_windows(features_mirrored, WINDOW_SIZE))

    return windows


def _create_windows(features_list, window_size):
    """Create sliding windows with 50% overlap."""
    windows = []
    step = max(1, window_size // 2)
    for i in range(0, len(features_list) - window_size + 1, step):
        windows.append(features_list[i : i + window_size])
    return windows


def run_training():
    """Main training function."""
    print("=" * 60)
    print("  Squat Posture AI — Training Pipeline")
    print("=" * 60)

    # 1. Scan videos
    print(f"\nScanning: {SQUAT_VIDEOS_DIR}")
    training_data = scan_training_videos()

    if not training_data:
        print("ERROR: No training videos found!")
        sys.exit(1)

    # Group by label
    label_counts = {}
    for _, label in training_data:
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nFound {len(training_data)} videos:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} video(s)")

    # 2. Extract features from all videos
    print("\n" + "=" * 60)
    print("  Feature Extraction")
    print("=" * 60)

    all_X = []
    all_y = []

    for filepath, label in training_data:
        label_idx = CLASSES.index(label)
        basename = os.path.basename(filepath)
        print(f"\n  Processing: {basename} -> {label}")

        try:
            windows = process_video(filepath, augment=True)
            print(f"    ✓ Extracted {len(windows)} windows")

            for w in windows:
                all_X.append(w)
                all_y.append(label_idx)
        except Exception as e:
            print(f"    ✗ Error: {e}")

    if not all_X:
        print("\nERROR: No features extracted from any video!")
        sys.exit(1)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)

    print(f"\n  Total dataset: {len(X)} samples, shape {X.shape}")

    # 3. Class balancing
    class_counts = np.bincount(y, minlength=NUM_CLASSES)
    total = len(y)
    class_weights = {}
    for i in range(NUM_CLASSES):
        if class_counts[i] > 0:
            class_weights[i] = total / (NUM_CLASSES * class_counts[i])
        else:
            class_weights[i] = 1.0

    print(f"\n  Class distribution:")
    for i, name in enumerate(CLASSES):
        w = class_weights.get(i, 0)
        print(f"    {name}: {class_counts[i]} samples (weight: {w:.2f})")

    # 4. Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # 5. One-hot encode
    import tensorflow as tf
    from tensorflow import keras

    y_onehot = keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

    # 6. Build model
    print("\n" + "=" * 60)
    print("  Model Training")
    print("=" * 60)

    model = build_model(WINDOW_SIZE, FEATURE_COUNT, NUM_CLASSES)
    model.summary()

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    lr_reduce = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1,
    )

    # 7. Train
    history = model.fit(
        X, y_onehot,
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        validation_split=TRAIN_VAL_SPLIT,
        class_weight=class_weights,
        callbacks=[early_stop, lr_reduce],
        verbose=1,
    )

    # 8. Results
    actual_epochs = len(history.history["loss"])
    final_val_acc = history.history.get("val_accuracy", [0])[-1]
    final_val_loss = history.history.get("val_loss", [0])[-1]

    print(f"\n  ✓ Training complete after {actual_epochs} epochs")
    print(f"  Final val_accuracy: {final_val_acc:.4f} ({final_val_acc * 100:.1f}%)")
    print(f"  Final val_loss:     {final_val_loss:.4f}")

    # 9. Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(CLASSES, f)

    print(f"\n  Model saved to: {MODEL_PATH}")
    print(f"  Labels saved to: {LABELS_PATH}")
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
