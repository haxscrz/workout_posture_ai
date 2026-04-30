"""
video_processor.py — Video → Pose Features Pipeline

Reads videos, extracts MediaPipe landmarks frame by frame,
computes feature vectors, and groups them into sliding windows.

Supports:
  - Frame subsampling (process every Nth frame for speed)
  - Video augmentation (horizontal mirror to double training data)
  - Progress reporting
"""

import cv2
import numpy as np
import mediapipe as mp
from pipeline.feature_extractor import extract_features, get_lm, LM, FEATURE_COUNT

mp_pose = mp.solutions.pose


def process_video_for_features(video_path, window_size=15, subsample=2, augment=True, on_progress=None):
    """
    Process a video file into sliding windows of feature vectors.

    Args:
        video_path: Path to the video file
        window_size: Number of frames per sliding window
        subsample: Process every Nth frame (2 = skip every other frame)
        augment: If True, also generate horizontally mirrored windows
        on_progress: Optional callback(current_frame, total_frames)

    Returns:
        list of np.arrays, each shape (window_size, FEATURE_COUNT)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    features_list = []
    features_mirrored = []
    baseline = {}
    baseline_mirror = {}

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        raw_frame_idx = 0
        valid_frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Progress reporting
            if on_progress and raw_frame_idx % 30 == 0:
                on_progress(raw_frame_idx, total_frames)

            # Subsample: skip frames for speed
            if raw_frame_idx % subsample != 0:
                raw_frame_idx += 1
                continue

            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks

                # Set baseline from first valid frame
                if valid_frame_count == 0:
                    l_heel = get_lm(landmarks, LM.LEFT_HEEL)
                    r_heel = get_lm(landmarks, LM.RIGHT_HEEL)
                    if l_heel:
                        baseline['leftHeelY'] = l_heel['y']
                    if r_heel:
                        baseline['rightHeelY'] = r_heel['y']

                features = extract_features(landmarks, baseline)
                if features is not None:
                    features_list.append(features)
                    valid_frame_count += 1

                # Augmentation: horizontal mirror
                if augment:
                    mirrored_frame = cv2.flip(frame, 1)
                    mirrored_image = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
                    mirrored_image.flags.writeable = False

                    # Need a fresh Pose instance for mirrored — reuse existing
                    mirror_results = pose.process(mirrored_image)
                    if mirror_results.pose_landmarks:
                        mirror_lm = mirror_results.pose_landmarks

                        if valid_frame_count == 1:  # First valid mirror frame
                            ml_heel = get_lm(mirror_lm, LM.LEFT_HEEL)
                            mr_heel = get_lm(mirror_lm, LM.RIGHT_HEEL)
                            if ml_heel:
                                baseline_mirror['leftHeelY'] = ml_heel['y']
                            if mr_heel:
                                baseline_mirror['rightHeelY'] = mr_heel['y']

                        mirror_features = extract_features(mirror_lm, baseline_mirror)
                        if mirror_features is not None:
                            features_mirrored.append(mirror_features)

            raw_frame_idx += 1

    cap.release()

    # Create sliding windows from original features
    windows = _create_windows(features_list, window_size)

    # Add mirrored windows
    if augment and len(features_mirrored) >= window_size:
        mirror_windows = _create_windows(features_mirrored, window_size)
        windows.extend(mirror_windows)

    return windows


def _create_windows(features_list, window_size):
    """Create sliding windows with 50% overlap."""
    windows = []
    step = max(1, window_size // 2)  # 50% overlap

    if len(features_list) >= window_size:
        for i in range(0, len(features_list) - window_size + 1, step):
            window = features_list[i:i + window_size]
            windows.append(window)

    return windows
