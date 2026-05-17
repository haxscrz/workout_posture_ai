"""
features.py — Body-size-invariant feature extraction.

Extracts 15 features per frame from pose landmarks. This module is shared
between training (video processing) and real-time inference to guarantee
feature parity — the model always sees the same features it was trained on.

Features:
  0  left_knee_angle        Hip-Knee-Ankle angle, normalized 0-1
  1  right_knee_angle       Hip-Knee-Ankle angle, normalized 0-1
  2  back_angle             Shoulder-Hip-Knee angle, normalized 0-1
  3  left_knee_cave_ratio   Knee inward offset / body width
  4  right_knee_cave_ratio  Knee inward offset / body width
  5  hip_depth_ratio        (ankle_y - hip_y) / body height
  6  left_heel_delta        Heel Y change from baseline / body height
  7  right_heel_delta       Heel Y change from baseline / body height
  8  shoulder_z_diff        |left_shoulder.z - right_shoulder.z|
  9  torso_lean_ratio       |shoulder_x - hip_x| / body width
  10 hip_symmetry           |left_hip.y - right_hip.y| / body height
  11 knee_angle_diff        Left knee angle - right knee angle (signed)
  12 trunk_vertical_angle   Angle of torso from vertical (degrees/180)
  13 knee_forward_ratio     Avg knee Z offset from ankle Z
  14 hip_lateral_shift      Hip center X offset from ankle center X / body width
"""

import math
from src.config import (
    LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER,
    LM_LEFT_HIP, LM_RIGHT_HIP,
    LM_LEFT_KNEE, LM_RIGHT_KNEE,
    LM_LEFT_ANKLE, LM_RIGHT_ANKLE,
    LM_LEFT_HEEL, LM_RIGHT_HEEL,
    LM_LEFT_FOOT_INDEX, LM_RIGHT_FOOT_INDEX,
    FEATURE_COUNT,
)
from src.utils import get_lm, midpoint, compute_angle, get_body_height, get_body_width


def extract_features(landmarks, baseline=None):
    """
    Extract a 15-element feature vector from a single frame.

    Args:
        landmarks: list[dict] with keys x, y, z, visibility (33 landmarks).
        baseline: dict with optional 'left_heel_y' and 'right_heel_y' for
                  heel delta computation. Set from the first standing frame.

    Returns:
        list[float] of length FEATURE_COUNT, or None if landmarks are insufficient.
    """
    if baseline is None:
        baseline = {}

    if landmarks is None or len(landmarks) < 33:
        return None

    # Extract required landmarks
    l_shoulder = get_lm(landmarks, LM_LEFT_SHOULDER)
    r_shoulder = get_lm(landmarks, LM_RIGHT_SHOULDER)
    l_hip      = get_lm(landmarks, LM_LEFT_HIP)
    r_hip      = get_lm(landmarks, LM_RIGHT_HIP)
    l_knee     = get_lm(landmarks, LM_LEFT_KNEE)
    r_knee     = get_lm(landmarks, LM_RIGHT_KNEE)
    l_ankle    = get_lm(landmarks, LM_LEFT_ANKLE)
    r_ankle    = get_lm(landmarks, LM_RIGHT_ANKLE)
    l_heel     = get_lm(landmarks, LM_LEFT_HEEL)
    r_heel     = get_lm(landmarks, LM_RIGHT_HEEL)
    l_foot     = get_lm(landmarks, LM_LEFT_FOOT_INDEX)
    r_foot     = get_lm(landmarks, LM_RIGHT_FOOT_INDEX)

    # All core landmarks must be present
    if not all([l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]):
        return None

    mid_shoulder = midpoint(l_shoulder, r_shoulder)
    mid_hip      = midpoint(l_hip, r_hip)
    mid_knee     = midpoint(l_knee, r_knee)

    body_w = get_body_width(landmarks)
    body_h = get_body_height(landmarks)

    # ── Feature 0-2: Joint angles (normalized 0-1) ──
    left_knee_angle  = (compute_angle(l_hip, l_knee, l_ankle) or 180) / 180.0
    right_knee_angle = (compute_angle(r_hip, r_knee, r_ankle) or 180) / 180.0
    back_angle       = (compute_angle(mid_shoulder, mid_hip, mid_knee) or 180) / 180.0

    # ── Feature 3-4: Knee cave ratio ──
    # Positive = knee is outward of ankle (good), negative = knee caving in (bad)
    center_x = mid_hip["x"]
    l_knee_dist  = abs(l_knee["x"] - center_x)
    l_ankle_dist = abs(l_ankle["x"] - center_x)
    left_knee_cave = (l_ankle_dist - l_knee_dist) / body_w if body_w > 0.01 else 0

    r_knee_dist  = abs(r_knee["x"] - center_x)
    r_ankle_dist = abs(r_ankle["x"] - center_x)
    right_knee_cave = (r_ankle_dist - r_knee_dist) / body_w if body_w > 0.01 else 0

    # ── Feature 5: Hip depth ratio ──
    ankle_y_avg = (l_ankle["y"] + r_ankle["y"]) / 2
    hip_depth = (ankle_y_avg - mid_hip["y"]) / body_h if body_h > 0.01 else 0.5

    # ── Feature 6-7: Heel deltas from baseline ──
    left_heel_delta = 0.0
    if l_heel and baseline.get("left_heel_y") is not None and body_h > 0.01:
        left_heel_delta = (l_heel["y"] - baseline["left_heel_y"]) / body_h

    right_heel_delta = 0.0
    if r_heel and baseline.get("right_heel_y") is not None and body_h > 0.01:
        right_heel_delta = (r_heel["y"] - baseline["right_heel_y"]) / body_h

    # ── Feature 8: Shoulder Z-depth difference ──
    shoulder_z_diff = abs(l_shoulder.get("z", 0) - r_shoulder.get("z", 0))

    # ── Feature 9: Torso lean ratio (mirror invariant) ──
    torso_lean = abs(mid_shoulder["x"] - mid_hip["x"]) / body_w if body_w > 0.01 else 0

    # ── Feature 10: Hip symmetry ──
    hip_symmetry = abs(l_hip["y"] - r_hip["y"]) / body_h if body_h > 0.01 else 0

    # ── Feature 11: Knee angle difference (signed) ──
    knee_angle_diff = left_knee_angle - right_knee_angle

    # ── Feature 12: Trunk vertical angle (normalized 0-1) ──
    # Angle of shoulder→hip line from vertical. 0 = perfectly upright.
    dx = mid_shoulder["x"] - mid_hip["x"]
    dy = mid_shoulder["y"] - mid_hip["y"]  # In image coords, Y goes down
    # Vertical is (0, -1) in image coords (up)
    trunk_len = math.sqrt(dx * dx + dy * dy)
    if trunk_len > 1e-6:
        # cos(angle) = dot(trunk, vertical) / |trunk|
        # vertical = (0, -1), trunk = (dx, dy)
        cos_trunk = (-dy) / trunk_len  # dot product with (0, -1)
        trunk_angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_trunk))))
    else:
        trunk_angle = 0.0
    trunk_vertical = trunk_angle / 90.0  # Normalize: 0=upright, 1=horizontal

    # ── Feature 13: Knee forward ratio (Z-axis) ──
    # How far knees are in front of ankles in the Z dimension
    knee_z_avg  = (l_knee.get("z", 0) + r_knee.get("z", 0)) / 2
    ankle_z_avg = (l_ankle.get("z", 0) + r_ankle.get("z", 0)) / 2
    knee_forward = knee_z_avg - ankle_z_avg  # Negative = knees forward

    # ── Feature 14: Hip lateral shift ──
    # How far hip center is offset from ankle center (lateral sway)
    ankle_x_avg = (l_ankle["x"] + r_ankle["x"]) / 2
    hip_shift = (mid_hip["x"] - ankle_x_avg) / body_w if body_w > 0.01 else 0

    features = [
        left_knee_angle,       # 0
        right_knee_angle,      # 1
        back_angle,            # 2
        left_knee_cave,        # 3
        right_knee_cave,       # 4
        hip_depth,             # 5
        left_heel_delta,       # 6
        right_heel_delta,      # 7
        shoulder_z_diff,       # 8
        torso_lean,            # 9
        hip_symmetry,          # 10
        knee_angle_diff,       # 11
        trunk_vertical,        # 12
        knee_forward,          # 13
        hip_shift,             # 14
    ]

    assert len(features) == FEATURE_COUNT, f"Expected {FEATURE_COUNT} features, got {len(features)}"
    return features
