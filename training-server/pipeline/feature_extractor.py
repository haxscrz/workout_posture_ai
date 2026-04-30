"""
feature_extractor.py — Body-Size-Invariant Feature Extraction (Python)

CRITICAL: This MUST produce EXACTLY the same features as ml/ml-inference.js.
Both use:
  - 3D angle computation (X, Y, Z)
  - Same body width/height normalization
  - Same feature order and count

Features (12 per frame):
  0: leftKneeAngle       (normalized 0-1)
  1: rightKneeAngle      (normalized 0-1)
  2: backAngle           (normalized 0-1)
  3: leftKneeCaveRatio   (knee X offset from ankle / body width)
  4: rightKneeCaveRatio  (ankle X offset from knee / body width)
  5: hipDepthRatio       (ankle-hip vertical distance / body height)
  6: leftHeelDelta       (heel Y change from baseline / body height)
  7: rightHeelDelta      (heel Y change from baseline / body height)
  8: shoulderZDiff       (absolute Z difference between shoulders)
  9: torsoLeanRatio      (shoulder-hip X offset / body width)
  10: hipSymmetry        (left-right hip Y difference / body height)
  11: kneeAngleDiff      (left knee angle - right knee angle)
"""

import math
import mediapipe as mp

mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark

FEATURE_NAMES = [
    'leftKneeAngle',
    'rightKneeAngle',
    'backAngle',
    'leftKneeCaveRatio',
    'rightKneeCaveRatio',
    'hipDepthRatio',
    'leftHeelDelta',
    'rightHeelDelta',
    'shoulderZDiff',
    'torsoLeanRatio',
    'hipSymmetry',
    'kneeAngleDiff',
]

FEATURE_COUNT = len(FEATURE_NAMES)


def get_lm(landmarks, index):
    """Get a landmark by index with visibility check."""
    if not landmarks or index >= len(landmarks.landmark):
        return None
    lm = landmarks.landmark[index]
    if lm.visibility < 0.5:
        return None
    return {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}


def midpoint(p1, p2):
    """Midpoint between two landmarks."""
    if not p1 or not p2:
        return None
    return {
        'x': (p1['x'] + p2['x']) / 2,
        'y': (p1['y'] + p2['y']) / 2,
        'z': (p1['z'] + p2['z']) / 2,
    }


def compute_angle(p1, p2, p3):
    """
    Compute the angle (degrees) at joint p2 formed by the path p1→p2→p3.
    Uses 3D coordinates (X, Y, Z) — MATCHES the JS version exactly.
    """
    if not p1 or not p2 or not p3:
        return None

    v1x = p1['x'] - p2['x']
    v1y = p1['y'] - p2['y']
    v1z = (p1.get('z', 0) or 0) - (p2.get('z', 0) or 0)

    v2x = p3['x'] - p2['x']
    v2y = p3['y'] - p2['y']
    v2z = (p3.get('z', 0) or 0) - (p2.get('z', 0) or 0)

    dot = v1x * v2x + v1y * v2y + v1z * v2z
    mag1 = math.sqrt(v1x**2 + v1y**2 + v1z**2)
    mag2 = math.sqrt(v2x**2 + v2y**2 + v2z**2)

    if mag1 < 1e-6 or mag2 < 1e-6:
        return None

    val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    angle = math.acos(val) * (180.0 / math.pi)
    return angle


def get_body_width(landmarks):
    """Shoulder-to-shoulder width — MATCHES JS getBodyWidth() exactly."""
    l_shoulder = get_lm(landmarks, LM.LEFT_SHOULDER)
    r_shoulder = get_lm(landmarks, LM.RIGHT_SHOULDER)
    if l_shoulder and r_shoulder:
        return abs(l_shoulder['x'] - r_shoulder['x']) or 0.25
    return 0.25


def get_body_height(landmarks):
    """Nose-to-ankle height — MATCHES JS getBodyHeight() exactly."""
    nose = get_lm(landmarks, LM.NOSE)
    l_ankle = get_lm(landmarks, LM.LEFT_ANKLE)
    r_ankle = get_lm(landmarks, LM.RIGHT_ANKLE)
    if not nose:
        return 0.7
    ankle = l_ankle or r_ankle
    if not ankle:
        return 0.7
    return abs(ankle['y'] - nose['y']) or 0.7


def extract_features(landmarks, baseline=None):
    """
    Extract a body-size-invariant feature vector from a single frame.
    MUST match ml/ml-inference.js _extractFeatures() EXACTLY.

    Returns: list of FEATURE_COUNT floats, or None if landmarks are insufficient.
    """
    if baseline is None:
        baseline = {}

    if not landmarks or len(landmarks.landmark) < 33:
        return None

    l_shoulder = get_lm(landmarks, LM.LEFT_SHOULDER)
    r_shoulder = get_lm(landmarks, LM.RIGHT_SHOULDER)
    l_hip      = get_lm(landmarks, LM.LEFT_HIP)
    r_hip      = get_lm(landmarks, LM.RIGHT_HIP)
    l_knee     = get_lm(landmarks, LM.LEFT_KNEE)
    r_knee     = get_lm(landmarks, LM.RIGHT_KNEE)
    l_ankle    = get_lm(landmarks, LM.LEFT_ANKLE)
    r_ankle    = get_lm(landmarks, LM.RIGHT_ANKLE)
    l_heel     = get_lm(landmarks, LM.LEFT_HEEL)
    r_heel     = get_lm(landmarks, LM.RIGHT_HEEL)
    nose       = get_lm(landmarks, LM.NOSE)

    if not all([l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]):
        return None

    mid_shoulder = midpoint(l_shoulder, r_shoulder)
    mid_hip      = midpoint(l_hip, r_hip)
    mid_knee     = midpoint(l_knee, r_knee)

    body_w = get_body_width(landmarks)
    body_h = get_body_height(landmarks)

    # 1. Joint Angles (normalized 0-1, using 3D)
    left_knee_angle  = (compute_angle(l_hip, l_knee, l_ankle) or 180) / 180.0
    right_knee_angle = (compute_angle(r_hip, r_knee, r_ankle) or 180) / 180.0
    back_angle       = (compute_angle(mid_shoulder, mid_hip, mid_knee) or 180) / 180.0

    # 2. Knee Cave Ratio (Mirror Invariant)
    # Distance from knee to centerline vs distance from ankle to centerline
    center_x = mid_hip['x']
    left_knee_dist = abs(l_knee['x'] - center_x)
    left_ankle_dist = abs(l_ankle['x'] - center_x)
    left_knee_cave_ratio = (left_ankle_dist - left_knee_dist) / body_w if body_w > 0.01 else 0

    right_knee_dist = abs(r_knee['x'] - center_x)
    right_ankle_dist = abs(r_ankle['x'] - center_x)
    right_knee_cave_ratio = (right_ankle_dist - right_knee_dist) / body_w if body_w > 0.01 else 0

    # 3. Hip Depth Ratio
    hip_y = mid_hip['y']
    ankle_y = (l_ankle['y'] + r_ankle['y']) / 2
    hip_depth_ratio = (ankle_y - hip_y) / body_h if body_h > 0.01 else 0.5

    # 4. Heel Delta
    left_heel_delta = 0
    if l_heel and baseline.get('leftHeelY') is not None and body_h > 0.01:
        left_heel_delta = (l_heel['y'] - baseline['leftHeelY']) / body_h

    right_heel_delta = 0
    if r_heel and baseline.get('rightHeelY') is not None and body_h > 0.01:
        right_heel_delta = (r_heel['y'] - baseline['rightHeelY']) / body_h

    # 5. Shoulder Z Depth Difference
    shoulder_z_diff = abs((l_shoulder.get('z', 0) or 0) - (r_shoulder.get('z', 0) or 0))

    # 6. Torso Lean Ratio (Mirror invariant - absolute lean)
    torso_lean_ratio = abs(mid_shoulder['x'] - mid_hip['x']) / body_w if body_w > 0.01 else 0

    # 7. Hip Symmetry (Absolute difference)
    hip_symmetry = abs(l_hip['y'] - r_hip['y']) / body_h if body_h > 0.01 else 0

    # 8. Knee Angle Difference
    knee_angle_diff = left_knee_angle - right_knee_angle

    return [
        left_knee_angle,
        right_knee_angle,
        back_angle,
        left_knee_cave_ratio,
        right_knee_cave_ratio,
        hip_depth_ratio,
        left_heel_delta,
        right_heel_delta,
        shoulder_z_diff,
        torso_lean_ratio,
        hip_symmetry,
        knee_angle_diff,
    ]
