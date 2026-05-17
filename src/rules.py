"""
rules.py — Hard-coded squat posture rules.

Six rule functions that detect specific form errors using geometric
relationships between landmarks. These provide immediate, explainable
feedback alongside the ML model predictions.

Each function returns None (no issue) or a dict:
  {"code": str, "message": str, "severity": "error"|"warning", "joints": list[int]}
"""

import math
from src.config import (
    LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER,
    LM_LEFT_HIP, LM_RIGHT_HIP,
    LM_LEFT_KNEE, LM_RIGHT_KNEE,
    LM_LEFT_ANKLE, LM_RIGHT_ANKLE,
    LM_LEFT_HEEL, LM_RIGHT_HEEL,
    LM_LEFT_FOOT_INDEX, LM_RIGHT_FOOT_INDEX,
    THRESH_KNEE_CAVE, THRESH_FORWARD_LEAN, THRESH_HEEL_RISE,
    THRESH_KNEES_OVER_TOES, THRESH_SHALLOW_DEPTH,
    THRESH_UNEVEN_HIP, THRESH_UNEVEN_KNEE,
)
from src.utils import get_lm, midpoint, compute_angle, get_body_height, get_body_width


def check_knee_cave(landmarks):
    """
    Detect knees collapsing inward (valgus).
    Compares each knee's X distance from hip center vs ankle's X distance.
    If knee is significantly closer to center than ankle → knee cave.
    """
    mid_hip = midpoint(get_lm(landmarks, LM_LEFT_HIP), get_lm(landmarks, LM_RIGHT_HIP))
    if mid_hip is None:
        return None

    body_w = get_body_width(landmarks)
    issues = []

    # Left knee
    l_knee  = get_lm(landmarks, LM_LEFT_KNEE)
    l_ankle = get_lm(landmarks, LM_LEFT_ANKLE)
    if l_knee and l_ankle:
        knee_dist  = abs(l_knee["x"] - mid_hip["x"])
        ankle_dist = abs(l_ankle["x"] - mid_hip["x"])
        if knee_dist < ankle_dist - body_w * THRESH_KNEE_CAVE:
            issues.append({
                "code": "KNEE_CAVE",
                "message": "Push your LEFT knee outward — it's caving in.",
                "severity": "error",
                "joints": [LM_LEFT_KNEE, LM_LEFT_ANKLE],
            })

    # Right knee
    r_knee  = get_lm(landmarks, LM_RIGHT_KNEE)
    r_ankle = get_lm(landmarks, LM_RIGHT_ANKLE)
    if r_knee and r_ankle:
        knee_dist  = abs(r_knee["x"] - mid_hip["x"])
        ankle_dist = abs(r_ankle["x"] - mid_hip["x"])
        if knee_dist < ankle_dist - body_w * THRESH_KNEE_CAVE:
            issues.append({
                "code": "KNEE_CAVE",
                "message": "Push your RIGHT knee outward — it's caving in.",
                "severity": "error",
                "joints": [LM_RIGHT_KNEE, LM_RIGHT_ANKLE],
            })

    return issues if issues else None


def check_forward_lean(landmarks):
    """
    Detect excessive forward lean.
    Measures the angle of the trunk (shoulder→hip) from vertical.
    """
    mid_shoulder = midpoint(
        get_lm(landmarks, LM_LEFT_SHOULDER),
        get_lm(landmarks, LM_RIGHT_SHOULDER),
    )
    mid_hip = midpoint(
        get_lm(landmarks, LM_LEFT_HIP),
        get_lm(landmarks, LM_RIGHT_HIP),
    )
    if mid_shoulder is None or mid_hip is None:
        return None

    dx = mid_shoulder["x"] - mid_hip["x"]
    dy = mid_shoulder["y"] - mid_hip["y"]  # Y goes down in image coords
    trunk_len = math.sqrt(dx * dx + dy * dy)
    if trunk_len < 1e-6:
        return None

    # Angle from vertical (0, -1)
    cos_angle = (-dy) / trunk_len
    trunk_angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))

    if trunk_angle > THRESH_FORWARD_LEAN:
        return [{
            "code": "FORWARD_LEAN",
            "message": f"Chest up — you're leaning forward ({trunk_angle:.0f}°).",
            "severity": "error",
            "joints": [LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER, LM_LEFT_HIP, LM_RIGHT_HIP],
        }]
    return None


def check_heel_rise(landmarks, baseline):
    """
    Detect heels lifting off the ground.
    Compares current heel Y to standing baseline Y.
    In image coords, Y decreasing = heel rising.
    """
    if not baseline:
        return None

    body_h = get_body_height(landmarks)
    rising = False

    l_heel = get_lm(landmarks, LM_LEFT_HEEL)
    if l_heel and baseline.get("left_heel_y") is not None:
        delta = l_heel["y"] - baseline["left_heel_y"]
        if delta < -body_h * THRESH_HEEL_RISE:
            rising = True

    r_heel = get_lm(landmarks, LM_RIGHT_HEEL)
    if r_heel and baseline.get("right_heel_y") is not None:
        delta = r_heel["y"] - baseline["right_heel_y"]
        if delta < -body_h * THRESH_HEEL_RISE:
            rising = True

    if rising:
        return [{
            "code": "HEEL_RISE",
            "message": "Keep your heels flat on the ground.",
            "severity": "warning",
            "joints": [LM_LEFT_HEEL, LM_RIGHT_HEEL, LM_LEFT_ANKLE, LM_RIGHT_ANKLE],
        }]
    return None


def check_knees_over_toes(landmarks):
    """
    Detect knees extending too far past the toes.
    Uses the Z-axis depth: if knee Z is significantly ahead of foot index Z.
    """
    l_knee = get_lm(landmarks, LM_LEFT_KNEE)
    r_knee = get_lm(landmarks, LM_RIGHT_KNEE)
    l_foot = get_lm(landmarks, LM_LEFT_FOOT_INDEX)
    r_foot = get_lm(landmarks, LM_RIGHT_FOOT_INDEX)

    if not l_knee or not r_knee:
        return None

    over = False
    # Z is negative when closer to camera. Knee forward = more negative Z.
    if l_foot:
        if l_knee.get("z", 0) - l_foot.get("z", 0) < -THRESH_KNEES_OVER_TOES:
            over = True
    if r_foot:
        if r_knee.get("z", 0) - r_foot.get("z", 0) < -THRESH_KNEES_OVER_TOES:
            over = True

    if over:
        return [{
            "code": "KNEES_OVER_TOES",
            "message": "Sit back more — your knees are going past your toes.",
            "severity": "warning",
            "joints": [LM_LEFT_KNEE, LM_RIGHT_KNEE, LM_LEFT_FOOT_INDEX, LM_RIGHT_FOOT_INDEX],
        }]
    return None


def check_shallow_depth(hip_depth_ratio):
    """
    Detect insufficient squat depth.
    Called at the end of a rep with the minimum hip_depth_ratio from the squat.
    Lower ratio = deeper squat. Above threshold = too shallow.
    """
    if hip_depth_ratio is not None and hip_depth_ratio > THRESH_SHALLOW_DEPTH:
        return [{
            "code": "SHALLOW_DEPTH",
            "message": "Go deeper — aim for thighs parallel to the ground.",
            "severity": "warning",
            "joints": [LM_LEFT_HIP, LM_RIGHT_HIP],
        }]
    return None


def check_uneven_weight(landmarks):
    """
    Detect asymmetric weight distribution.
    Checks hip height difference and knee angle asymmetry.
    """
    l_hip   = get_lm(landmarks, LM_LEFT_HIP)
    r_hip   = get_lm(landmarks, LM_RIGHT_HIP)
    l_knee  = get_lm(landmarks, LM_LEFT_KNEE)
    r_knee  = get_lm(landmarks, LM_RIGHT_KNEE)
    l_ankle = get_lm(landmarks, LM_LEFT_ANKLE)
    r_ankle = get_lm(landmarks, LM_RIGHT_ANKLE)

    if not all([l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]):
        return None

    body_h = get_body_height(landmarks)

    # Check hip height difference
    hip_diff = abs(l_hip["y"] - r_hip["y"]) / body_h if body_h > 0.01 else 0
    if hip_diff > THRESH_UNEVEN_HIP:
        return [{
            "code": "UNEVEN_WEIGHT",
            "message": "Balance your weight evenly — one hip is dropping.",
            "severity": "warning",
            "joints": [LM_LEFT_HIP, LM_RIGHT_HIP],
        }]

    # Check knee angle asymmetry
    l_angle = compute_angle(l_hip, l_knee, l_ankle) or 180
    r_angle = compute_angle(r_hip, r_knee, r_ankle) or 180
    if abs(l_angle - r_angle) > THRESH_UNEVEN_KNEE:
        return [{
            "code": "UNEVEN_WEIGHT",
            "message": "Bend both knees equally — weight is uneven.",
            "severity": "warning",
            "joints": [LM_LEFT_KNEE, LM_RIGHT_KNEE],
        }]

    return None


def run_all_rules(landmarks, baseline, is_squatting=False, min_hip_depth=None):
    """
    Run all posture rules and return a list of issue dicts.

    Args:
        landmarks: list[dict] of 33 landmarks.
        baseline: dict with heel baseline Y values.
        is_squatting: whether user is currently in squat position.
        min_hip_depth: minimum hip depth ratio during current rep (for depth check).

    Returns:
        list[dict] of detected issues.
    """
    issues = []

    if not is_squatting:
        return issues

    # Only check form during the squat phase
    for check_fn in [check_knee_cave, check_forward_lean, check_knees_over_toes, check_uneven_weight]:
        result = check_fn(landmarks)
        if result:
            issues.extend(result)

    # Heel rise needs baseline
    result = check_heel_rise(landmarks, baseline)
    if result:
        issues.extend(result)

    return issues
