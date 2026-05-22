"""
rules.py — Hard-coded squat posture rules.

Mathematically robust, profile-aware posture checks that detect form errors
using relative geometric relationships. Supports front, side, and oblique views.
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


def get_viewing_angle(landmarks):
    """
    Determine the camera viewing profile: 'side', 'oblique', or 'front'.
    Compares the horizontal width of hips/shoulders to torso trunk length in 2D.
    Uses raw landmark coordinates directly for maximum view-stabilization.
    """
    if landmarks is None or len(landmarks) < 33:
        return 'front'

    l_hip = landmarks[LM_LEFT_HIP]
    r_hip = landmarks[LM_RIGHT_HIP]
    l_shoulder = landmarks[LM_LEFT_SHOULDER]
    r_shoulder = landmarks[LM_RIGHT_SHOULDER]

    hip_w = abs(l_hip["x"] - r_hip["x"])
    shoulder_w = abs(l_shoulder["x"] - r_shoulder["x"])
    
    shoulder_pt = {"x": (l_shoulder["x"] + r_shoulder["x"]) / 2, "y": (l_shoulder["y"] + r_shoulder["y"]) / 2}
    hip_pt = {"x": (l_hip["x"] + r_hip["x"]) / 2, "y": (l_hip["y"] + r_hip["y"]) / 2}

    dx = shoulder_pt["x"] - hip_pt["x"]
    dy = shoulder_pt["y"] - hip_pt["y"]
    trunk_len = math.sqrt(dx * dx + dy * dy)
    
    if trunk_len < 1e-5:
        return 'front'
        
    ratio = max(hip_w, shoulder_w) / trunk_len
    if ratio < 0.15:
        return 'side'
    elif ratio < 0.38:
        return 'oblique'
    else:
        return 'front'


def check_knee_cave(landmarks, view='front'):
    """
    Detect knees collapsing inward (valgus).
    Only reliable in front or oblique views.
    Compares knee horizontal X to ankle horizontal X.
    """
    if view == 'side':
        return None

    body_w = get_body_width(landmarks)
    issues = []

    l_knee  = get_lm(landmarks, LM_LEFT_KNEE)
    l_ankle = get_lm(landmarks, LM_LEFT_ANKLE)
    r_knee  = get_lm(landmarks, LM_RIGHT_KNEE)
    r_ankle = get_lm(landmarks, LM_RIGHT_ANKLE)

    # MediaPipe coordinate space: X increases from screen left to screen right.
    # Left body parts are on screen right (larger X), Right body parts on screen left (smaller X).
    
    # Left knee cave: Left knee moves inward (screen-left, smaller X) relative to left ankle
    if l_knee and l_ankle:
        if l_knee["x"] < l_ankle["x"] - body_w * THRESH_KNEE_CAVE:
            issues.append({
                "code": "KNEE_CAVE",
                "message": "Push your LEFT knee outward — it's collapsing inward.",
                "severity": "error",
                "joints": [LM_LEFT_KNEE, LM_LEFT_ANKLE],
            })

    # Right knee cave: Right knee moves inward (screen-right, larger X) relative to right ankle
    if r_knee and r_ankle:
        if r_knee["x"] > r_ankle["x"] + body_w * THRESH_KNEE_CAVE:
            issues.append({
                "code": "KNEE_CAVE",
                "message": "Push your RIGHT knee outward — it's collapsing inward.",
                "severity": "error",
                "joints": [LM_RIGHT_KNEE, LM_RIGHT_ANKLE],
            })

    return issues if issues else None


def check_forward_lean(landmarks):
    """
    Detect excessive forward lean.
    Measures torso angle from vertical.
    """
    l_shoulder = get_lm(landmarks, LM_LEFT_SHOULDER)
    r_shoulder = get_lm(landmarks, LM_RIGHT_SHOULDER)
    l_hip = get_lm(landmarks, LM_LEFT_HIP)
    r_hip = get_lm(landmarks, LM_RIGHT_HIP)

    if l_shoulder and r_shoulder:
        shoulder_pt = midpoint(l_shoulder, r_shoulder)
    else:
        shoulder_pt = l_shoulder or r_shoulder

    if l_hip and r_hip:
        hip_pt = midpoint(l_hip, r_hip)
    else:
        hip_pt = l_hip or r_hip

    if shoulder_pt is None or hip_pt is None:
        return None

    dx = shoulder_pt["x"] - hip_pt["x"]
    dy = shoulder_pt["y"] - hip_pt["y"]  # Y increases downwards
    trunk_len = math.sqrt(dx * dx + dy * dy)
    if trunk_len < 1e-6:
        return None

    # Angle from vertical (0, -1)
    cos_angle = (-dy) / trunk_len
    trunk_angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))

    if trunk_angle > THRESH_FORWARD_LEAN:
        return [{
            "code": "FORWARD_LEAN",
            "message": f"Chest up — you're leaning forward too much ({trunk_angle:.0f}°).",
            "severity": "error",
            "joints": [LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER, LM_LEFT_HIP, LM_RIGHT_HIP],
        }]
    return None


def check_butt_first(hip_y_history, shoulder_y_history, body_h):
    """
    Detect "Lifting Butt First" (Good Morning squat / hips rising too fast).
    Triggers if hips rise significantly while shoulders stay flat/drop during ascent.
    """
    if not hip_y_history or not shoulder_y_history or len(hip_y_history) < 6:
        return None

    # Delta from oldest (history[0]) to newest (history[-1]) in frame buffer.
    # In screen coordinates, Y decreases as body parts move UP.
    hip_rise = hip_y_history[0] - hip_y_history[-1]
    shoulder_rise = shoulder_y_history[0] - shoulder_y_history[-1]

    # If hips moved up by > 2.5% of body height, but shoulders did not rise (or fell)
    if hip_rise > 0.025 * body_h and shoulder_rise < 0.005 * body_h:
        return [{
            "code": "FORWARD_LEAN",  # Map to FORWARD_LEAN for ML compatibility
            "message": "Lifting butt first! Keep your chest and hips rising together.",
            "severity": "error",
            "joints": [LM_LEFT_HIP, LM_RIGHT_HIP, LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER],
        }]
    return None


def check_heel_rise(landmarks, baseline):
    """
    Detect heels lifting off the floor.
    Self-calibrating: tracks (foot_y - heel_y) compared to standing baseline.
    Uses raw coordinate indexing for maximum reliability.
    """
    if not baseline or landmarks is None or len(landmarks) < 33:
        return None

    body_h = get_body_height(landmarks)
    rising = False

    l_heel = landmarks[LM_LEFT_HEEL]
    l_foot = landmarks[LM_LEFT_FOOT_INDEX]
    r_heel = landmarks[LM_RIGHT_HEEL]
    r_foot = landmarks[LM_RIGHT_FOOT_INDEX]

    # Left side
    if baseline.get("left_heel_y") is not None and baseline.get("left_foot_y") is not None:
        baseline_diff = baseline["left_foot_y"] - baseline["left_heel_y"]
        current_diff = l_foot["y"] - l_heel["y"]
        # If heel rises, it moves up (Y decreases), so current_diff increases
        if current_diff - baseline_diff > body_h * THRESH_HEEL_RISE:
            rising = True

    # Right side
    if baseline.get("right_heel_y") is not None and baseline.get("right_foot_y") is not None:
        baseline_diff = baseline["right_foot_y"] - baseline["right_heel_y"]
        current_diff = r_foot["y"] - r_heel["y"]
        if current_diff - baseline_diff > body_h * THRESH_HEEL_RISE:
            rising = True

    if rising:
        return [{
            "code": "HEEL_RISE",
            "message": "Keep your heels flat on the floor.",
            "severity": "warning",
            "joints": [LM_LEFT_HEEL, LM_RIGHT_HEEL, LM_LEFT_ANKLE, LM_RIGHT_ANKLE],
        }]
    return None


def check_knees_over_toes(landmarks, view='front'):
    """
    Detect knees extending past the toes.
    Sagittal plane metric: Only run in side view (ignores noisy Z coordinates).
    """
    if view != 'side':
        return None

    l_knee = get_lm(landmarks, LM_LEFT_KNEE)
    r_knee = get_lm(landmarks, LM_RIGHT_KNEE)
    l_ankle = get_lm(landmarks, LM_LEFT_ANKLE)
    r_ankle = get_lm(landmarks, LM_RIGHT_ANKLE)
    l_foot = get_lm(landmarks, LM_LEFT_FOOT_INDEX)
    r_foot = get_lm(landmarks, LM_RIGHT_FOOT_INDEX)

    body_h = get_body_height(landmarks)
    over = False
    flagged_joints = []

    # Check both left and right legs if landmarks are visible
    legs_to_check = []
    if l_knee and l_ankle and l_foot:
        legs_to_check.append((l_knee, l_ankle, l_foot, LM_LEFT_KNEE, LM_LEFT_FOOT_INDEX))
    if r_knee and r_ankle and r_foot:
        legs_to_check.append((r_knee, r_ankle, r_foot, LM_RIGHT_KNEE, LM_RIGHT_FOOT_INDEX))

    for knee, ankle, foot, knee_id, foot_id in legs_to_check:
        # Detect facing direction: if foot index X > ankle X, user is facing screen-right.
        if foot["x"] > ankle["x"]:
            # Facing right: knee X should not pass foot X to the right
            if knee["x"] - foot["x"] > body_h * THRESH_KNEES_OVER_TOES:
                over = True
                flagged_joints.extend([knee_id, foot_id])
        else:
            # Facing left: knee X should not pass foot X to the left
            if foot["x"] - knee["x"] > body_h * THRESH_KNEES_OVER_TOES:
                over = True
                flagged_joints.extend([knee_id, foot_id])

    if over:
        return [{
            "code": "KNEES_OVER_TOES",
            "message": "Sit back more — your knees are drifting past your toes.",
            "severity": "warning",
            "joints": flagged_joints if flagged_joints else [LM_LEFT_KNEE, LM_LEFT_FOOT_INDEX],
        }]
    return None


def check_shallow_depth(hip_depth_ratio):
    """
    Detect insufficient squat depth.
    Lower ratio = deeper squat. Above threshold = too shallow.
    """
    if hip_depth_ratio is not None and hip_depth_ratio > THRESH_SHALLOW_DEPTH:
        return [{
            "code": "SHALLOW_DEPTH",
            "message": "Go deeper — try to get your thighs parallel to the floor.",
            "severity": "warning",
            "joints": [LM_LEFT_HIP, LM_RIGHT_HIP],
        }]
    return None


def check_uneven_weight(landmarks, view='front'):
    """
    Detect asymmetric weight distribution.
    Only reliable in front or oblique views.
    """
    if view == 'side':
        return None

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
            "message": "Bend both knees equally — weight distribution is uneven.",
            "severity": "warning",
            "joints": [LM_LEFT_KNEE, LM_RIGHT_KNEE],
        }]

    return None


def run_all_rules(landmarks, baseline, is_squatting=False, hip_y_history=None, shoulder_y_history=None, view=None):
    """
    Run all profile-aware posture rules and return detected issues and the view type.
    Returns:
        (issues, view)
    """
    issues = []

    # 1. Detect view profile dynamically if not provided
    if view is None:
        view = get_viewing_angle(landmarks)
    body_h = get_body_height(landmarks)

    if not is_squatting:
        return issues, view

    # 2. Run standard lateral & transverse posture rules
    result = check_knee_cave(landmarks, view)
    if result:
        issues.extend(result)

    result = check_forward_lean(landmarks)
    if result:
        issues.extend(result)

    result = check_heel_rise(landmarks, baseline)
    if result:
        issues.extend(result)

    result = check_knees_over_toes(landmarks, view)
    if result:
        issues.extend(result)

    result = check_uneven_weight(landmarks, view)
    if result:
        issues.extend(result)

    # 3. Kinetic rules (Butt First)
    result = check_butt_first(hip_y_history, shoulder_y_history, body_h)
    if result:
        issues.extend(result)

    return issues, view
