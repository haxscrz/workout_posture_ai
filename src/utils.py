"""
utils.py — Pose math utilities.

Pure functions for angle computation, landmark extraction, body measurements,
and landmark smoothing. No MediaPipe dependency — works with list[dict] format.
"""

import math
import numpy as np

from src.config import (
    LM_NOSE, LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER,
    LM_LEFT_ANKLE, LM_RIGHT_ANKLE, VISIBILITY_THRESHOLD,
)


# ─── Landmark Conversion ─────────────────────────────────────────────────────

def landmarks_to_list(pose_landmarks):
    """
    Convert MediaPipe NormalizedLandmarkList to list of dicts.
    Call this once per frame, then pass the list to all other functions.
    """
    return [
        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
        for lm in pose_landmarks.landmark
    ]


# ─── Landmark Access ─────────────────────────────────────────────────────────

def get_lm(landmarks, idx):
    """
    Get landmark dict by index. Returns None if not visible enough.
    landmarks: list of dicts from landmarks_to_list().
    """
    if landmarks is None or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    if lm["visibility"] < VISIBILITY_THRESHOLD:
        return None
    return lm


def are_landmarks_visible(landmarks, ids):
    """Check if all specified landmark IDs are visible."""
    if landmarks is None:
        return False
    return all(get_lm(landmarks, i) is not None for i in ids)


# ─── Core Math ───────────────────────────────────────────────────────────────

def compute_angle(a, b, c):
    """
    Compute the angle (degrees) at joint b formed by path a→b→c.
    Uses 3D coords (x, y, z). Returns None if any point is None.
    """
    if a is None or b is None or c is None:
        return None

    v1 = np.array([
        a["x"] - b["x"],
        a["y"] - b["y"],
        a.get("z", 0) - b.get("z", 0),
    ])
    v2 = np.array([
        c["x"] - b["x"],
        c["y"] - b["y"],
        c.get("z", 0) - b.get("z", 0),
    ])

    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return None

    cos_val = np.clip(np.dot(v1, v2) / (mag1 * mag2), -1.0, 1.0)
    return math.degrees(math.acos(cos_val))


def midpoint(a, b):
    """Midpoint between two landmark dicts."""
    if a is None or b is None:
        return None
    return {
        "x": (a["x"] + b["x"]) / 2,
        "y": (a["y"] + b["y"]) / 2,
        "z": (a.get("z", 0) + b.get("z", 0)) / 2,
    }


# ─── Body Measurements ───────────────────────────────────────────────────────

def get_body_height(landmarks):
    """Nose-to-ankle distance in normalized coords. Fallback: 0.7."""
    nose = get_lm(landmarks, LM_NOSE)
    l_ankle = get_lm(landmarks, LM_LEFT_ANKLE)
    r_ankle = get_lm(landmarks, LM_RIGHT_ANKLE)
    if nose is None:
        return 0.7
    ankle = l_ankle or r_ankle
    if ankle is None:
        return 0.7
    return abs(ankle["y"] - nose["y"]) or 0.7


def get_body_width(landmarks):
    """Shoulder-to-shoulder width in normalized coords. Fallback: 0.25."""
    ls = get_lm(landmarks, LM_LEFT_SHOULDER)
    rs = get_lm(landmarks, LM_RIGHT_SHOULDER)
    if ls is None or rs is None:
        return 0.25
    return abs(ls["x"] - rs["x"]) or 0.25


# ─── Smoothing ────────────────────────────────────────────────────────────────

def smooth_value(prev, current, alpha=0.6):
    """Exponential moving average for a single value."""
    if prev is None:
        return current
    if current is None:
        return prev
    return prev + alpha * (current - prev)


def smooth_landmarks(prev, current, alpha=0.6):
    """
    Smooth landmark positions using EMA.
    Both prev and current are list[dict] from landmarks_to_list().
    Returns new list[dict].
    """
    if prev is None or current is None:
        return current
    if len(prev) != len(current):
        return current

    smoothed = []
    for p, c in zip(prev, current):
        smoothed.append({
            "x": p["x"] + alpha * (c["x"] - p["x"]),
            "y": p["y"] + alpha * (c["y"] - p["y"]),
            "z": p["z"] + alpha * (c["z"] - p["z"]),
            "visibility": p["visibility"] + alpha * (c["visibility"] - p["visibility"]),
        })
    return smoothed
