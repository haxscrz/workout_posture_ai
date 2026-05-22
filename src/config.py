"""
config.py — Central configuration for Squat Posture AI.

All thresholds, paths, class definitions, and hyperparameters live here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQUAT_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "SQUAT_VIDEOS")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "squat_model.keras")
LABELS_PATH = os.path.join(MODELS_DIR, "squat_labels.json")

# ─── Posture Classes (7 total) ───────────────────────────────────────────────
CLASSES = [
    "CORRECT",
    "FORWARD_LEAN",
    "HEEL_RISE",
    "KNEE_CAVE",
    "KNEES_OVER_TOES",
    "SHALLOW_DEPTH",
    "UNEVEN_WEIGHT",
]
NUM_CLASSES = len(CLASSES)

# Human-readable messages for each error class
CLASS_MESSAGES = {
    "CORRECT":        "Good form! Keep it up.",
    "FORWARD_LEAN":   "Chest up — you're leaning too far forward.",
    "HEEL_RISE":      "Keep your heels flat on the ground.",
    "KNEE_CAVE":      "Push your knees outward — they're caving in.",
    "KNEES_OVER_TOES": "Sit back more — knees are going past your toes.",
    "SHALLOW_DEPTH":  "Go deeper — aim for thighs parallel to the ground.",
    "UNEVEN_WEIGHT":  "Balance your weight evenly on both legs.",
}

# ─── Video → Label Mapping (for training) ────────────────────────────────────
# Maps partial filename/path matches to class labels.
VIDEO_LABEL_MAP = {
    # Correct squats — all files under ✅ CORRECT SQUAT
    "CORRECT SQUAT": "CORRECT",
    # Wrong squats — GEMINI [BODYWEIGHT]
    "Excessive Forward Lean":          "FORWARD_LEAN",
    "Heels lifting off the floor":     "HEEL_RISE",
    "Knees Collapsing Inward":         "KNEE_CAVE",
    "Knees_extending_too_far_over_t":  "KNEES_OVER_TOES",
    "Shallow squat depth":             "SHALLOW_DEPTH",
    "Uneven weight distribution":      "UNEVEN_WEIGHT",
    # Wrong squats — YT Clip
    "Bending Forward From Your Back":  "FORWARD_LEAN",
    "Knees Collapsing Inwards":        "KNEE_CAVE",
    "Lifting From Your Butt First":    "FORWARD_LEAN",
}

# Files to skip entirely
SKIP_FILES = {"Squad.mov"}

# ─── MediaPipe Landmark IDs (33-point BlazePose) ─────────────────────────────
LM_NOSE              = 0
LM_LEFT_SHOULDER     = 11
LM_RIGHT_SHOULDER    = 12
LM_LEFT_ELBOW        = 13
LM_RIGHT_ELBOW       = 14
LM_LEFT_WRIST        = 15
LM_RIGHT_WRIST       = 16
LM_LEFT_HIP          = 23
LM_RIGHT_HIP         = 24
LM_LEFT_KNEE         = 25
LM_RIGHT_KNEE        = 26
LM_LEFT_ANKLE        = 27
LM_RIGHT_ANKLE       = 28
LM_LEFT_HEEL         = 29
LM_RIGHT_HEEL        = 30
LM_LEFT_FOOT_INDEX   = 31
LM_RIGHT_FOOT_INDEX  = 32

REQUIRED_LANDMARKS = [
    LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER,
    LM_LEFT_HIP, LM_RIGHT_HIP,
    LM_LEFT_KNEE, LM_RIGHT_KNEE,
    LM_LEFT_ANKLE, LM_RIGHT_ANKLE,
]

# Skeleton drawing connections (pairs of landmark IDs)
POSE_CONNECTIONS = [
    (LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER),
    (LM_LEFT_SHOULDER, LM_LEFT_ELBOW),
    (LM_LEFT_ELBOW, LM_LEFT_WRIST),
    (LM_RIGHT_SHOULDER, LM_RIGHT_ELBOW),
    (LM_RIGHT_ELBOW, LM_RIGHT_WRIST),
    (LM_LEFT_SHOULDER, LM_LEFT_HIP),
    (LM_RIGHT_SHOULDER, LM_RIGHT_HIP),
    (LM_LEFT_HIP, LM_RIGHT_HIP),
    (LM_LEFT_HIP, LM_LEFT_KNEE),
    (LM_LEFT_KNEE, LM_LEFT_ANKLE),
    (LM_RIGHT_HIP, LM_RIGHT_KNEE),
    (LM_RIGHT_KNEE, LM_RIGHT_ANKLE),
    (LM_LEFT_ANKLE, LM_LEFT_HEEL),
    (LM_LEFT_ANKLE, LM_LEFT_FOOT_INDEX),
    (LM_RIGHT_ANKLE, LM_RIGHT_HEEL),
    (LM_RIGHT_ANKLE, LM_RIGHT_FOOT_INDEX),
]

# ─── Feature Engineering ─────────────────────────────────────────────────────
FEATURE_COUNT = 15
WINDOW_SIZE = 15

FEATURE_NAMES = [
    "left_knee_angle",
    "right_knee_angle",
    "back_angle",
    "left_knee_cave_ratio",
    "right_knee_cave_ratio",
    "hip_depth_ratio",
    "left_heel_delta",
    "right_heel_delta",
    "shoulder_z_diff",
    "torso_lean_ratio",
    "hip_symmetry",
    "knee_angle_diff",
    "trunk_vertical_angle",
    "knee_forward_ratio",
    "hip_lateral_shift",
]

# ─── Hard-Coded Rule Thresholds ──────────────────────────────────────────────
THRESH_KNEE_CAVE       = 0.12   # Knee inward as fraction of body width
THRESH_FORWARD_LEAN    = 33.0   # Trunk angle from vertical (degrees)
THRESH_HEEL_RISE       = 0.02   # Heel Y change as fraction of body height
THRESH_KNEES_OVER_TOES = 0.15   # Knee Z past foot-index Z (very relaxed — Z unreliable)
THRESH_SHALLOW_DEPTH   = 0.09   # Hip-knee ratio for parallel
THRESH_UNEVEN_HIP      = 0.06   # Hip Y diff as fraction of body height
THRESH_UNEVEN_KNEE     = 25.0   # Knee angle difference in degrees

# ─── Rep Detection State Machine ─────────────────────────────────────────────
STANDING_HIP_KNEE_RATIO = 0.25
SQUAT_HIP_KNEE_RATIO    = 0.20

# ─── Model Training ──────────────────────────────────────────────────────────
TRAIN_EPOCHS        = 100
TRAIN_BATCH_SIZE    = 32
TRAIN_VAL_SPLIT     = 0.2
EARLY_STOP_PATIENCE = 15
SUBSAMPLE_RATE      = 2
LEARNING_RATE       = 0.001

# ─── Real-Time Analysis ──────────────────────────────────────────────────────
VISIBILITY_THRESHOLD    = 0.5
SMOOTHING_ALPHA         = 0.6
ML_CONFIDENCE_THRESHOLD = 0.70
