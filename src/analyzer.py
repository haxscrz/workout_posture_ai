"""
analyzer.py — Real-time squat posture analysis.

Opens a camera feed (webcam or IP webcam URL), runs MediaPipe Pose detection,
draws skeleton overlay, counts reps, and detects bad posture using both
hard-coded rules and the trained ML model.

Controls:
  Q / ESC — Quit
  R       — Reset rep counter
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import mediapipe as mp

from src.config import (
    MODEL_PATH, LABELS_PATH,
    CLASSES, CLASS_MESSAGES, NUM_CLASSES,
    FEATURE_COUNT, WINDOW_SIZE,
    REQUIRED_LANDMARKS, POSE_CONNECTIONS,
    LM_LEFT_HEEL, LM_RIGHT_HEEL,
    LM_LEFT_HIP, LM_RIGHT_HIP,
    LM_LEFT_KNEE, LM_RIGHT_KNEE,
    LM_LEFT_FOOT_INDEX, LM_RIGHT_FOOT_INDEX,
    LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER,
    STANDING_HIP_KNEE_RATIO, SQUAT_HIP_KNEE_RATIO,
    SMOOTHING_ALPHA, ML_CONFIDENCE_THRESHOLD,
)
from src.utils import (
    landmarks_to_list, get_lm, are_landmarks_visible,
    midpoint, get_body_height, smooth_landmarks,
)
from src.features import extract_features
from src.rules import run_all_rules, check_shallow_depth

mp_pose = mp.solutions.pose


class SquatAnalyzer:
    """Real-time squat analysis engine."""

    def __init__(self):
        # State machine
        self.state = "WAITING"  # WAITING → STANDING → SQUATTING → STANDING (rep++)
        self.rep_count = 0
        self.good_reps = 0
        self.fair_reps = 0
        self.poor_reps = 0

        # Per-rep tracking
        self.rep_issues = []
        self.min_hip_depth = None

        # Baseline (set at first standing position)
        self.baseline = {}

        # Smoothing
        self.smoothed_landmarks = None

        # ML model
        self.ml_model = None
        self.ml_labels = None
        self.ml_ready = False
        self.frame_buffer = []
        self.ml_prediction = None  # Latest: {"label": str, "confidence": float}

        # Rolling coordinate histories for rate-of-change posture rules
        self.hip_y_history = []
        self.shoulder_y_history = []

        # Current camera view angle profile
        self.current_view = "front"

        # Current frame state
        self.current_issues = []
        self.last_rep_quality = None

    def load_ml_model(self):
        """Load the trained ML model if available."""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            print("  [ML] No trained model found. Using rules only.")
            print(f"       Run 'python src/main.py --train' to train a model.")
            return False

        try:
            import tensorflow as tf
            self.ml_model = tf.keras.models.load_model(MODEL_PATH)
            with open(LABELS_PATH, "r") as f:
                self.ml_labels = json.load(f)
            self.ml_ready = True
            print(f"  [ML] Model loaded: {len(self.ml_labels)} classes")
            return True
        except Exception as e:
            print(f"  [ML] Failed to load model: {e}")
            return False

    def process_frame(self, landmarks_raw):
        """
        Process a single frame of raw MediaPipe landmarks.
        Returns dict with analysis results.
        """
        if landmarks_raw is None:
            return {
                "state": self.state,
                "landmarks": None,
                "issues": [],
                "rep_counted": False,
            }

        # Convert to list of dicts
        lm_list = landmarks_to_list(landmarks_raw)

        # Smooth landmarks
        self.smoothed_landmarks = smooth_landmarks(
            self.smoothed_landmarks, lm_list, SMOOTHING_ALPHA
        )
        landmarks = self.smoothed_landmarks

        # Check visibility
        if not are_landmarks_visible(landmarks, REQUIRED_LANDMARKS):
            return {
                "state": self.state,
                "landmarks": landmarks,
                "issues": [{"code": "VISIBILITY", "message": "Full body must be visible (hips to ankles).", "severity": "info", "joints": []}],
                "rep_counted": False,
            }

        # Compute hip-knee depth ratio for state machine
        mid_hip = midpoint(get_lm(landmarks, LM_LEFT_HIP), get_lm(landmarks, LM_RIGHT_HIP))
        mid_knee = midpoint(get_lm(landmarks, LM_LEFT_KNEE), get_lm(landmarks, LM_RIGHT_KNEE))
        body_h = get_body_height(landmarks)

        if mid_hip is None or mid_knee is None:
            return {"state": self.state, "landmarks": landmarks, "issues": [], "rep_counted": False}

        hip_knee_dist = (mid_knee["y"] - mid_hip["y"]) / body_h
        is_squatting = hip_knee_dist < STANDING_HIP_KNEE_RATIO - 0.02

        # Get shoulder midpoint for rate-of-ascent (lifting butt first) check
        mid_shoulder = midpoint(get_lm(landmarks, LM_LEFT_SHOULDER), get_lm(landmarks, LM_RIGHT_SHOULDER))

        # Update rolling histories
        if mid_hip and mid_shoulder:
            self.hip_y_history.append(mid_hip["y"])
            self.shoulder_y_history.append(mid_shoulder["y"])
            if len(self.hip_y_history) > 10:
                self.hip_y_history.pop(0)
                self.shoulder_y_history.pop(0)

        # ── Run rules ──
        self.current_issues, self.current_view = run_all_rules(
            landmarks, self.baseline, is_squatting,
            self.hip_y_history, self.shoulder_y_history
        )

        # ── Run ML ──
        if self.ml_ready:
            feats = extract_features(landmarks, self.baseline)
            if feats is not None:
                self.frame_buffer.append(feats)
                if len(self.frame_buffer) > WINDOW_SIZE:
                    self.frame_buffer.pop(0)

                if len(self.frame_buffer) == WINDOW_SIZE:
                    self._run_ml_inference()

        # ── State machine ──
        rep_counted = False

        if self.state == "WAITING":
            if hip_knee_dist >= STANDING_HIP_KNEE_RATIO:
                self.state = "STANDING"
                self._set_baseline(landmarks)
                self._start_new_rep()

        elif self.state == "STANDING":
            if hip_knee_dist < SQUAT_HIP_KNEE_RATIO:
                self.state = "SQUATTING"
                self.min_hip_depth = hip_knee_dist

        elif self.state == "SQUATTING":
            # Track deepest point
            if self.min_hip_depth is None or hip_knee_dist < self.min_hip_depth:
                self.min_hip_depth = hip_knee_dist

            # Record issues during squat
            if self.current_issues:
                self.rep_issues.extend(self.current_issues)

            # Check if returned to standing
            if hip_knee_dist >= STANDING_HIP_KNEE_RATIO:
                # Check depth at end of rep
                depth_issues = check_shallow_depth(self.min_hip_depth)
                if depth_issues:
                    self.current_issues.extend(depth_issues)
                    self.rep_issues.extend(depth_issues)

                # Score rep
                self.last_rep_quality = self._score_rep()
                self.rep_count += 1
                if self.last_rep_quality == "good":
                    self.good_reps += 1
                elif self.last_rep_quality == "fair":
                    self.fair_reps += 1
                else:
                    self.poor_reps += 1

                rep_counted = True
                self.state = "STANDING"
                self._set_baseline(landmarks)
                self._start_new_rep()

        return {
            "state": self.state,
            "landmarks": landmarks,
            "issues": self.current_issues,
            "rep_counted": rep_counted,
            "rep_quality": self.last_rep_quality if rep_counted else None,
        }

    def _set_baseline(self, landmarks):
        """Set heel and foot baseline from standing position."""
        l_heel = get_lm(landmarks, LM_LEFT_HEEL)
        r_heel = get_lm(landmarks, LM_RIGHT_HEEL)
        l_foot = get_lm(landmarks, LM_LEFT_FOOT_INDEX)
        r_foot = get_lm(landmarks, LM_RIGHT_FOOT_INDEX)
        
        if l_heel:
            self.baseline["left_heel_y"] = l_heel["y"]
        if r_heel:
            self.baseline["right_heel_y"] = r_heel["y"]
        if l_foot:
            self.baseline["left_foot_y"] = l_foot["y"]
        if r_foot:
            self.baseline["right_foot_y"] = r_foot["y"]

    def _start_new_rep(self):
        """Reset per-rep tracking."""
        self.rep_issues = []
        self.min_hip_depth = None

    def _score_rep(self):
        """
        Score the current rep based on accumulated issues.
        Uses unique issue codes with a minimum frequency threshold
        to avoid penalizing single-frame noise.
        """
        MIN_FRAMES = 10  # Issue must appear on 10+ frames to count

        # Count frames per issue code
        code_counts = {}
        code_severity = {}
        for issue in self.rep_issues:
            code = issue.get("code", "?")
            code_counts[code] = code_counts.get(code, 0) + 1
            code_severity[code] = issue.get("severity", "warning")

        # Only count issues that persisted across multiple frames
        real_errors = 0
        real_warnings = 0
        for code, count in code_counts.items():
            if count < MIN_FRAMES:
                continue  # Transient noise, ignore
            if code_severity[code] == "error":
                real_errors += 1
            else:
                real_warnings += 1

        if real_errors == 0 and real_warnings == 0:
            return "good"
        if real_errors == 0 and real_warnings <= 1:
            return "fair"
        return "poor"

    def _run_ml_inference(self):
        """Run ML model on current frame buffer."""
        try:
            input_data = np.array([self.frame_buffer], dtype=np.float32)
            # Call model directly as callable for 10x faster execution (avoids Keras predict overhead)
            probs = self.ml_model(input_data, training=False).numpy()[0]

            max_idx = int(np.argmax(probs))
            label = self.ml_labels[max_idx]
            confidence = float(probs[max_idx])

            self.ml_prediction = {"label": label, "confidence": confidence}

            # If ML finds an issue with high confidence that rules didn't catch
            if label != "CORRECT" and confidence >= ML_CONFIDENCE_THRESHOLD:
                rule_codes = {i["code"] for i in self.current_issues}
                if label not in rule_codes:
                    self.current_issues.append({
                        "code": label,
                        "message": f"[ML] {CLASS_MESSAGES.get(label, label)} ({confidence*100:.0f}%)",
                        "severity": "warning",
                        "joints": [],
                    })
        except Exception:
            pass

    def reset(self):
        """Reset all counters."""
        self.state = "WAITING"
        self.rep_count = 0
        self.good_reps = 0
        self.fair_reps = 0
        self.poor_reps = 0
        self.rep_issues = []
        self.min_hip_depth = None
        self.baseline = {}
        self.smoothed_landmarks = None
        self.frame_buffer = []
        self.ml_prediction = None
        self.hip_y_history = []
        self.shoulder_y_history = []
        self.current_view = "front"
        self.current_issues = []
        self.last_rep_quality = None


# ─── Drawing Helpers ──────────────────────────────────────────────────────────

COLOR_GREEN  = (0, 220, 0)
COLOR_RED    = (0, 0, 255)
COLOR_YELLOW = (0, 220, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_GRAY   = (60, 60, 60)
COLOR_BG     = (30, 30, 30)


def draw_skeleton(frame, landmarks, issues):
    """Draw pose skeleton on frame with color-coded joints."""
    if landmarks is None:
        return

    h, w = frame.shape[:2]

    # Collect problematic joint IDs
    bad_joints = set()
    for issue in issues:
        bad_joints.update(issue.get("joints", []))

    # Draw connections
    for id1, id2 in POSE_CONNECTIONS:
        lm1 = landmarks[id1]
        lm2 = landmarks[id2]
        if lm1["visibility"] < 0.5 or lm2["visibility"] < 0.5:
            continue

        x1, y1 = int(lm1["x"] * w), int(lm1["y"] * h)
        x2, y2 = int(lm2["x"] * w), int(lm2["y"] * h)

        color = COLOR_RED if (id1 in bad_joints or id2 in bad_joints) else COLOR_GREEN
        cv2.line(frame, (x1, y1), (x2, y2), color, 3)

    # Draw joint circles
    for idx, lm in enumerate(landmarks):
        if lm["visibility"] < 0.5:
            continue
        x, y = int(lm["x"] * w), int(lm["y"] * h)
        color = COLOR_RED if idx in bad_joints else COLOR_GREEN
        cv2.circle(frame, (x, y), 6, color, -1)
        cv2.circle(frame, (x, y), 6, COLOR_WHITE, 1)


def draw_ui(frame, analyzer):
    """Draw the HUD overlay on the frame."""
    h, w = frame.shape[:2]

    # ── Top bar: Title + rep count ──
    cv2.rectangle(frame, (0, 0), (w, 50), COLOR_BG, -1)

    # Title
    cv2.putText(frame, "SQUAT POSTURE AI", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)

    # Rep count
    rep_text = f"Reps: {analyzer.rep_count}"
    cv2.putText(frame, rep_text, (w - 200, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)

    # ── Rep quality bar ──
    bar_y = 55
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + 30), COLOR_BG, -1)
    quality_text = f"Good: {analyzer.good_reps}  Fair: {analyzer.fair_reps}  Poor: {analyzer.poor_reps}"
    cv2.putText(frame, quality_text, (10, bar_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)

    # State indicator
    state_colors = {"WAITING": COLOR_YELLOW, "STANDING": COLOR_GREEN, "SQUATTING": (255, 165, 0)}
    state_color = state_colors.get(analyzer.state, COLOR_WHITE)
    cv2.putText(frame, analyzer.state, (w - 180, bar_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)

    # ── Form issues (bottom of screen) ──
    issues = analyzer.current_issues
    if issues:
        panel_h = 30 * len(issues) + 10
        panel_y = h - panel_h - 10
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, panel_y), (w - 5, h - 5), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        for i, issue in enumerate(issues):
            y_pos = panel_y + 25 + i * 30
            severity = issue.get("severity", "info")
            if severity == "error":
                icon, color = "X", COLOR_RED
            elif severity == "warning":
                icon, color = "!", COLOR_YELLOW
            else:
                icon, color = "i", COLOR_WHITE

            text = f"[{icon}] {issue['message']}"
            cv2.putText(frame, text, (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # ── Camera View Guide HUD Card (Upper Right) ──
    # Draw a nice semi-transparent card showing camera profile and active rules
    card_w, card_h = 320, 110
    card_x, card_y = w - card_w - 10, 95
    
    # Overlay background
    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), COLOR_BG, -1)
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), COLOR_GRAY, 1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Heading
    view_upper = analyzer.current_view.upper()
    if view_upper == "OBLIQUE":
        view_color = COLOR_GREEN
        view_status = "OBLIQUE (45) [BEST]"
    elif view_upper == "SIDE":
        view_color = COLOR_YELLOW
        view_status = "SIDE (90) [GOOD]"
    else:
        view_color = (0, 140, 255)  # Orange/Warning
        view_status = "FRONT (0) [LIMITS]"
        
    cv2.putText(frame, f"CAMERA: {view_status}", (card_x + 10, card_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, view_color, 2)
    cv2.line(frame, (card_x + 10, card_y + 28), (card_x + card_w - 10, card_y + 28), COLOR_GRAY, 1)
    
    # Active Rules list
    rules_y = card_y + 45
    def draw_rule_status(name, active, rx, ry):
        symbol = "[x]" if active else "[ ]"
        r_color = COLOR_GREEN if active else COLOR_GRAY
        cv2.putText(frame, f"{symbol} {name}", (rx, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, r_color, 1)
                    
    # Col 1
    draw_rule_status("Knee Cave", analyzer.current_view in ("front", "oblique"), card_x + 15, rules_y)
    draw_rule_status("Heel Rise", analyzer.current_view in ("side", "oblique"), card_x + 15, rules_y + 20)
    draw_rule_status("Butt First", analyzer.current_view in ("side", "oblique"), card_x + 15, rules_y + 40)
    
    # Col 2
    draw_rule_status("Fwd Lean", analyzer.current_view in ("side", "oblique"), card_x + 160, rules_y)
    draw_rule_status("Knees>Toes", analyzer.current_view == "side", card_x + 160, rules_y + 20)
    draw_rule_status("Weight Sym", analyzer.current_view in ("front", "oblique"), card_x + 160, rules_y + 40)

    # ── Rep quality flash ──
    if analyzer.last_rep_quality and analyzer.rep_count > 0:
        quality_colors = {"good": COLOR_GREEN, "fair": COLOR_YELLOW, "poor": COLOR_RED}
        q_color = quality_colors.get(analyzer.last_rep_quality, COLOR_WHITE)
        q_text = f"Rep {analyzer.rep_count}: {analyzer.last_rep_quality.upper()}"
        text_size = cv2.getTextSize(q_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        tx = (w - text_size[0]) // 2
        cv2.putText(frame, q_text, (tx, h // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, q_color, 3)

    # ── ML prediction badge ──
    if analyzer.ml_ready and analyzer.ml_prediction:
        pred = analyzer.ml_prediction
        ml_text = f"ML: {pred['label']} ({pred['confidence']*100:.0f}%)"
        ml_color = COLOR_GREEN if pred["label"] == "CORRECT" else COLOR_YELLOW
        cv2.putText(frame, ml_text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ml_color, 1)

    # ── Controls hint ──
    cv2.putText(frame, "[Q] Quit  [R] Reset  [C] Camera  [O] Rotate", (w - 340, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)


def run_camera(source=0, rotate=0):
    """
    Main camera analysis loop.

    Args:
        source: Camera index (int) or IP webcam URL (str).
        rotate: Initial rotation in degrees (0, 90, 180, 270).
    """
    print("=" * 60)
    print("  Squat Posture AI — Real-Time Analysis")
    print("=" * 60)

    # Init analyzer
    analyzer = SquatAnalyzer()
    analyzer.load_ml_model()

    # Open camera
    print(f"\n  Opening camera: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera source: {source}")
        print("  For webcam: use 0, 1, etc.")
        print("  For phone: use IP webcam URL like http://192.168.1.X:8080/video")
        sys.exit(1)

    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_source = source
    current_rotation = rotate
    # Query screen dimensions using tkinter to make portrait video as large as possible
    screen_width = 1920
    screen_height = 1080  # Default fallbacks
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        print(f"  [UI] Detected screen resolution: {screen_width}x{screen_height}")
    except Exception:
        pass

    # Create a normal resizable window so we can control scale and prevent clipping
    cv2.namedWindow("Squat Posture AI", cv2.WINDOW_NORMAL)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # Balance speed/accuracy for real-time
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("  Camera feed lost. Retrying...")
                time.sleep(0.5)
                continue

            # Apply rotation if needed (before mirroring and MediaPipe)
            if current_rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif current_rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif current_rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Mirror for selfie view
            frame = cv2.flip(frame, 1)

            # Run MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)

            # Analyze
            result = analyzer.process_frame(
                results.pose_landmarks if results.pose_landmarks else None
            )

            # Draw
            draw_skeleton(frame, result.get("landmarks"), result.get("issues", []))
            draw_ui(frame, analyzer)

            # FPS counter
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 1)

            # Resize the OpenCV window dynamically if the frame is in portrait layout
            # (height > width) so it is as large as possible to fit on the screen.
            fh, fw = frame.shape[:2]
            if fh > fw:
                # Tall portrait mode: scale to fit the screen height and width as much as possible
                max_h = screen_height - 120  # leave 120px for taskbar + window title bar
                max_w = screen_width - 80    # leave 80px for margins
                scale = min(max_w / fw, max_h / fh)
                display_w = int(fw * scale)
                display_h = int(fh * scale)
                cv2.resizeWindow("Squat Posture AI", display_w, display_h)
            else:
                # Normal landscape mode
                cv2.resizeWindow("Squat Posture AI", fw, fh)

            # Show
            cv2.imshow("Squat Posture AI", frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                break
            elif key in (ord("r"), ord("R")):
                analyzer.reset()
                print("  [Reset] Counters cleared.")
            elif key in (ord("c"), ord("C")):
                # Cycle camera source (0 → 1 → 2 → 3 → 0)
                if isinstance(current_source, int):
                    current_source = (current_source + 1) % 4
                    print(f"  [Camera] Switching to camera {current_source}...")
                    cap.release()
                    cap = cv2.VideoCapture(current_source)
                    if not cap.isOpened():
                        print(f"  [Camera] Camera {current_source} not available, trying next...")
                        current_source = 0
                        cap = cv2.VideoCapture(0)
                    else:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    analyzer.reset()
            elif key in (ord("o"), ord("O")):
                # Cycle rotation (0 -> 90 -> 180 -> 270 -> 0)
                current_rotation = (current_rotation + 90) % 360
                print(f"  [Rotation] Set to {current_rotation} degrees.")

    # Print session summary
    print("\n" + "=" * 60)
    print("  Session Summary")
    print("=" * 60)
    print(f"  Total Reps: {analyzer.rep_count}")
    print(f"  Good: {analyzer.good_reps}  Fair: {analyzer.fair_reps}  Poor: {analyzer.poor_reps}")
    print("=" * 60)


def run_video(video_path):
    """
    Analyze a video file (for testing).
    Same as run_camera but reads from a file.
    """
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    print("=" * 60)
    print("  Squat Posture AI — Video Analysis")
    print(f"  File: {video_path}")
    print("=" * 60)

    analyzer = SquatAnalyzer()
    analyzer.load_ml_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps_video)

    print(f"  Frames: {total_frames}, FPS: {fps_video:.1f}")
    print("  Press Q or ESC to quit, SPACE to pause.\n")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)

            result = analyzer.process_frame(
                results.pose_landmarks if results.pose_landmarks else None
            )

            draw_skeleton(frame, result.get("landmarks"), result.get("issues", []))
            draw_ui(frame, analyzer)

            cv2.imshow("Squat Posture AI — Video", frame)

            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == 32:  # Space = pause
                cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("  Video Analysis Summary")
    print("=" * 60)
    print(f"  Total Reps: {analyzer.rep_count}")
    print(f"  Good: {analyzer.good_reps}  Fair: {analyzer.fair_reps}  Poor: {analyzer.poor_reps}")
    print("=" * 60)
