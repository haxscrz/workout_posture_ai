/**
 * pose-engine.js
 * Core pose analysis utilities. Zero DOM/browser dependencies.
 * ─────────────────────────────────────────────────────────────
 * UPGRADE-FRIENDLY: This module is framework-agnostic.
 *   - Runs identically in browser (MediaPipe) and React Native (TF.js BlazePose)
 *   - Landmark schema matches @tensorflow-models/pose-detection for easy migration
 *   - When adding an ML classifier, import these utils in your feature pipeline
 * ─────────────────────────────────────────────────────────────
 * PERFORMANCE: All functions are pure, allocation-light, and safe to call
 *   every animation frame on mobile hardware.
 */

// ─── Landmark IDs (MediaPipe Pose / BlazePose 33-point schema) ───────────────
export const LM = Object.freeze({
  NOSE: 0,
  LEFT_EYE_INNER: 1, LEFT_EYE: 2, LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4, RIGHT_EYE: 5, RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7, RIGHT_EAR: 8,
  MOUTH_LEFT: 9, MOUTH_RIGHT: 10,
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_PINKY: 17, RIGHT_PINKY: 18,
  LEFT_INDEX: 19, RIGHT_INDEX: 20,
  LEFT_THUMB: 21, RIGHT_THUMB: 22,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26,
  LEFT_ANKLE: 27, RIGHT_ANKLE: 28,
  LEFT_HEEL: 29, RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31, RIGHT_FOOT_INDEX: 32,
});

// ─── Core Math ───────────────────────────────────────────────────────────────

/**
 * Compute the angle (degrees) at joint `b` formed by the path a→b→c.
 * Uses 3D coords (z) when available — robust to camera tilt and distance.
 * KEY: Angles are camera-position invariant as long as all 3 joints are visible.
 * Returns null if any point is missing.
 */
export function computeAngle(a, b, c) {
  if (!a || !b || !c) return null;
  const v1x = a.x - b.x, v1y = a.y - b.y, v1z = (a.z || 0) - (b.z || 0);
  const v2x = c.x - b.x, v2y = c.y - b.y, v2z = (c.z || 0) - (b.z || 0);
  const dot = v1x * v2x + v1y * v2y + v1z * v2z;
  const mag1 = Math.sqrt(v1x ** 2 + v1y ** 2 + v1z ** 2);
  const mag2 = Math.sqrt(v2x ** 2 + v2y ** 2 + v2z ** 2);
  if (mag1 < 1e-6 || mag2 < 1e-6) return null;
  return Math.acos(Math.max(-1, Math.min(1, dot / (mag1 * mag2)))) * (180 / Math.PI);
}

/** Get a landmark by ID with null-safety. */
export function getLM(landmarks, id) {
  return landmarks?.[id] ?? null;
}

/** Midpoint between two landmarks. */
export function midpoint(a, b) {
  if (!a || !b) return null;
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: ((a.z || 0) + (b.z || 0)) / 2 };
}

// ─── Visibility Checks ───────────────────────────────────────────────────────

/**
 * Returns true if all specified landmark IDs are visible above threshold.
 * MediaPipe visibility is 0–1; we default to 0.6 for reliable detection.
 */
export function areLandmarksVisible(landmarks, ids, threshold = 0.6) {
  if (!landmarks) return false;
  return ids.every(id => {
    const lm = landmarks[id];
    return lm && (lm.visibility == null || lm.visibility >= threshold);
  });
}

/** Returns a 0–1 score of how many required landmarks are visible. */
export function visibilityScore(landmarks, ids, threshold = 0.5) {
  if (!landmarks || ids.length === 0) return 0;
  const visible = ids.filter(id => {
    const lm = landmarks[id];
    return lm && (lm.visibility == null || lm.visibility >= threshold);
  });
  return visible.length / ids.length;
}

// ─── Body Proportion Helpers (floor-camera adaptation) ────────────────────────

/**
 * Approximate body height in normalized image coordinates (0–1).
 * Used to convert proportion-based thresholds so they work regardless
 * of the user's distance from the camera or camera angle.
 */
export function getBodyHeight(landmarks) {
  const nose = getLM(landmarks, LM.NOSE);
  const lAnkle = getLM(landmarks, LM.LEFT_ANKLE);
  const rAnkle = getLM(landmarks, LM.RIGHT_ANKLE);
  if (!nose) return 0.7;
  const ankle = lAnkle || rAnkle;
  if (!ankle) return 0.7;
  return Math.abs(ankle.y - nose.y) || 0.7;
}

/** Shoulder-to-shoulder width in normalized coords. */
export function getBodyWidth(landmarks) {
  const ls = getLM(landmarks, LM.LEFT_SHOULDER);
  const rs = getLM(landmarks, LM.RIGHT_SHOULDER);
  if (!ls || !rs) return 0.25;
  return Math.abs(ls.x - rs.x) || 0.25;
}

/** Hip center point. */
export function getHipCenter(landmarks) {
  return midpoint(getLM(landmarks, LM.LEFT_HIP), getLM(landmarks, LM.RIGHT_HIP));
}

/** Shoulder center point. */
export function getShoulderCenter(landmarks) {
  return midpoint(getLM(landmarks, LM.LEFT_SHOULDER), getLM(landmarks, LM.RIGHT_SHOULDER));
}

// ─── Orientation Detection ────────────────────────────────────────────────────

/**
 * Detect which side the user is presenting to the camera.
 * Returns: 'front' | 'side' | 'unknown'
 * 
 * Logic: In a side profile, one shoulder's Z depth is significantly different
 * from the other, AND left/right landmarks have similar X positions.
 * In front view, both shoulders have similar Z values and are spread in X.
 */
export function detectOrientation(landmarks) {
  const ls = getLM(landmarks, LM.LEFT_SHOULDER);
  const rs = getLM(landmarks, LM.RIGHT_SHOULDER);
  if (!ls || !rs) return 'unknown';

  const xSpread = Math.abs(ls.x - rs.x);
  const bodyW = getBodyWidth(landmarks);

  // Side view: shoulders are close together in X relative to body scale
  if (xSpread < bodyW * 0.4) return 'side';
  // Front view: shoulders spread wide
  if (xSpread > bodyW * 0.6) return 'front';
  return 'unknown';
}

/**
 * Estimate how much of the frame the user's body occupies (0–1).
 * Used by setup wizard to ensure the user is at the right distance.
 */
export function getBodyFrameCoverage(landmarks) {
  const nose = getLM(landmarks, LM.NOSE);
  const lAnkle = getLM(landmarks, LM.LEFT_ANKLE);
  const rAnkle = getLM(landmarks, LM.RIGHT_ANKLE);
  if (!nose) return 0;
  const ankle = lAnkle || rAnkle;
  if (!ankle) return 0;
  return Math.abs(ankle.y - nose.y);
}

// ─── Smoothing ────────────────────────────────────────────────────────────────

/**
 * Exponential moving average for smoothing noisy angle values.
 * alpha: 0 = no update, 1 = no smoothing. 0.3–0.5 is good for mobile.
 */
export function smoothAngle(prev, current, alpha = 0.4) {
  if (prev == null) return current;
  if (current == null) return prev;
  return prev + alpha * (current - prev);
}

/**
 * 3D Landmark Smoothing Filter (Low-pass)
 * Applies exponential smoothing to the raw X, Y, Z coordinates to eliminate
 * micro-jitters in pose detection, significantly improving form calculation accuracy.
 */
export function smoothLandmarks(prev, current, alpha = 0.6) {
  if (!prev || !current || prev.length !== current.length) return current;
  
  const smoothed = [];
  for (let i = 0; i < current.length; i++) {
    const c = current[i];
    const p = prev[i];
    if (!c || !p) {
      smoothed.push(c);
      continue;
    }
    smoothed.push({
      x: p.x + alpha * (c.x - p.x),
      y: p.y + alpha * (c.y - p.y),
      z: p.z + alpha * (c.z - p.z),
      visibility: (p.visibility ?? 1) + alpha * ((c.visibility ?? 1) - (p.visibility ?? 1))
    });
  }
  return smoothed;
}
