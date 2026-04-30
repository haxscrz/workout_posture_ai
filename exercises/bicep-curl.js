/**
 * bicep-curl.js — Bicep Curl Analyzer
 * Camera: Front-facing (user faces the camera)
 * Detects: both arms simultaneously; uses the arm with higher visibility.
 */

import { ExerciseBase } from './ExerciseBase.js';
import {
  LM, getLM, computeAngle, areLandmarksVisible,
  getBodyHeight, smoothAngle
} from '../pose-engine.js';

// ─── Configuration (tune thresholds here without touching logic) ──────────────
const DEFAULT_CONFIG = {
  // State transitions
  BOTTOM_ANGLE: 155,     // Arm fully extended (degrees)
  TOP_ANGLE: 55,         // Arm fully curled (degrees)

  // Form checks
  ELBOW_DRIFT_Z: 0.07,   // Max allowed Z-drift of elbow from shoulder (world coords)
  TORSO_SWAY_DEG: 12,    // Max shoulder tilt before calling "stop swinging"
  WRIST_BREAK_DEG: 25,   // Max lateral wrist deviation
  MIN_CURL_ANGLE: 65,    // If top angle > this, "didn't curl enough"
  MIN_EXTEND_ANGLE: 145, // If bottom angle < this, "didn't extend enough"

  // Visibility
  VISIBILITY_THRESHOLD: 0.6,
};

const REQUIRED_LANDMARKS = {
  LEFT: [LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST, LM.LEFT_HIP],
  RIGHT: [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST, LM.RIGHT_HIP],
};

export class BicepCurlAnalyzer extends ExerciseBase {
  constructor(config = {}) {
    super({ ...DEFAULT_CONFIG, ...config });
    this._smoothedAngle = null;
    this._peakAngle = null;    // lowest angle reached this rep (most curled)
    this._valleyAngle = null;  // highest angle reached (most extended)
    this._lastShoulderTilt = null;
    this.state = 'WAITING'; // WAITING → BOTTOM → TOP → BOTTOM ...
  }

  getRequirements() {
    return {
      orientation: 'front',
      instruction: 'Face the camera directly. Place phone on the floor or prop it up slightly so your full body is visible.',
      cameraPlacement: 'On the floor in front of you, or propped up at shin height.',
      requiredLandmarks: [...REQUIRED_LANDMARKS.LEFT, ...REQUIRED_LANDMARKS.RIGHT],
      minFrameCoverage: 0.45,
    };
  }

  analyze(landmarks, worldLandmarks) {
    const issues = [];

    // ── Pick the best visible arm ──────────────────────────────────────────
    const leftVisible  = areLandmarksVisible(landmarks, REQUIRED_LANDMARKS.LEFT,  this.config.VISIBILITY_THRESHOLD);
    const rightVisible = areLandmarksVisible(landmarks, REQUIRED_LANDMARKS.RIGHT, this.config.VISIBILITY_THRESHOLD);

    let side = null;
    if (leftVisible && rightVisible) {
      // Prefer the arm closer to the camera (lower Z in world coords)
      const lElbow = getLM(worldLandmarks, LM.LEFT_ELBOW);
      const rElbow = getLM(worldLandmarks, LM.RIGHT_ELBOW);
      side = (!lElbow || !rElbow || lElbow.z <= rElbow.z) ? 'LEFT' : 'RIGHT';
    } else if (leftVisible) {
      side = 'LEFT';
    } else if (rightVisible) {
      side = 'RIGHT';
    }

    if (!side) {
      return this._emptyResult('Position your arm so the camera can see your shoulder, elbow, and wrist.');
    }

    // ── Extract joints ─────────────────────────────────────────────────────
    const [shoulderId, elbowId, wristId, hipId] =
      side === 'LEFT'
        ? [LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST, LM.LEFT_HIP]
        : [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST, LM.RIGHT_HIP];

    const shoulder = getLM(landmarks, shoulderId);
    const elbow    = getLM(landmarks, elbowId);
    const wrist    = getLM(landmarks, wristId);
    const hip      = getLM(landmarks, hipId);

    const wShoulder = getLM(worldLandmarks, shoulderId);
    const wElbow    = getLM(worldLandmarks, elbowId);

    // ── Compute angles ─────────────────────────────────────────────────────
    const rawAngle = computeAngle(shoulder, elbow, wrist);
    this._smoothedAngle = smoothAngle(this._smoothedAngle, rawAngle, 0.35);
    const angle = this._smoothedAngle;

    const angleData = { elbowAngle: Math.round(angle) };

    // ── Form checks (run every frame) ──────────────────────────────────────

    // 1. Elbow drifting forward (Z-depth in world coords)
    if (wShoulder && wElbow && Math.abs(wElbow.z - wShoulder.z) > this.config.ELBOW_DRIFT_Z) {
      issues.push({
        jointIds: [elbowId],
        message: 'Keep your elbow pinned to your side',
        severity: 'error',
        code: 'BICEP_ELBOW_DRIFT',
      });
    }

    // 2. Torso sway — compare shoulder tilt to hip tilt
    if (shoulder && hip) {
      const tilt = Math.abs(shoulder.y - hip.y);
      if (this._lastShoulderTilt !== null) {
        const sway = Math.abs(tilt - this._lastShoulderTilt) * 300; // scale to degrees approx
        if (sway > this.config.TORSO_SWAY_DEG) {
          issues.push({
            jointIds: [shoulderId, hipId],
            message: 'Stop swinging — control the lift',
            severity: 'error',
            code: 'BICEP_TORSO_SWAY',
          });
        }
      }
      this._lastShoulderTilt = tilt;
    }

    // ── State machine ──────────────────────────────────────────────────────
    let repCounted = false;

    if (this.state === 'WAITING' || this.state === 'BOTTOM') {
      if (angle !== null && angle >= this.config.BOTTOM_ANGLE) {
        if (this.state === 'WAITING') {
          this.state = 'BOTTOM';
        }
        this._valleyAngle = angle;
        this._peakAngle = null;
        this._startNewRep();
      }
    }

    if (this.state === 'BOTTOM') {
      if (angle !== null) {
        if (this._peakAngle === null || angle < this._peakAngle) {
          this._peakAngle = angle;
        }
        if (angle <= this.config.TOP_ANGLE) {
          this.state = 'TOP';
        }
      }
    }

    if (this.state === 'TOP') {
      if (angle !== null && angle >= this.config.BOTTOM_ANGLE) {
        // Rep complete — check quality
        if (this._peakAngle !== null && this._peakAngle > this.config.MIN_CURL_ANGLE) {
          issues.push({
            jointIds: [elbowId, wristId],
            message: "Curl higher — aim for full range of motion",
            severity: 'warning',
            code: 'BICEP_INCOMPLETE_CURL',
          });
        }
        if (this._valleyAngle !== null && this._valleyAngle < this.config.MIN_EXTEND_ANGLE) {
          issues.push({
            jointIds: [elbowId],
            message: "Fully extend your arm at the bottom",
            severity: 'warning',
            code: 'BICEP_INCOMPLETE_EXTENSION',
          });
        }
        this._recordIssues(issues);
        this.repQuality = this._scoreRep();
        this.repCount++;
        repCounted = true;
        this.state = 'BOTTOM';
        this._valleyAngle = angle;
        this._peakAngle = null;
        this._startNewRep();
      }
    }

    return {
      state: this.state,
      repCount: this.repCount,
      formIssues: issues,
      repQuality: repCounted ? this.repQuality : null,
      angleData,
      activeSide: side,
    };
  }

  reset() {
    super.reset();
    this._smoothedAngle = null;
    this._peakAngle = null;
    this._valleyAngle = null;
    this._lastShoulderTilt = null;
    this.state = 'WAITING';
  }

  _emptyResult(hint = '') {
    return {
      state: this.state,
      repCount: this.repCount,
      formIssues: hint ? [{ jointIds: [], message: hint, severity: 'warning', code: 'VISIBILITY' }] : [],
      repQuality: null,
      angleData: {},
      activeSide: null,
    };
  }
}
