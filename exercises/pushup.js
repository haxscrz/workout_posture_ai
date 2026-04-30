/**
 * pushup.js — Push-Up Analyzer
 * Camera: Side profile (user perpendicular to camera)
 * Key checks: body alignment (plank), elbow depth, head neutral, hip position
 */

import { ExerciseBase } from './ExerciseBase.js';
import {
  LM, getLM, computeAngle, areLandmarksVisible,
  getBodyHeight, smoothAngle, midpoint
} from '../pose-engine.js';

const DEFAULT_CONFIG = {
  // State transitions
  TOP_ELBOW_ANGLE: 155,    // Arms extended (top of push-up)
  BOTTOM_ELBOW_ANGLE: 100, // Arms bent (bottom of push-up)

  // Body alignment (plank) — shoulder/hip/ankle should be ~180°
  PLANK_MIN_ANGLE: 158,    // Below this = hip sagging or piking too much
  PLANK_MAX_ANGLE: 195,    // Above this = piking up

  // Head/neck neutral — ear/shoulder/hip angle
  HEAD_NEUTRAL_MIN: 155,

  // Elbow flare — elbow should stay close to torso in side view
  ELBOW_FLARE_THRESHOLD: 0.15, // elbow x-deviation from shoulder, normalized

  VISIBILITY_THRESHOLD: 0.55,
};

// Side-view: prefer whichever side is facing the camera (higher visibility)
const LEFT_LANDMARKS  = [LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST, LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE];
const RIGHT_LANDMARKS = [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST, LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE];

export class PushUpAnalyzer extends ExerciseBase {
  constructor(config = {}) {
    super({ ...DEFAULT_CONFIG, ...config });
    this._smoothedElbow = null;
    this._smoothedPlank = null;
    this.state = 'WAITING'; // WAITING → UP → DOWN → UP ...
  }

  getRequirements() {
    return {
      orientation: 'side',
      instruction: 'Turn sideways to the camera. Place phone on the floor to your side. Your full body from head to feet must be visible.',
      cameraPlacement: 'On the floor to your left or right side, pointing at you.',
      requiredLandmarks: [...LEFT_LANDMARKS, ...RIGHT_LANDMARKS],
      minFrameCoverage: 0.5,
    };
  }

  analyze(landmarks, worldLandmarks) {
    // Pick best visible side
    const leftVis  = areLandmarksVisible(landmarks, LEFT_LANDMARKS,  this.config.VISIBILITY_THRESHOLD);
    const rightVis = areLandmarksVisible(landmarks, RIGHT_LANDMARKS, this.config.VISIBILITY_THRESHOLD);

    let side = null;
    if (leftVis && rightVis) {
      // Use left (arbitrary preference for side view)
      side = 'LEFT';
    } else if (leftVis) {
      side = 'LEFT';
    } else if (rightVis) {
      side = 'RIGHT';
    }

    if (!side) {
      return this._emptyResult('Turn sideways so the camera can see your full body profile.');
    }

    const [shoulderId, elbowId, wristId, hipId, kneeId, ankleId, earId] =
      side === 'LEFT'
        ? [LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST, LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE, LM.LEFT_EAR]
        : [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST, LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE, LM.RIGHT_EAR];

    const shoulder = getLM(landmarks, shoulderId);
    const elbow    = getLM(landmarks, elbowId);
    const wrist    = getLM(landmarks, wristId);
    const hip      = getLM(landmarks, hipId);
    const knee     = getLM(landmarks, kneeId);
    const ankle    = getLM(landmarks, ankleId);
    const ear      = getLM(landmarks, earId);

    // ── Angles ────────────────────────────────────────────────────────────
    const rawElbow = computeAngle(shoulder, elbow, wrist);
    const rawPlank = computeAngle(shoulder, hip, ankle);   // body alignment
    const headAngle = computeAngle(ear, shoulder, hip);

    this._smoothedElbow = smoothAngle(this._smoothedElbow, rawElbow, 0.35);
    this._smoothedPlank = smoothAngle(this._smoothedPlank, rawPlank, 0.25);

    const elbowAngle = this._smoothedElbow;
    const plankAngle = this._smoothedPlank;

    const angleData = {
      elbowAngle: Math.round(elbowAngle ?? 0),
      plankAngle: Math.round(plankAngle ?? 0),
      headAngle:  Math.round(headAngle  ?? 0),
    };

    const issues = [];

    // ── Form checks ────────────────────────────────────────────────────────

    // 1. Hip sagging (plank angle too small — hip drops below line)
    if (plankAngle !== null && plankAngle < this.config.PLANK_MIN_ANGLE) {
      issues.push({
        jointIds: [hipId],
        message: 'Engage your core — hips are dropping',
        severity: 'error',
        code: 'PUSHUP_HIP_SAG',
      });
    }

    // 2. Hips piking (angle too large — hip raised above line)
    if (plankAngle !== null && plankAngle > this.config.PLANK_MAX_ANGLE) {
      issues.push({
        jointIds: [hipId],
        message: 'Lower your hips — don\'t pike up',
        severity: 'error',
        code: 'PUSHUP_HIP_PIKE',
      });
    }

    // 3. Head/neck neutral
    if (headAngle !== null && headAngle < this.config.HEAD_NEUTRAL_MIN) {
      issues.push({
        jointIds: [earId, shoulderId],
        message: 'Keep your head neutral — look at the floor',
        severity: 'warning',
        code: 'PUSHUP_HEAD_DROOP',
      });
    }

    // 4. Elbow flare (visible in side view as elbow moving away from torso in X)
    if (shoulder && elbow) {
      const bodyH = getBodyHeight(landmarks);
      const flare = Math.abs(elbow.x - shoulder.x) / bodyH;
      if (flare > this.config.ELBOW_FLARE_THRESHOLD) {
        issues.push({
          jointIds: [elbowId, shoulderId],
          message: 'Tuck elbows closer to your body',
          severity: 'warning',
          code: 'PUSHUP_ELBOW_FLARE',
        });
      }
    }

    // ── State machine ──────────────────────────────────────────────────────
    let repCounted = false;

    if (this.state === 'WAITING') {
      // Wait for user to get into push-up up position
      if (elbowAngle !== null && elbowAngle >= this.config.TOP_ELBOW_ANGLE &&
          plankAngle !== null && plankAngle >= this.config.PLANK_MIN_ANGLE) {
        this.state = 'UP';
        this._startNewRep();
      }
    }

    if (this.state === 'UP') {
      if (elbowAngle !== null && elbowAngle <= this.config.BOTTOM_ELBOW_ANGLE) {
        this.state = 'DOWN';
      }
    }

    if (this.state === 'DOWN') {
      if (elbowAngle !== null && elbowAngle >= this.config.TOP_ELBOW_ANGLE) {
        this._recordIssues(issues);
        this.repQuality = this._scoreRep();
        this.repCount++;
        repCounted = true;
        this.state = 'UP';
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
    this._smoothedElbow = null;
    this._smoothedPlank = null;
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
