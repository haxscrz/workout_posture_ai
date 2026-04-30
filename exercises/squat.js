/**
 * squat.js — Squat Analyzer
 * Camera: Front or 45° angle from the floor
 * Key checks: knee cave, back rounding, squat depth, heel rise
 */

import { ExerciseBase } from './ExerciseBase.js';
import {
  LM, getLM, computeAngle, areLandmarksVisible,
  getBodyHeight, getBodyWidth, smoothAngle, midpoint
} from '../pose-engine.js';

const DEFAULT_CONFIG = {
  // State transitions (Hip vs Knee Y distance as fraction of body height)
  STANDING_HIP_KNEE_RATIO: 0.23, // Hips must rise to at least 23% of body height above knees
  SQUAT_HIP_KNEE_RATIO: 0.18,    // Hips must drop to 18% or below to count as a squat

  // Form checks
  KNEE_CAVE_THRESHOLD: 0.10,  // Increased to 0.10 to allow wider stances without false positives
  BACK_ROUND_MIN: 80,         // Lowered significantly (from 120) for front-facing low camera angles
  HEEL_RISE_THRESHOLD: 0.015, // Heel Y change > this fraction of body height = rising

  VISIBILITY_THRESHOLD: 0.55,
};

const REQUIRED_LANDMARKS = [
  LM.LEFT_HIP, LM.RIGHT_HIP,
  LM.LEFT_KNEE, LM.RIGHT_KNEE,
  LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
  LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
];

export class SquatAnalyzer extends ExerciseBase {
  constructor(config = {}) {
    super({ ...DEFAULT_CONFIG, ...config });
    this._smoothedLeftKnee  = null;
    this._smoothedRightKnee = null;
    this._smoothedBack      = null;
    this._baseLeftHeelY  = null;
    this._baseRightHeelY = null;
    this._minHipDepth = null; // track deepest point of current squat
    this.state = 'WAITING';
  }

  getRequirements() {
    return {
      orientation: 'front',
      instruction: 'Face the camera. Place phone on the floor in front of you, 1–1.5m away. Your full body from head to feet must be visible.',
      cameraPlacement: 'On the floor directly in front of you, angled up slightly.',
      requiredLandmarks: REQUIRED_LANDMARKS,
      minFrameCoverage: 0.50,
    };
  }

  analyze(landmarks, worldLandmarks) {
    if (!areLandmarksVisible(landmarks, REQUIRED_LANDMARKS, this.config.VISIBILITY_THRESHOLD)) {
      return this._emptyResult('Make sure your full body is visible — hips to ankles are required.');
    }

    // ── Extract joints ─────────────────────────────────────────────────────
    const lShoulder = getLM(landmarks, LM.LEFT_SHOULDER);
    const rShoulder = getLM(landmarks, LM.RIGHT_SHOULDER);
    const lHip   = getLM(landmarks, LM.LEFT_HIP);
    const rHip   = getLM(landmarks, LM.RIGHT_HIP);
    const lKnee  = getLM(landmarks, LM.LEFT_KNEE);
    const rKnee  = getLM(landmarks, LM.RIGHT_KNEE);
    const lAnkle = getLM(landmarks, LM.LEFT_ANKLE);
    const rAnkle = getLM(landmarks, LM.RIGHT_ANKLE);
    const lHeel  = getLM(landmarks, LM.LEFT_HEEL);
    const rHeel  = getLM(landmarks, LM.RIGHT_HEEL);

    // Mid-points for symmetrical checks
    const midShoulder = midpoint(lShoulder, rShoulder);
    const midHip      = midpoint(lHip, rHip);
    const midKnee     = midpoint(lKnee, rKnee);

    // ── Angles ────────────────────────────────────────────────────────────
    const rawLKnee = computeAngle(lHip, lKnee, lAnkle);
    const rawRKnee = computeAngle(rHip, rKnee, rAnkle);
    const rawBack  = computeAngle(midShoulder, midHip, midKnee); // torso upright check

    this._smoothedLeftKnee  = smoothAngle(this._smoothedLeftKnee,  rawLKnee, 0.35);
    this._smoothedRightKnee = smoothAngle(this._smoothedRightKnee, rawRKnee, 0.35);
    this._smoothedBack      = smoothAngle(this._smoothedBack, rawBack, 0.25);

    const avgKneeAngle = ((this._smoothedLeftKnee ?? 180) + (this._smoothedRightKnee ?? 180)) / 2;
    const backAngle    = this._smoothedBack;

    const angleData = {
      leftKneeAngle:  Math.round(this._smoothedLeftKnee  ?? 0),
      rightKneeAngle: Math.round(this._smoothedRightKnee ?? 0),
      backAngle:      Math.round(backAngle ?? 0),
    };

    const issues = [];
    const bodyW = getBodyWidth(landmarks);
    const bodyH = getBodyHeight(landmarks);

    // ── State machine (Hip vs Knee Vertical Distance) ───────────────────────
    let repCounted = false;
    // Calculate vertical distance from knee to hip (positive means hip is above knee)
    // Remember in image coords, Y goes down, so knee.y is > hip.y when standing
    const hipKneeDist = (midKnee.y - midHip.y) / bodyH;

    // ── Form checks ────────────────────────────────────────────────────────
    // Only check form when actively squatting (hips have dropped below standing threshold)
    const isSquatting = hipKneeDist < this.config.STANDING_HIP_KNEE_RATIO - 0.02;

    if (isSquatting) {
      // 1. Knee cave (knees collapsing inward)
      // Mirror-invariant check: distance from knee to body center vs ankle to body center
      if (lKnee && lAnkle && midHip) {
        const lKneeDist = Math.abs(lKnee.x - midHip.x);
        const lAnkleDist = Math.abs(lAnkle.x - midHip.x);
        if (lKneeDist < lAnkleDist - bodyW * this.config.KNEE_CAVE_THRESHOLD) {
          issues.push({
            jointIds: [LM.LEFT_KNEE, LM.LEFT_ANKLE],
            message: 'Push your left knee outward — it\'s caving in',
            severity: 'error',
            code: 'SQUAT_KNEE_CAVE_LEFT',
          });
        }
      }
      
      if (rKnee && rAnkle && midHip) {
        const rKneeDist = Math.abs(rKnee.x - midHip.x);
        const rAnkleDist = Math.abs(rAnkle.x - midHip.x);
        if (rKneeDist < rAnkleDist - bodyW * this.config.KNEE_CAVE_THRESHOLD) {
          issues.push({
            jointIds: [LM.RIGHT_KNEE, LM.RIGHT_ANKLE],
            message: 'Push your right knee outward — it\'s caving in',
            severity: 'error',
            code: 'SQUAT_KNEE_CAVE_RIGHT',
          });
        }
      }

      // 2. Back rounding (torso angle too small)
      if (backAngle !== null && backAngle < this.config.BACK_ROUND_MIN) {
        issues.push({
          jointIds: [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP],
          message: 'Chest up — your back is rounding',
          severity: 'error',
          code: 'SQUAT_BACK_ROUND',
        });
      }

      // 3. Heel rise (Y decreases when heel lifts in MediaPipe coords)
      let heelRising = false;
      if (this._baseLeftHeelY !== null && lHeel) {
        const leftDelta = lHeel.y - this._baseLeftHeelY;
        if (leftDelta < -bodyH * this.config.HEEL_RISE_THRESHOLD) heelRising = true;
      }
      if (this._baseRightHeelY !== null && rHeel) {
        const rightDelta = rHeel.y - this._baseRightHeelY;
        if (rightDelta < -bodyH * this.config.HEEL_RISE_THRESHOLD) heelRising = true;
      }
      if (heelRising) {
        issues.push({
          jointIds: [LM.LEFT_HEEL, LM.LEFT_ANKLE, LM.RIGHT_HEEL, LM.RIGHT_ANKLE],
          message: 'Keep your heels flat on the ground',
          severity: 'warning',
          code: 'SQUAT_HEEL_RISE',
        });
      }
    }

    if (this.state === 'WAITING') {
      if (hipKneeDist >= this.config.STANDING_HIP_KNEE_RATIO) {
        this.state = 'STANDING';
        // Capture heel baseline at standing position
        if (lHeel) this._baseLeftHeelY  = lHeel.y;
        if (rHeel) this._baseRightHeelY = rHeel.y;
        this._startNewRep();
        this._minHipDepth = null;
      }
    }

    if (this.state === 'STANDING') {
      if (hipKneeDist < this.config.SQUAT_HIP_KNEE_RATIO) {
        this.state = 'SQUAT';
      }
      this._minHipDepth = hipKneeDist;
    }

    if (this.state === 'SQUAT') {
      if (this._minHipDepth === null || hipKneeDist < this._minHipDepth) {
        this._minHipDepth = hipKneeDist;
      }

      if (hipKneeDist >= this.config.STANDING_HIP_KNEE_RATIO) {
        // Rep complete
        // Check if they went deep enough
        if (this._minHipDepth !== null && this._minHipDepth > this.config.SQUAT_HIP_KNEE_RATIO - 0.02) {
          issues.push({
            jointIds: [LM.LEFT_HIP, LM.RIGHT_HIP],
            message: 'Go deeper — aim for hips parallel to knees',
            severity: 'warning',
            code: 'SQUAT_INSUFFICIENT_DEPTH',
          });
        }
        this._recordIssues(issues);
        this.repQuality = this._scoreRep();
        this.repCount++;
        repCounted = true;
        this.state = 'STANDING';
        // Reset heel baseline for next rep
        if (lHeel) this._baseLeftHeelY  = lHeel.y;
        if (rHeel) this._baseRightHeelY = rHeel.y;
        this._startNewRep();
        this._minHipDepth = null;
      }
    }

    return {
      state: this.state,
      repCount: this.repCount,
      formIssues: issues,
      repQuality: repCounted ? this.repQuality : null,
      angleData,
    };
  }

  reset() {
    super.reset();
    this._smoothedLeftKnee  = null;
    this._smoothedRightKnee = null;
    this._smoothedBack      = null;
    this._baseLeftHeelY  = null;
    this._baseRightHeelY = null;
    this._minKneeAngle   = null;
    this.state = 'WAITING';
  }

  _emptyResult(hint = '') {
    return {
      state: this.state,
      repCount: this.repCount,
      formIssues: hint ? [{ jointIds: [], message: hint, severity: 'warning', code: 'VISIBILITY' }] : [],
      repQuality: null,
      angleData: {},
    };
  }
}
