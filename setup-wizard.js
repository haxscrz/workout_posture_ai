/**
 * setup-wizard.js
 * Pre-workout camera setup validation.
 * Ensures camera angle, user position, and landmark visibility are all correct
 * before allowing the workout to begin.
 */

import {
  areLandmarksVisible, visibilityScore,
  getBodyFrameCoverage, detectOrientation
} from './pose-engine.js';

export class SetupWizard {
  /**
   * @param {Object} requirements - From exerciseAnalyzer.getRequirements()
   */
  constructor(requirements) {
    this.requirements = requirements;
    this._readyFrames = 0;
    this._READY_FRAMES_NEEDED = 18; // ~0.6s at 30fps — must be consistently ready
  }

  /**
   * Evaluate current pose against setup requirements.
   * @param {Array} landmarks
   * @returns {{ passed: boolean, checks: Check[], readyProgress: number }}
   *
   * @typedef {Object} Check
   * @property {string} id
   * @property {string} label
   * @property {boolean} passed
   * @property {string} hint
   */
  evaluate(landmarks) {
    const checks = [];

    // ── Check 1: Key landmarks visible ──────────────────────────────────
    const visScore = visibilityScore(
      landmarks,
      this.requirements.requiredLandmarks,
      0.55
    );
    const landmarksOk = visScore >= 0.8;
    checks.push({
      id: 'landmarks',
      label: 'Body fully visible',
      passed: landmarksOk,
      hint: landmarksOk
        ? 'All key joints detected'
        : `Step back or adjust camera — ${Math.round(visScore * 100)}% of body visible`,
    });

    // ── Check 2: Orientation ─────────────────────────────────────────────
    const orientation = detectOrientation(landmarks);
    const requiredOrientation = this.requirements.orientation; // 'front' | 'side'
    const orientationOk =
      requiredOrientation === 'front'
        ? orientation === 'front'
        : orientation === 'side';
    checks.push({
      id: 'orientation',
      label: requiredOrientation === 'front' ? 'Facing camera' : 'Sideways to camera',
      passed: orientationOk,
      hint: orientationOk
        ? 'Orientation correct'
        : requiredOrientation === 'front'
          ? 'Turn to face the camera directly'
          : 'Turn sideways so the camera sees your profile',
    });

    // ── Check 3: Distance (body coverage) ───────────────────────────────
    const coverage = getBodyFrameCoverage(landmarks);
    const minCoverage = this.requirements.minFrameCoverage ?? 0.45;
    const maxCoverage = 0.92;
    const coverageOk = coverage >= minCoverage && coverage <= maxCoverage;
    checks.push({
      id: 'distance',
      label: 'Good distance from camera',
      passed: coverageOk,
      hint: coverageOk
        ? 'Distance looks good'
        : coverage < minCoverage
          ? 'Move closer to the camera'
          : 'Step back — you\'re too close',
    });

    // ── All passed? ──────────────────────────────────────────────────────
    const allPassed = checks.every(c => c.passed);

    if (allPassed) {
      this._readyFrames++;
    } else {
      this._readyFrames = Math.max(0, this._readyFrames - 2); // decay
    }

    const readyProgress = Math.min(1, this._readyFrames / this._READY_FRAMES_NEEDED);
    const isReady = this._readyFrames >= this._READY_FRAMES_NEEDED;

    return { passed: isReady, checks, readyProgress };
  }

  reset() {
    this._readyFrames = 0;
  }
}
