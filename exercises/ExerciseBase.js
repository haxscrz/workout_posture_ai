/**
 * ExerciseBase.js
 * Abstract base class for all exercise analyzers.
 * ─────────────────────────────────────────────────────────────
 * UPGRADE PATH: To replace rule-based logic with an ML classifier:
 *   1. Create class MyExerciseML extends ExerciseBase
 *   2. Override analyze() to run your TF.js model on landmarks
 *   3. Return the same AnalysisResult shape
 *   4. The UI layer (skeleton.js, feedback.js) needs ZERO changes.
 * ─────────────────────────────────────────────────────────────
 *
 * @typedef {Object} FormIssue
 * @property {number[]} jointIds   - Landmark IDs to highlight red
 * @property {string}   message    - User-facing correction message
 * @property {'error'|'warning'} severity
 * @property {string}   code       - Machine-readable code (for ML training labels)
 *
 * @typedef {Object} AnalysisResult
 * @property {string}         state       - Current state machine state
 * @property {number}         repCount    - Total completed reps
 * @property {FormIssue[]}    formIssues  - Active form violations this frame
 * @property {'good'|'fair'|'poor'|null} repQuality - Quality of last rep
 * @property {Object}         angleData   - Raw angles for display/debug
 * @property {string[]}       setupChecks - Per-requirement pass/fail for wizard
 */

export class ExerciseBase {
  constructor(config = {}) {
    this.config = config;
    this.repCount = 0;
    this.state = 'IDLE';
    this.repQuality = null;
    this._issueHistory = []; // track issues per rep for quality scoring
    this._lastResult = {
      state: 'IDLE',
      repCount: 0,
      formIssues: [],
      repQuality: null,
      angleData: {}
    };
  }

  /**
   * Analyze a single frame of pose landmarks.
   * @param {Array} landmarks      - Normalized landmarks [{x,y,z,visibility}×33]
   * @param {Array} worldLandmarks - World-space landmarks in meters
   * @returns {AnalysisResult}
   */
  analyze(landmarks, worldLandmarks) {
    throw new Error(`${this.constructor.name}.analyze() not implemented`);
  }

  /** Reset state machine and rep counter for a new session. */
  reset() {
    this.repCount = 0;
    this.state = 'IDLE';
    this.repQuality = null;
    this._issueHistory = [];
  }

  /**
   * Return the setup requirements for the camera wizard.
   * @returns {{ orientation: string, instruction: string, requiredLandmarks: number[], cameraPlacement: string }}
   */
  getRequirements() {
    throw new Error(`${this.constructor.name}.getRequirements() not implemented`);
  }

  // ─── Protected helpers ──────────────────────────────────────────────────────

  /** Score the current rep based on issue history and emit repQuality. */
  _scoreRep() {
    const errors = this._issueHistory.filter(i => i.severity === 'error').length;
    const warnings = this._issueHistory.filter(i => i.severity === 'warning').length;
    if (errors === 0 && warnings === 0) return 'good';
    if (errors === 0 && warnings <= 1) return 'fair';
    return 'poor';
  }

  /** Accumulate issues for rep quality scoring. */
  _recordIssues(issues) {
    this._issueHistory.push(...issues);
  }

  /** Clear issue history at the start of a new rep. */
  _startNewRep() {
    this._issueHistory = [];
  }
}
