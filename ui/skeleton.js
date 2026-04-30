/**
 * skeleton.js — Canvas skeleton renderer
 * Draws the MediaPipe pose skeleton on a canvas element with joint coloring
 * based on active form issues from the exercise analyzer.
 */

import { LM } from '../pose-engine.js';

// Skeleton connections [fromId, toId]
const CONNECTIONS = [
  // Face
  [LM.LEFT_EAR, LM.LEFT_EYE], [LM.LEFT_EYE, LM.NOSE],
  [LM.NOSE, LM.RIGHT_EYE], [LM.RIGHT_EYE, LM.RIGHT_EAR],
  // Torso
  [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER],
  [LM.LEFT_SHOULDER, LM.LEFT_HIP], [LM.RIGHT_SHOULDER, LM.RIGHT_HIP],
  [LM.LEFT_HIP, LM.RIGHT_HIP],
  // Left arm
  [LM.LEFT_SHOULDER, LM.LEFT_ELBOW], [LM.LEFT_ELBOW, LM.LEFT_WRIST],
  // Right arm
  [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW], [LM.RIGHT_ELBOW, LM.RIGHT_WRIST],
  // Left leg
  [LM.LEFT_HIP, LM.LEFT_KNEE], [LM.LEFT_KNEE, LM.LEFT_ANKLE],
  [LM.LEFT_ANKLE, LM.LEFT_HEEL], [LM.LEFT_HEEL, LM.LEFT_FOOT_INDEX],
  // Right leg
  [LM.RIGHT_HIP, LM.RIGHT_KNEE], [LM.RIGHT_KNEE, LM.RIGHT_ANKLE],
  [LM.RIGHT_ANKLE, LM.RIGHT_HEEL], [LM.RIGHT_HEEL, LM.RIGHT_FOOT_INDEX],
];

const COLORS = {
  default:  '#00e5cc',   // teal  — normal joint
  error:    '#ff4757',   // red   — form error
  warning:  '#ffa502',   // amber — form warning
  bone:     'rgba(0,229,204,0.35)',
  boneBad:  'rgba(255,71,87,0.5)',
};

export class SkeletonRenderer {
  constructor(canvasEl) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext('2d');
    // Pre-build error joint set for O(1) lookup each frame
    this._errorJoints   = new Map(); // jointId → severity
  }

  /**
   * Update which joints have issues (called each frame with analyzer result).
   * @param {FormIssue[]} formIssues
   */
  setFormIssues(formIssues = []) {
    this._errorJoints.clear();
    for (const issue of formIssues) {
      for (const id of issue.jointIds) {
        // Keep the most severe issue if a joint has multiple
        const existing = this._errorJoints.get(id);
        if (!existing || issue.severity === 'error') {
          this._errorJoints.set(id, issue.severity);
        }
      }
    }
  }

  /**
   * Draw the full skeleton for a single frame.
   * @param {Array} landmarks - Normalized [{x,y,z,visibility}×33]
   * @param {number} videoWidth
   * @param {number} videoHeight
   */
  draw(landmarks, videoWidth, videoHeight) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (!landmarks || landmarks.length === 0) return;

    const scaleX = w;
    const scaleY = h;

    const px = (lm) => lm ? lm.x * scaleX : null;
    const py = (lm) => lm ? lm.y * scaleY : null;

    // ── Draw bones (connections) ─────────────────────────────────────────
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';

    for (const [aId, bId] of CONNECTIONS) {
      const a = landmarks[aId];
      const b = landmarks[bId];
      if (!a || !b) continue;
      if ((a.visibility != null && a.visibility < 0.4) ||
          (b.visibility != null && b.visibility < 0.4)) continue;

      const hasError = this._errorJoints.has(aId) || this._errorJoints.has(bId);
      const severityA = this._errorJoints.get(aId);
      const severityB = this._errorJoints.get(bId);
      const isError = severityA === 'error' || severityB === 'error';

      ctx.beginPath();
      ctx.strokeStyle = hasError
        ? (isError ? COLORS.boneBad : 'rgba(255,165,2,0.45)')
        : COLORS.bone;
      ctx.moveTo(px(a), py(a));
      ctx.lineTo(px(b), py(b));
      ctx.stroke();
    }

    // ── Draw joints ──────────────────────────────────────────────────────
    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      if (!lm) continue;
      if (lm.visibility != null && lm.visibility < 0.4) continue;

      const x = lm.x * scaleX;
      const y = lm.y * scaleY;
      const severity = this._errorJoints.get(i);

      let color = COLORS.default;
      let radius = 4;
      if (severity === 'error') {
        color = COLORS.error;
        radius = 6;
      } else if (severity === 'warning') {
        color = COLORS.warning;
        radius = 5;
      }

      // Outer glow ring
      const glowColor = this._hexToRgba(color, 0.2);
      ctx.beginPath();
      ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
      ctx.fillStyle = glowColor;
      ctx.shadowColor = color;
      ctx.shadowBlur = severity ? 12 : 6;
      ctx.fill();

      // Inner dot
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.shadowBlur = 0;
      ctx.fill();
    }

    ctx.shadowBlur = 0;
  }

  /** Convert hex color to rgba string. */
  _hexToRgba(hex, alpha) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!result) return `rgba(0,229,204,${alpha})`; // fallback to teal
    const r = parseInt(result[1], 16);
    const g = parseInt(result[2], 16);
    const b = parseInt(result[3], 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  /** Show a subtle "searching for body" animation when no landmarks. */
  drawSearching() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    // Draw a pulsing silhouette outline — handled by CSS animation on overlay
  }
}
