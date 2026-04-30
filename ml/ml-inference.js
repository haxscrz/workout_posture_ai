/**
 * ml-inference.js — Lightweight browser-side ML model inference
 *
 * Loads a trained TensorFlow.js model exported by the Python training pipeline
 * and provides a simple predict() API for real-time pose classification.
 *
 * Used as an ENHANCEMENT layer alongside the rule-based analyzers:
 *   - Rules provide explainable, specific coaching text
 *   - ML provides confidence scores and catches edge cases
 *
 * Model location: public/models/<exercise>/model.json
 * Labels location: public/models/<exercise>/labels.json
 */

import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/+esm';
import {
  LM, getLM, computeAngle, midpoint,
  getBodyHeight, getBodyWidth
} from '../pose-engine.js';

const WINDOW_SIZE = 15;   // Must match the training pipeline's window size
const MODEL_BASE = 'models';  // Relative to public/

export class MLInference {
  constructor() {
    this._model = null;
    this._labels = null;
    this._frameBuffer = [];
    this._featureCount = 0;
    this._ready = false;
    this._baseline = {};
    this._prevFeatures = null;  // For velocity computation
  }

  get isReady() { return this._ready; }
  get labels() { return this._labels; }

  /**
   * Attempt to load a trained model for the given exercise.
   * Fails silently if no model is found — the app falls back to rules only.
   * @param {string} exercise — e.g. 'squat', 'bicep-curl', 'pushup'
   * @returns {Promise<boolean>} — true if loaded successfully
   */
  async load(exercise) {
    try {
      const modelUrl = `${MODEL_BASE}/${exercise}/model.json`;
      const labelsUrl = `${MODEL_BASE}/${exercise}/labels.json`;

      // Load labels first (lightweight check if model exists)
      const labelsRes = await fetch(labelsUrl);
      if (!labelsRes.ok) {
        console.log(`[MLInference] No trained model found for ${exercise} — using rules only.`);
        return false;
      }
      this._labels = await labelsRes.json();

      // Load TF.js model
      this._model = await tf.loadLayersModel(modelUrl);
      this._featureCount = Math.round(this._model.inputs[0].shape[2] || 12);
      this._frameBuffer = [];
      this._baseline = {};
      this._prevFeatures = null;
      this._ready = true;

      console.log(`[MLInference] Model loaded for ${exercise} — ${this._labels.length} classes, ${this._featureCount} features`);
      return true;
    } catch (e) {
      console.log(`[MLInference] Could not load model for ${exercise}: ${e.message}`);
      this._ready = false;
      return false;
    }
  }

  /**
   * Set baseline measurements (call at the start of a workout, when user is standing).
   * @param {Array} landmarks — MediaPipe normalized landmarks
   */
  setBaseline(landmarks) {
    if (!landmarks) return;
    const lHeel = getLM(landmarks, LM.LEFT_HEEL);
    const rHeel = getLM(landmarks, LM.RIGHT_HEEL);
    if (lHeel) this._baseline.leftHeelY = lHeel.y;
    if (rHeel) this._baseline.rightHeelY = rHeel.y;
  }

  /**
   * Feed a single frame of landmarks and get a prediction.
   * Returns null until the sliding window is full.
   *
   * @param {Array} landmarks — MediaPipe normalized landmarks
   * @returns {{ label: string, confidence: number, probabilities: Object }|null}
   */
  predict(landmarks) {
    if (!this._ready || !this._model || !landmarks) return null;

    const features = this._extractFeatures(landmarks);
    if (!features) return null;

    // Add to sliding window
    this._frameBuffer.push(features);
    if (this._frameBuffer.length > WINDOW_SIZE) {
      this._frameBuffer.shift();
    }

    // Need a full window before predicting
    if (this._frameBuffer.length < WINDOW_SIZE) return null;

    // Build input tensor: [1, WINDOW_SIZE, featureCount]
    const inputData = this._frameBuffer.map(f => f.slice(0, this._featureCount));
    const inputTensor = tf.tensor3d([inputData]);
    const outputTensor = this._model.predict(inputTensor);
    const probs = outputTensor.dataSync();

    inputTensor.dispose();
    outputTensor.dispose();

    // Find best class
    let maxIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }

    const probabilities = {};
    for (let i = 0; i < this._labels.length; i++) {
      probabilities[this._labels[i]] = probs[i];
    }

    return {
      label: this._labels[maxIdx],
      confidence: probs[maxIdx],
      probabilities,
    };
  }

  /** Reset the sliding window (call when starting a new workout session). */
  reset() {
    this._frameBuffer = [];
    this._baseline = {};
    this._prevFeatures = null;
  }

  /**
   * Extract features from landmarks — MUST match the Python feature_extractor.py exactly.
   * Returns an array of floats, or null if landmarks are insufficient.
   */
  _extractFeatures(landmarks) {
    if (!landmarks || landmarks.length < 33) return null;

    const lShoulder = getLM(landmarks, LM.LEFT_SHOULDER);
    const rShoulder = getLM(landmarks, LM.RIGHT_SHOULDER);
    const lHip      = getLM(landmarks, LM.LEFT_HIP);
    const rHip      = getLM(landmarks, LM.RIGHT_HIP);
    const lKnee     = getLM(landmarks, LM.LEFT_KNEE);
    const rKnee     = getLM(landmarks, LM.RIGHT_KNEE);
    const lAnkle    = getLM(landmarks, LM.LEFT_ANKLE);
    const rAnkle    = getLM(landmarks, LM.RIGHT_ANKLE);
    const lHeel     = getLM(landmarks, LM.LEFT_HEEL);
    const rHeel     = getLM(landmarks, LM.RIGHT_HEEL);
    const nose      = getLM(landmarks, LM.NOSE);

    if (!lShoulder || !rShoulder || !lHip || !rHip || !lKnee || !rKnee || !lAnkle || !rAnkle) {
      return null;
    }

    const midShoulder = midpoint(lShoulder, rShoulder);
    const midHip      = midpoint(lHip, rHip);
    const midKnee     = midpoint(lKnee, rKnee);

    const bodyW = getBodyWidth(landmarks);
    const bodyH = getBodyHeight(landmarks);

    // 1. Joint Angles (normalized 0-1, using 3D)
    const leftKneeAngle  = (computeAngle(lHip, lKnee, lAnkle) ?? 180) / 180;
    const rightKneeAngle = (computeAngle(rHip, rKnee, rAnkle) ?? 180) / 180;
    const backAngle      = (computeAngle(midShoulder, midHip, midKnee) ?? 180) / 180;

    // 2. Knee Cave Ratio (Mirror Invariant)
    const centerX = midHip.x;
    const lKneeDist = Math.abs(lKnee.x - centerX);
    const lAnkleDist = Math.abs(lAnkle.x - centerX);
    const leftKneeCaveRatio = bodyW > 0.01 ? (lAnkleDist - lKneeDist) / bodyW : 0;

    const rKneeDist = Math.abs(rKnee.x - centerX);
    const rAnkleDist = Math.abs(rAnkle.x - centerX);
    const rightKneeCaveRatio = bodyW > 0.01 ? (rAnkleDist - rKneeDist) / bodyW : 0;

    // 3. Hip Depth Ratio
    const hipY = midHip.y;
    const ankleY = (lAnkle.y + rAnkle.y) / 2;
    const hipDepthRatio = bodyH > 0.01 ? (ankleY - hipY) / bodyH : 0.5;

    // 4. Heel Delta
    const leftHeelDelta = (lHeel && this._baseline.leftHeelY !== null && bodyH > 0.01)
      ? (lHeel.y - this._baseline.leftHeelY) / bodyH : 0;
    const rightHeelDelta = (rHeel && this._baseline.rightHeelY !== null && bodyH > 0.01)
      ? (rHeel.y - this._baseline.rightHeelY) / bodyH : 0;

    // 5. Shoulder Z Depth Difference
    const shoulderZDiff = Math.abs((lShoulder.z || 0) - (rShoulder.z || 0));

    // 6. Torso Lean Ratio (Mirror invariant - absolute lean)
    const torsoLeanRatio = bodyW > 0.01 ? Math.abs(midShoulder.x - midHip.x) / bodyW : 0;

    // 7. Hip Symmetry (Absolute difference)
    const hipSymmetry = bodyH > 0.01 ? Math.abs(lHip.y - rHip.y) / bodyH : 0;

    // 8. Knee Angle Difference
    const kneeAngleDiff = leftKneeAngle - rightKneeAngle;

    const features = [
      leftKneeAngle,
      rightKneeAngle,
      backAngle,
      leftKneeCaveRatio,
      rightKneeCaveRatio,
      hipDepthRatio,
      leftHeelDelta,
      rightHeelDelta,
      shoulderZDiff,
      torsoLeanRatio,
      hipSymmetry,
      kneeAngleDiff,
    ];

    this._prevFeatures = features;
    return features;
  }
}
