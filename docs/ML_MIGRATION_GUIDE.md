# 🧠 Gym Posture AI — ML Migration Guide

This document outlines the roadmap for migrating the current **rule-based** posture correction system to a **data-driven Machine Learning (ML) classifier**. 

Currently, the app uses MediaPipe Tasks Vision to extract 3D landmarks, and relies on hard-coded trigonometry (e.g., `bicep-curl.js`) to evaluate form. While robust, an ML approach allows the system to recognize nuanced form issues without manually tuning angles for every body type.

---

## Phase 1: Data Collection & Feature Extraction

To train an ML model, you need a dataset of both **good form** and **bad form** for each exercise.

### 1. Leverage Open-Source Datasets (Kaggle)
Instead of recording thousands of videos yourself, utilize existing fitness datasets:
- [Kaggle: Workout Freaks / Exercise Pose Dataset](https://www.kaggle.com/datasets)
- **Penn Action Dataset**: Contains 2326 video sequences of 15 different actions.
- **Fitness-AQA**: A dataset specifically designed for assessing exercise form quality.

### 2. Feature Engineering
Raw 3D coordinates ($X, Y, Z$) are sensitive to camera distance and user placement. Instead of feeding raw coordinates to the neural network, extract **normalized features**:
1. **Relative Angles**: The `computeAngle()` utility in `pose-engine.js` is perfect for this. Extract arrays of joint angles (e.g., elbow, shoulder, hip, knee) per frame.
2. **Velocity/Acceleration**: The change in angles over time ($\Delta \text{Angle}$).
3. **Bounding Box Normalization**: If using raw coordinates, normalize them so the user's torso always spans from 0.0 to 1.0.

> [!TIP]
> The current `ExerciseBase.js` structure makes it easy to log data. You can temporarily add a `console.log(angleData)` or a local CSV export function to record your own movements as training data.

---

## Phase 2: Model Architecture

Since this app runs in the browser on mobile devices, the model must be extremely lightweight.

### 1. The Classifier (TensorFlow.js)
Instead of a heavy convolutional neural network (CNN), use a **shallow Dense Neural Network (DNN)** or a **Random Forest** (via something like `ml5.js` or `tfjs`).

**Input**: A 1D array of features (e.g., 5 angles × 10 recent frames = 50 floats).
**Output**: Softmax probabilities for classes (e.g., `[0.85 Good Form, 0.10 Elbow Drift, 0.05 Fast Rep]`).

### 2. Temporal Smoothing (LSTM/GRU)
Form analysis relies on movement over time. A simple Feed-Forward network only looks at a single frame.
- **Approach A (Simple Windowing)**: Feed a sliding window of the last 15 frames into a standard Dense layer.
- **Approach B (RNN)**: Use a lightweight Long Short-Term Memory (LSTM) layer in TensorFlow.js to "remember" the trajectory of the rep.

---

## Phase 3: Integration into the App

The current architecture was specifically designed to support an easy ML drop-in.

### 1. Updating `ExerciseBase.js`
Create a new ML-based analyzer class that extends `ExerciseBase`:

```javascript
import * as tf from '@tensorflow/tfjs';

export class MLBicepCurlAnalyzer extends ExerciseBase {
  constructor() {
    super();
    this.model = await tf.loadLayersModel('assets/models/bicep-curl/model.json');
    this.frameBuffer = [];
  }

  analyze(landmarks, worldLandmarks) {
    // 1. Extract angles (just like the current rule-based system)
    const features = this._extractFeatures(worldLandmarks);
    this.frameBuffer.push(features);

    if (this.frameBuffer.length > 15) this.frameBuffer.shift();

    // 2. Run Inference
    const inputTensor = tf.tensor2d([this.frameBuffer.flat()]);
    const predictions = this.model.predict(inputTensor).dataSync();
    
    // 3. Map predictions to the existing FormIssue format
    const issues = [];
    if (predictions[1] > 0.7) { // Index 1 might represent "Elbow Drift"
      issues.push({ message: 'Keep elbows tucked', severity: 'warning' });
    }

    return { formIssues: issues, ... };
  }
}
```

### 2. Benefits of the Dual Approach
You do not need to replace everything at once. You can use the ML model to detect **rep phases** (Up, Down, Resting) to count reps, while keeping the **rule-based geometry** (Math.abs(elbow.z - shoulder.z)) to provide specific, explainable coaching feedback.

---

## Next Steps for the Developer
1. Install `@tensorflow/tfjs` in your project (`npm install @tensorflow/tfjs`).
2. Create a script to extract angles from a Kaggle dataset using MediaPipe Python, and export to CSV.
3. Train a lightweight Keras model (`.h5`) in Python, and use `tensorflowjs_converter` to export it to `model.json` for the web app.
4. Swap the `Analyzer` class in `app.js` `EXERCISES` registry to test the new model!
