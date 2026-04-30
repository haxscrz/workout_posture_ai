/**
 * app.js — Main application controller
 * Orchestrates MediaPipe, exercise analyzers, setup wizard, ML inference, and UI.
 *
 * ARCHITECTURE:
 *   Rule-based analyzers (exercises/*.js) — primary, always active
 *   ML inference (ml/ml-inference.js) — enhancement layer, active when trained model exists
 *
 * PERFORMANCE: Uses pose_landmarker_lite model + frame-skip throttling
 * to stay within ~20-25 FPS budget on mid-range mobile hardware.
 */

import { PoseLandmarker, FilesetResolver } from
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

import { BicepCurlAnalyzer } from './exercises/bicep-curl.js';
import { PushUpAnalyzer }    from './exercises/pushup.js';
import { SquatAnalyzer }     from './exercises/squat.js';
import { SetupWizard }       from './setup-wizard.js';
import { SkeletonRenderer }  from './ui/skeleton.js';
import { FeedbackManager }   from './ui/feedback.js';
import { smoothLandmarks }   from './pose-engine.js';

// ML inference (loads pre-trained models if available)
import { MLInference }       from './ml/ml-inference.js';

// ML Dashboard (Python training backend)
import { initDashboard }     from './training-dashboard.js';

// ─── Exercise Registry ────────────────────────────────────────────────────────
const EXERCISES = {
  'bicep-curl': {
    label: 'Bicep Curl',
    Analyzer: BicepCurlAnalyzer,
    icon: '💪',
    color: '#00e5cc',
    modelKey: 'bicep-curl',
  },
  'pushup': {
    label: 'Push-Up',
    Analyzer: PushUpAnalyzer,
    icon: '🫸',
    color: '#a78bfa',
    modelKey: 'pushup',
  },
  'squat': {
    label: 'Squat',
    Analyzer: SquatAnalyzer,
    icon: '🏋',
    color: '#fbbf24',
    modelKey: 'squat',
  },
};

// ─── App State ────────────────────────────────────────────────────────────────
const state = {
  phase: 'landing',         // landing | loading | setup | countdown | workout | summary | dashboard
  exerciseKey: null,
  analyzer: null,
  wizard: null,
  smoothedLandmarks: null,
  smoothedWorldLandmarks: null,
  poseLandmarker: null,
  lastVideoTime: -1,
  frameSkip: 0,             // frame counter for throttling
  FRAME_SKIP_INTERVAL: 2,   // process every Nth frame (perf tuning)
  sessionStats: {
    repCount: 0,
    goodReps: 0,
    fairReps: 0,
    poorReps: 0,
    issueCodes: {},
  },
  // ML inference
  mlInference: null,
  mlPrediction: null,       // Latest ML prediction (or null)
};

// ─── DOM refs ─────────────────────────────────────────────────────────────────
let videoEl, canvasEl, skeleton, feedback;

// ─── MediaPipe Init ───────────────────────────────────────────────────────────
async function initMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );

  state.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      // LITE model — ~30% faster than full, designed for mobile
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

// ─── Camera ───────────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user',
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false,
    });
    videoEl.srcObject = stream;

    return new Promise((resolve) => {
      videoEl.onloadedmetadata = async () => {
        try { await videoEl.play(); } catch(e) {}
        resolve(videoEl);
      };
    });
  } catch (err) {
    // Handle camera permission denial or no camera available
    console.error('[FormAI] Camera error:', err);

    let message = 'Camera access failed.';
    if (err.name === 'NotAllowedError') {
      message = 'Camera permission denied. Please allow camera access in your browser settings and reload.';
    } else if (err.name === 'NotFoundError') {
      message = 'No camera found. Please connect a camera and reload.';
    } else if (err.name === 'NotReadableError') {
      message = 'Camera is in use by another app. Close it and try again.';
    }

    // Show error to user
    showView('view-loading');
    document.getElementById('loading-label').textContent = message;
    document.querySelector('.loader')?.classList.add('hidden');
    throw err;
  }
}

// ─── Detection Loop ───────────────────────────────────────────────────────────
// Active phases: the loop runs continuously during these phases.
const ACTIVE_PHASES = new Set(['setup', 'countdown', 'workout']);

function detectionLoop() {
  // If we're not in an active phase (e.g. landing, loading, summary), stop the loop entirely.
  if (!ACTIVE_PHASES.has(state.phase)) return;

  // Keep looping even if not ready yet — the video may still be loading.
  if (!state.poseLandmarker || videoEl.readyState < 2) {
    requestAnimationFrame(detectionLoop);
    return;
  }

  // Frame-skip: only run inference every N frames for mobile perf
  state.frameSkip = (state.frameSkip + 1) % state.FRAME_SKIP_INTERVAL;
  const shouldDetect = state.frameSkip === 0;

  const now = performance.now();
  if (shouldDetect && (state.lastVideoTime === -1 || now - state.lastVideoTime > 20)) {
    state.lastVideoTime = now;

    try {
      const ts = Math.round(now);
      const result = state.poseLandmarker.detectForVideo(videoEl, ts);
      let landmarks = result.landmarks?.[0] ?? null;
      let worldLandmarks = result.worldLandmarks?.[0] ?? null;

      // Apply landmark smoothing to reduce jitter
      if (landmarks) {
        landmarks = smoothLandmarks(state.smoothedLandmarks, landmarks, 0.6);
        state.smoothedLandmarks = landmarks;
      }
      if (worldLandmarks) {
        worldLandmarks = smoothLandmarks(state.smoothedWorldLandmarks, worldLandmarks, 0.6);
        state.smoothedWorldLandmarks = worldLandmarks;
      }

      // Sync canvas size to video
      if (canvasEl.width !== videoEl.videoWidth) {
        canvasEl.width  = videoEl.videoWidth;
        canvasEl.height = videoEl.videoHeight;
      }

      // Always draw skeleton if we have landmarks, regardless of phase
      if (landmarks) {
        skeleton.draw(landmarks, videoEl.videoWidth, videoEl.videoHeight);
      } else {
        // No person detected — clear the canvas so stale skeleton doesn't linger
        skeleton.draw([], 0, 0);
      }

      // Phase-specific logic (analysis, setup wizard, etc.)
      if (state.phase === 'setup') {
        handleSetupFrame(landmarks);
      } else if (state.phase === 'workout') {
        handleWorkoutFrame(landmarks, worldLandmarks);
      }
      // 'countdown' phase: skeleton is drawn above, no analysis needed.

    } catch (err) {
      console.error('[FormAI] Error in detection loop:', err);
      const debugEl = document.getElementById('angle-debug');
      if (debugEl) {
        debugEl.style.display = 'block';
        debugEl.style.color = 'red';
        debugEl.innerHTML = `Crash: ${err.message}<br>${err.stack}`;
      }
    }
  }

  // CRITICAL: Always schedule the next frame. Never let the loop die.
  requestAnimationFrame(detectionLoop);
}

// ─── Setup Phase ──────────────────────────────────────────────────────────────
function handleSetupFrame(landmarks) {
  skeleton.setFormIssues([]);

  const evaluation = state.wizard.evaluate(landmarks ?? []);
  feedback.feedbackEl = feedback._setupPanel;
  feedback.showSetupStatus(evaluation.checks);

  // Update progress ring
  const progressRing = document.getElementById('setup-progress-ring');
  if (progressRing) {
    const pct = Math.round(evaluation.readyProgress * 100);
    progressRing.style.setProperty('--progress', pct);
    progressRing.querySelector('.progress-pct').textContent = evaluation.passed ? '✓' : `${pct}%`;
  }

  if (evaluation.passed) {
    // Set ML baseline from the standing position
    if (state.mlInference?.isReady) {
      state.mlInference.setBaseline(landmarks);
    }
    startCountdown();
  }
}

function startCountdown() {
  if (state.phase !== 'setup') return;
  state.phase = 'countdown';

  let count = 3;
  const countdownEl = document.getElementById('countdown-overlay');
  countdownEl.classList.remove('hidden');

  const tick = () => {
    countdownEl.querySelector('.countdown-number').textContent = count;
    countdownEl.querySelector('.countdown-number').classList.remove('countdown-pop');
    void countdownEl.querySelector('.countdown-number').offsetWidth;
    countdownEl.querySelector('.countdown-number').classList.add('countdown-pop');

    if (count === 0) {
      countdownEl.classList.add('hidden');
      beginWorkout();
      return;
    }
    count--;
    setTimeout(tick, 1000);
  };
  tick();
}

// ─── Workout Phase ────────────────────────────────────────────────────────────
function beginWorkout() {
  state.phase = 'workout';
  state.analyzer.reset();
  if (state.mlInference?.isReady) state.mlInference.reset();
  feedback.announce(`Starting ${EXERCISES[state.exerciseKey].label}. Let's go!`);
  document.getElementById('setup-panel').style.display = 'none';
  document.getElementById('rep-card').style.display = 'block';
  document.getElementById('workout-controls-bar').classList.remove('hidden');
  feedback.feedbackEl = feedback._workoutPanel;

  // Show/hide ML confidence badge
  const mlBadge = document.getElementById('ml-confidence-badge');
  if (mlBadge) mlBadge.style.display = state.mlInference?.isReady ? 'flex' : 'none';
}

function handleWorkoutFrame(landmarks, worldLandmarks) {
  if (!landmarks) return;

  const result = state.analyzer.analyze(landmarks, worldLandmarks ?? []);

  // Update skeleton colors (the main loop handles the actual draw call)
  skeleton.setFormIssues(result.formIssues);

  // Run ML inference (enhancement layer)
  if (state.mlInference?.isReady) {
    const mlResult = state.mlInference.predict(landmarks);
    if (mlResult) {
      state.mlPrediction = mlResult;
      updateMLBadge(mlResult);
    }
  }

  // Update feedback UI + TTS
  feedback.update(result);

  // Track session stats
  if (result.repQuality) {
    state.sessionStats.repCount = result.repCount;
    if (result.repQuality === 'good')      state.sessionStats.goodReps++;
    else if (result.repQuality === 'fair') state.sessionStats.fairReps++;
    else                                   state.sessionStats.poorReps++;

    // Track which form issues happened most
    for (const issue of result.formIssues) {
      state.sessionStats.issueCodes[issue.code] =
        (state.sessionStats.issueCodes[issue.code] ?? 0) + 1;
    }
  }
}

function updateMLBadge(prediction) {
  const badge = document.getElementById('ml-confidence-badge');
  const label = document.getElementById('ml-confidence-label');
  const value = document.getElementById('ml-confidence-value');
  if (!badge || !label || !value) return;

  const isGood = prediction.label === 'GOOD_FORM';
  const pct = Math.round(prediction.confidence * 100);

  label.textContent = isGood ? 'Good Form' : prediction.label.replace(/_/g, ' ').toLowerCase();
  value.textContent = `${pct}%`;
  badge.className = `ml-confidence-badge ${isGood ? 'ml-good' : 'ml-issue'}`;
}

// ─── View Management ──────────────────────────────────────────────────────────
function showView(id) {
  document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
  document.getElementById(id)?.classList.remove('hidden');
}

// ─── Landing Page ─────────────────────────────────────────────────────────────
function initLandingPage() {
  document.querySelectorAll('.exercise-card').forEach(card => {
    card.addEventListener('click', async () => {
      // Must be called synchronously in the click handler for iOS Safari to allow audio
      if (feedback) {
        feedback.unlockTTS();
      }
      const key = card.dataset.exercise;
      await selectExercise(key);
    });
  });
}

async function selectExercise(key) {
  state.exerciseKey = key;
  const meta = EXERCISES[key];

  // Show loading state
  showView('view-loading');
  document.querySelector('.loader')?.classList.remove('hidden');
  document.getElementById('loading-label').textContent = `Loading ${meta.label}...`;

  // Init MediaPipe if first time
  if (!state.poseLandmarker) {
    await initMediaPipe();
  }

  // Init exercise
  state.analyzer = new meta.Analyzer();
  state.wizard   = new SetupWizard(state.analyzer.getRequirements());

  // Reset smoothing
  state.smoothedLandmarks = null;
  state.smoothedWorldLandmarks = null;

  // Try loading ML model for this exercise (non-blocking)
  if (!state.mlInference) {
    state.mlInference = new MLInference();
  }
  state.mlInference.load(meta.modelKey).catch(() => {});

  // Start camera
  await startCamera();
  await videoEl.play();

  // Reset session stats
  state.sessionStats = { repCount: 0, goodReps: 0, fairReps: 0, poorReps: 0, issueCodes: {} };

  // Populate setup view
  const reqs = state.analyzer.getRequirements();
  document.getElementById('setup-exercise-name').textContent = meta.label;
  document.getElementById('setup-instruction').textContent   = reqs.instruction;
  document.getElementById('setup-camera-tip').textContent    = reqs.cameraPlacement;

  state.phase = 'setup';
  showView('view-workout');
  document.getElementById('rep-card').style.display = 'none';
  document.getElementById('workout-controls-bar').classList.add('hidden');
  document.getElementById('setup-panel').style.display = 'flex';

  // Hide ML badge during setup
  const mlBadge = document.getElementById('ml-confidence-badge');
  if (mlBadge) mlBadge.style.display = 'none';

  // Update header
  document.getElementById('workout-exercise-name').textContent = `${meta.icon} ${meta.label}`;

  // Start detection loop
  detectionLoop();
}

// ─── Stop / Summary ───────────────────────────────────────────────────────────
function stopWorkout() {
  state.phase = 'summary';
  feedback.clear();

  const s = state.sessionStats;
  const total = s.repCount;

  document.getElementById('summary-exercise').textContent = EXERCISES[state.exerciseKey]?.label ?? '';
  document.getElementById('summary-total').textContent    = total;
  document.getElementById('summary-good').textContent     = s.goodReps;
  document.getElementById('summary-fair').textContent     = s.fairReps;
  document.getElementById('summary-poor').textContent     = s.poorReps;

  // Top form mistakes
  const sorted = Object.entries(s.issueCodes).sort((a, b) => b[1] - a[1]).slice(0, 3);
  const mistakesEl = document.getElementById('summary-mistakes');
  if (sorted.length === 0) {
    mistakesEl.innerHTML = '<p class="no-mistakes">No major form issues detected. Great work! 🎉</p>';
  } else {
    mistakesEl.innerHTML = sorted.map(([code, count]) =>
      `<div class="mistake-item">
        <span class="mistake-code">${formatCode(code)}</span>
        <span class="mistake-count">${count}×</span>
      </div>`
    ).join('');
  }

  // Stop camera
  videoEl.srcObject?.getTracks().forEach(t => t.stop());

  showView('view-summary');
}

function formatCode(code) {
  // Convert BICEP_ELBOW_DRIFT → "Elbow Drift"
  return code.split('_').slice(1).map(w => w[0] + w.slice(1).toLowerCase()).join(' ');
}

window.addEventListener('error', (e) => {
  const d = document.getElementById('angle-debug');
  if (d) { d.style.display = 'block'; d.style.color = 'red'; d.innerHTML += `<br>Global Error: ${e.message}`; }
});
window.addEventListener('unhandledrejection', (e) => {
  if (e.reason && e.reason.toString().includes('ws.send')) return; // Ignore Vite HMR disconnects
  const d = document.getElementById('angle-debug');
  if (d) { d.style.display = 'block'; d.style.color = 'red'; d.innerHTML += `<br>Promise Error: ${e.reason}`; }
});

// ─── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  videoEl  = document.getElementById('pose-video');
  canvasEl = document.getElementById('pose-canvas');

  skeleton = new SkeletonRenderer(canvasEl);
  feedback = new FeedbackManager({
    feedbackEl:   null, // switched per phase via feedback.setPanel()
    repCountEl:   document.getElementById('rep-count'),
    repQualityEl: document.getElementById('rep-quality'),
    angleDebugEl: document.getElementById('angle-debug'),
  });
  feedback._setupPanel  = document.getElementById('feedback-panel-setup');
  feedback._workoutPanel = document.getElementById('feedback-panel');

  initLandingPage();

  // Stop button
  document.getElementById('btn-stop')?.addEventListener('click', stopWorkout);

  // TTS toggle
  document.getElementById('btn-tts')?.addEventListener('click', (e) => {
    const btn = e.currentTarget;
    const enabled = btn.dataset.enabled !== 'false';
    feedback.setTTSEnabled(!enabled);
    btn.dataset.enabled = !enabled;
    btn.textContent = !enabled ? '🔊' : '🔇';
    btn.title = !enabled ? 'Mute voice' : 'Unmute voice';
  });

  // Summary buttons
  document.getElementById('btn-retry')?.addEventListener('click', () => {
    selectExercise(state.exerciseKey);
  });
  document.getElementById('btn-back-home')?.addEventListener('click', () => {
    videoEl.srcObject?.getTracks().forEach(t => t.stop());
    state.phase = 'landing';
    showView('view-landing');
  });

  // ─── ML Dashboard buttons ───────────────────────────────────────────────
  document.getElementById('btn-open-dashboard')?.addEventListener('click', () => {
    state.phase = 'dashboard';
    showView('view-dashboard');
  });
  document.getElementById('btn-dashboard-back')?.addEventListener('click', () => {
    state.phase = 'landing';
    showView('view-landing');
  });
  initDashboard();

  // ─── Test Video Upload ────────────────────────────────────────────────
  document.getElementById('btn-test-video')?.addEventListener('click', () => {
    document.getElementById('test-video-input')?.click();
  });
  
  document.getElementById('test-video-input')?.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Stop live camera
    if (videoEl.srcObject) {
      videoEl.srcObject.getTracks().forEach(t => t.stop());
      videoEl.srcObject = null;
    }

    // Play uploaded video
    videoEl.src = URL.createObjectURL(file);
    videoEl.loop = true;
    await videoEl.play();

    // Force setup complete after a short delay (so MediaPipe reads the first frame)
    setTimeout(() => {
      if (state.phase === 'setup') {
        if (state.mlInference?.isReady && state.smoothedLandmarks) {
          state.mlInference.setBaseline(state.smoothedLandmarks);
        }
        startCountdown();
      }
    }, 1500);
  });

  showView('view-landing');
});
