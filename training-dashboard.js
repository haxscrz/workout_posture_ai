/**
 * training-dashboard.js
 *
 * Handles interaction with the Python Flask Backend (http://localhost:5000).
 * Uploads videos, triggers training, renders live telemetry charts,
 * AND provides video playback with real-time skeleton overlay using MediaPipe.
 */

import { PoseLandmarker, FilesetResolver } from
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import { SkeletonRenderer } from './ui/skeleton.js';

const API_BASE = 'http://localhost:5000/api';

let lossChart = null;
let accChart = null;

// ─── Video Preview State ──────────────────────────────────────────────────────
let previewPoseLandmarker = null;
let previewSkeleton = null;
let previewAnimationId = null;
let previewPlaying = false;
let previewFrameCount = 0;
let previewPoseCount = 0;
let pendingFile = null;   // File waiting for upload after preview

export function initDashboard() {
  document.getElementById('btn-dash-upload')?.addEventListener('click', () => {
    document.getElementById('dash-file-upload').click();
  });

  document.getElementById('dash-file-upload')?.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    pendingFile = file;

    // Show video preview immediately (local, no server needed)
    loadVideoPreview(file);

    // Upload to server in parallel
    const exercise = document.getElementById('dash-exercise-select')?.value ?? 'squat';
    const label = document.getElementById('dash-label-select')?.value;
    await uploadVideo(file, exercise, label);
    e.target.value = ''; // reset
  });

  document.getElementById('btn-dash-train')?.addEventListener('click', startTraining);

  // Update label options when exercise changes
  document.getElementById('dash-exercise-select')?.addEventListener('change', (e) => {
    updateLabelOptions(e.target.value);
    fetchDataSummary();
  });

  // Preview controls
  document.getElementById('btn-preview-play')?.addEventListener('click', togglePreviewPlayback);
  document.getElementById('btn-preview-slow')?.addEventListener('click', () => setPreviewSpeed(0.5));
  document.getElementById('btn-preview-normal')?.addEventListener('click', () => setPreviewSpeed(1.0));

  initCharts();
  fetchDataSummary();
}

// ─── Video Preview with Skeleton Overlay ──────────────────────────────────────

async function initPreviewMediaPipe() {
  if (previewPoseLandmarker) return; // Already initialized

  const statusEl = document.getElementById('preview-desc');
  if (statusEl) statusEl.textContent = 'Loading AI model for skeleton detection...';

  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );

  previewPoseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  if (statusEl) statusEl.textContent = 'Upload a video to see what the AI detects.';
}

function loadVideoPreview(file) {
  const videoEl = document.getElementById('preview-video');
  const canvasEl = document.getElementById('preview-canvas');
  const container = document.getElementById('video-preview-container');
  const emptyState = document.getElementById('preview-empty');
  const descEl = document.getElementById('preview-desc');

  if (!videoEl || !canvasEl) return;

  // Stop any current playback
  stopPreviewPlayback();

  // Show preview container, hide empty state
  container.style.display = 'block';
  if (emptyState) emptyState.style.display = 'none';
  if (descEl) descEl.textContent = `Loading: ${file.name}`;

  // Create local URL for the video
  const url = URL.createObjectURL(file);
  videoEl.src = url;

  // Reset stats
  previewFrameCount = 0;
  previewPoseCount = 0;
  updatePreviewStats();

  videoEl.onloadedmetadata = () => {
    // Set canvas size to match video
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;

    // Init skeleton renderer for the preview canvas
    if (!previewSkeleton) {
      previewSkeleton = new SkeletonRenderer(canvasEl);
    } else {
      previewSkeleton.canvas = canvasEl;
      previewSkeleton.ctx = canvasEl.getContext('2d');
    }

    if (descEl) {
      descEl.textContent = `${file.name} — ${videoEl.videoWidth}×${videoEl.videoHeight} — Press Play to see skeleton`;
    }

    // Update play button text
    const playBtn = document.getElementById('btn-preview-play');
    if (playBtn) playBtn.textContent = '▶ Play with Skeleton';
  };
}

async function togglePreviewPlayback() {
  if (previewPlaying) {
    stopPreviewPlayback();
  } else {
    await startPreviewPlayback();
  }
}

async function startPreviewPlayback() {
  const videoEl = document.getElementById('preview-video');
  if (!videoEl || !videoEl.src) return;

  // Init MediaPipe if needed
  await initPreviewMediaPipe();

  previewPlaying = true;
  previewFrameCount = 0;
  previewPoseCount = 0;

  const playBtn = document.getElementById('btn-preview-play');
  if (playBtn) playBtn.textContent = '⏸ Pause';

  // Start video playback
  videoEl.currentTime = 0;
  try { await videoEl.play(); } catch(e) {}

  // Start the detection loop
  previewDetectionLoop();
}

function stopPreviewPlayback() {
  previewPlaying = false;
  const videoEl = document.getElementById('preview-video');
  if (videoEl) videoEl.pause();

  if (previewAnimationId) {
    cancelAnimationFrame(previewAnimationId);
    previewAnimationId = null;
  }

  const playBtn = document.getElementById('btn-preview-play');
  if (playBtn) playBtn.textContent = '▶ Play with Skeleton';
}

function setPreviewSpeed(speed) {
  const videoEl = document.getElementById('preview-video');
  if (videoEl) videoEl.playbackRate = speed;

  // Update active button
  document.getElementById('btn-preview-slow')?.classList.toggle('active', speed === 0.5);
  document.getElementById('btn-preview-normal')?.classList.toggle('active', speed === 1.0);
}

let lastPreviewTime = -1;
let previewSkipCounter = 0;
const PREVIEW_DETECT_INTERVAL = 3;  // Only detect every 3rd rAF frame

function previewDetectionLoop() {
  if (!previewPlaying) return;

  const videoEl = document.getElementById('preview-video');
  const canvasEl = document.getElementById('preview-canvas');

  if (!videoEl || videoEl.paused || videoEl.ended) {
    if (videoEl?.ended) {
      updatePreviewStats();
      const descEl = document.getElementById('preview-desc');
      if (descEl) descEl.textContent = `✓ Analysis complete — ${previewPoseCount} poses detected in ${previewFrameCount} frames`;
      stopPreviewPlayback();
    }
    return;
  }

  // Skip frames: only run heavy detection every Nth frame
  // This lets the video play smoothly at native FPS
  previewSkipCounter++;
  const shouldDetect = (previewSkipCounter % PREVIEW_DETECT_INTERVAL === 0);

  if (shouldDetect && videoEl.readyState >= 2 && previewPoseLandmarker) {
    const currentTime = videoEl.currentTime;
    if (currentTime !== lastPreviewTime) {
      lastPreviewTime = currentTime;
      previewFrameCount++;

      // Use setTimeout(0) to yield to the browser so video decoder isn't starved
      setTimeout(() => {
        if (!previewPlaying) return;
        try {
          const ts = Math.round(performance.now());
          const result = previewPoseLandmarker.detectForVideo(videoEl, ts);
          const landmarks = result.landmarks?.[0] ?? null;

          if (canvasEl.width !== videoEl.videoWidth) {
            canvasEl.width = videoEl.videoWidth;
            canvasEl.height = videoEl.videoHeight;
          }

          if (landmarks) {
            previewPoseCount++;
            previewSkeleton.setFormIssues([]);
            previewSkeleton.draw(landmarks, videoEl.videoWidth, videoEl.videoHeight);
          }
          // Don't clear canvas when no pose — keep last skeleton visible

          if (previewFrameCount % 8 === 0) updatePreviewStats();
        } catch (err) {
          console.warn('[Preview] Detection error:', err);
        }
      }, 0);
    }
  }

  previewAnimationId = requestAnimationFrame(previewDetectionLoop);
}

function updatePreviewStats() {
  const frameEl = document.getElementById('preview-frame-count');
  const poseEl = document.getElementById('preview-pose-count');
  const rateEl = document.getElementById('preview-detection-rate');

  if (frameEl) frameEl.textContent = previewFrameCount;
  if (poseEl) poseEl.textContent = previewPoseCount;
  if (rateEl) {
    if (previewFrameCount > 0) {
      const rate = Math.round((previewPoseCount / previewFrameCount) * 100);
      rateEl.textContent = `${rate}%`;
      rateEl.style.color = rate > 80 ? 'var(--green)' : rate > 50 ? 'var(--amber)' : 'var(--red)';
    } else {
      rateEl.textContent = '—';
    }
  }
}

// ─── Exercise Labels ──────────────────────────────────────────────────────────

const EXERCISE_LABELS = {
  squat: [
    { value: 'GOOD_FORM',      text: '✅ Good Form' },
    { value: 'KNEES_CAVING',   text: '❌ Knees Caving In' },
    { value: 'BACK_ROUNDING',  text: '❌ Back Rounding' },
    { value: 'SHALLOW_DEPTH',  text: '❌ Shallow Depth' },
    { value: 'HEELS_RISING',   text: '❌ Heels Rising' },
  ],
  'bicep-curl': [
    { value: 'GOOD_FORM',            text: '✅ Good Form' },
    { value: 'ELBOW_DRIFT',          text: '❌ Elbow Drifting' },
    { value: 'TORSO_SWAY',           text: '❌ Torso Swinging' },
    { value: 'INCOMPLETE_CURL',      text: '❌ Incomplete Curl' },
    { value: 'INCOMPLETE_EXTENSION', text: '❌ Incomplete Extension' },
  ],
  pushup: [
    { value: 'GOOD_FORM',     text: '✅ Good Form' },
    { value: 'HIP_SAG',       text: '❌ Hip Sagging' },
    { value: 'HIP_PIKE',      text: '❌ Hip Piking' },
    { value: 'HEAD_DROOP',    text: '❌ Head Drooping' },
    { value: 'ELBOW_FLARE',   text: '❌ Elbow Flare' },
  ],
};

function updateLabelOptions(exercise) {
  const select = document.getElementById('dash-label-select');
  if (!select) return;

  const labels = EXERCISE_LABELS[exercise] || EXERCISE_LABELS.squat;
  select.innerHTML = labels.map(l =>
    `<option value="${l.value}">${l.text}</option>`
  ).join('');
}

// ─── Upload ───────────────────────────────────────────────────────────────────

async function uploadVideo(file, exercise, label) {
  const statusEl = document.getElementById('upload-status');
  statusEl.textContent = 'Uploading...';
  statusEl.style.color = 'var(--cyan)';

  const formData = new FormData();
  formData.append('video', file);
  formData.append('exercise', exercise);
  formData.append('label', label);

  try {
    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData
    });
    const json = await res.json();
    if (json.success) {
      statusEl.textContent = `✓ Uploaded! ${file.name} (${json.size_mb} MB) → ${exercise}/${label}`;
      statusEl.style.color = 'var(--green)';
      fetchDataSummary();
    } else {
      throw new Error(json.error);
    }
  } catch (err) {
    statusEl.textContent = `✗ Upload failed: ${err.message}`;
    statusEl.style.color = 'var(--red)';
  }
}

// ─── Data Summary ─────────────────────────────────────────────────────────────

async function fetchDataSummary() {
  const container = document.getElementById('dash-data-stats');
  if (!container) return;

  try {
    const res = await fetch(`${API_BASE}/data`);
    const data = await res.json();

    if (!data || Object.keys(data).length === 0) {
      container.innerHTML = '<div class="dash-stat-row"><span>No training data uploaded yet.</span></div>';
      return;
    }

    let html = '';
    for (const [exercise, labels] of Object.entries(data)) {
      html += `<div class="dash-stat-header">${exercise}</div>`;
      for (const [label, count] of Object.entries(labels)) {
        html += `<div class="dash-stat-row">
          <span>${label.replace(/_/g, ' ')}</span>
          <span style="color:var(--cyan); font-weight:bold;">${count} video${count !== 1 ? 's' : ''}</span>
        </div>`;
      }
    }
    container.innerHTML = html;
  } catch (err) {
    container.innerHTML = `<div class="dash-stat-row"><span style="color:var(--red)">Python server not running. Start it with: python server.py</span></div>`;
  }
}

// ─── Charts ───────────────────────────────────────────────────────────────────

function initCharts() {
  const lossCtx = document.getElementById('loss-chart')?.getContext('2d');
  const accCtx = document.getElementById('accuracy-chart')?.getContext('2d');
  if (!lossCtx || !accCtx) return;

  Chart.defaults.color = '#a0aabf';
  Chart.defaults.font.family = "'Space Grotesk', sans-serif";

  lossChart = new Chart(lossCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Train Loss', data: [], borderColor: '#a78bfa', tension: 0.3 },
        { label: 'Val Loss', data: [], borderColor: '#ff4757', tension: 0.3 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top' }, title: { display: true, text: 'Loss Curve' } },
      scales: { y: { beginAtZero: true } }
    }
  });

  accChart = new Chart(accCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Train Acc', data: [], borderColor: '#00e5cc', tension: 0.3 },
        { label: 'Val Acc', data: [], borderColor: '#008cff', tension: 0.3 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top' }, title: { display: true, text: 'Accuracy Curve' } },
      scales: { y: { beginAtZero: true, max: 1.0 } }
    }
  });
}

// ─── Training ─────────────────────────────────────────────────────────────────

async function startTraining() {
  const terminal = document.getElementById('dash-terminal');
  const btn = document.getElementById('btn-dash-train');
  if (!terminal || !btn) return;

  terminal.innerHTML = '';
  btn.disabled = true;
  btn.textContent = 'Training in progress...';

  // Reset charts
  if (lossChart) {
    lossChart.data.labels = [];
    lossChart.data.datasets.forEach(d => d.data = []);
    lossChart.update();
  }
  if (accChart) {
    accChart.data.labels = [];
    accChart.data.datasets.forEach(d => d.data = []);
    accChart.update();
  }

  const log = (msg, type = '') => {
    terminal.innerHTML += `<div class="terminal-line ${type}">> ${msg}</div>`;
    terminal.scrollTop = terminal.scrollHeight;
  };

  try {
    const res = await fetch(`${API_BASE}/train`, { method: 'POST' });
    const json = await res.json();
    if (!json.success) throw new Error(json.message);

    log('Training initiated...', 'system');

    // Connect to SSE stream
    const evtSource = new EventSource(`${API_BASE}/stream`);

    evtSource.onmessage = (e) => {
      let msg;
      try {
        msg = JSON.parse(e.data);
      } catch {
        return;
      }

      if (msg.type === 'log') {
        log(msg.message);
      } else if (msg.type === 'epoch') {
        if (lossChart) {
          lossChart.data.labels.push(msg.epoch);
          lossChart.data.datasets[0].data.push(msg.loss);
          lossChart.data.datasets[1].data.push(msg.val_loss);
          lossChart.update();
        }
        if (accChart) {
          accChart.data.labels.push(msg.epoch);
          accChart.data.datasets[0].data.push(msg.accuracy);
          accChart.data.datasets[1].data.push(msg.val_accuracy);
          accChart.update();
        }
        log(`Epoch ${msg.epoch}/${msg.total_epochs} — Loss: ${msg.loss.toFixed(4)} — Acc: ${(msg.accuracy * 100).toFixed(1)}%`);
      } else if (msg.type === 'complete') {
        log(msg.message, 'success');
        evtSource.close();
        btn.disabled = false;
        btn.textContent = '🚀 Start Training';
        fetchDataSummary();
      } else if (msg.type === 'error') {
        log(msg.message, 'error');
        evtSource.close();
        btn.disabled = false;
        btn.textContent = '🚀 Start Training';
      }
    };

    evtSource.onerror = () => {
      log('Lost connection to training server.', 'error');
      evtSource.close();
      btn.disabled = false;
      btn.textContent = '🚀 Start Training';
    };

  } catch (err) {
    log(`Failed to start training: ${err.message}`, 'error');
    log('Make sure the Python server is running: cd training-server && python server.py', 'system');
    btn.disabled = false;
    btn.textContent = '🚀 Start Training';
  }
}
