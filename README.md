# Squat Posture AI 🏋️

A Python-powered real-time squat posture analyzer using **MediaPipe** pose detection and a **Conv1D neural network**. It detects 6 types of bad squat form and provides instant visual feedback through your webcam or phone camera.

## Features

- **Real-time skeleton overlay** with color-coded joint feedback
- **Rep counting** with quality scoring (Good / Fair / Poor)
- **6 form error types** detected: Forward Lean, Heel Rise, Knee Cave, Knees Over Toes, Shallow Depth, Uneven Weight
- **Dual detection engine**: Hard-coded rules + trained ML model (97.4% accuracy)
- **Camera switching** on the fly (press C)
- **Phone camera support** via IP Webcam

## Quick Start

### Option 1: Double-click launcher
```
RUN.bat
```

### Option 2: Command line
```bash
# Live webcam analysis
py -3.9 src/main.py --camera

# Phone camera (IP Webcam app)
py -3.9 src/main.py --camera --source "http://192.168.1.X:8080/video"

# Phone camera with 90-degree rotation (useful for vertical phone setups)
py -3.9 src/main.py --camera --rotate 90

# Analyze a video file
py -3.9 src/main.py --video "path/to/squat_video.mp4"

# Train/retrain the model
py -3.9 src/main.py --train
```

## Requirements

- **Python 3.9** (TensorFlow requirement)
- Windows 10/11

### Install dependencies
```bash
py -3.9 -m pip install -r requirements.txt
```

## Controls

| Key | Action |
|-----|--------|
| **Q** / **ESC** | Quit |
| **R** | Reset rep counter |
| **C** | Cycle camera source |
| **O** | Rotate camera orientation (0° → 90° → 180° → 270° → 0°) |
| **SPACE** | Pause (video mode) |

## Architecture

```
src/
├── config.py       # Thresholds, classes, paths
├── utils.py        # Math utilities (angles, smoothing)
├── features.py     # 15-feature extraction (body-size invariant)
├── rules.py        # 6 hard-coded posture rules
├── model.py        # Conv1D neural network
├── train.py        # Training pipeline
├── analyzer.py     # Real-time analysis + UI overlay
└── main.py         # CLI entry point

models/
├── squat_model.keras   # Pre-trained model (97.4% accuracy)
└── squat_labels.json   # Class labels
```

## Posture Classes

| Class | Description |
|-------|-------------|
| ✅ CORRECT | Good squat form |
| ❌ FORWARD_LEAN | Torso leaning too far forward |
| ❌ HEEL_RISE | Heels lifting off the ground |
| ❌ KNEE_CAVE | Knees collapsing inward |
| ❌ KNEES_OVER_TOES | Knees extending past toes |
| ❌ SHALLOW_DEPTH | Not squatting deep enough |
| ❌ UNEVEN_WEIGHT | Asymmetric weight distribution |

## Training Your Own Model

1. Place squat videos in `SQUAT_VIDEOS/` with subfolders:
   ```
   SQUAT_VIDEOS/
   ├── ✅ CORRECT SQUAT/
   │   ├── squat_1.mp4
   │   └── squat_2.mp4
   └── ❌ WRONG SQUAT/
       ├── GEMINI [BODYWEIGHT]/
       │   ├── Knees Collapsing Inward.mp4
       │   └── Excessive Forward Lean.mp4
       └── YT Clips/
           └── ❌ Bending Forward From Your Back.mov
   ```

2. Run training:
   ```bash
   py -3.9 src/main.py --train
   ```

3. The model will be saved to `models/squat_model.keras`

## How It Works

1. **MediaPipe** detects 33 body landmarks in real-time
2. **15 body-size-invariant features** are extracted per frame (joint angles, distances, ratios)
3. **Hard-coded rules** check for obvious form errors using geometric relationships
4. **Conv1D ML model** classifies sliding windows of 15 frames for high-confidence detection
5. Both engines combine: rules provide instant feedback, ML catches subtle patterns

## License

MIT
