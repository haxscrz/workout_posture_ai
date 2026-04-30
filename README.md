# 🏋️ FormAI — AI Gym Posture Coach

> Real-time AI-powered workout form correction using your device camera.  
> Supports **Squats**, **Bicep Curls**, and **Push-Ups** with instant voice feedback.

![License](https://img.shields.io/badge/license-MIT-blue)
![Node](https://img.shields.io/badge/node-%3E%3D18-green)
![Python](https://img.shields.io/badge/python-3.9%2B-yellow)

---

## ⚡ One-Click Setup (Windows)

**Double-click `START.bat`** — that's it!

The script will automatically:
1. ✅ Request administrator access (for firewall rules)
2. ✅ Install Node.js and Python dependencies
3. ✅ Open Windows Firewall ports so your **phone can connect**
4. ✅ Start both the App Server and the Training Server
5. ✅ Open the app in your default browser

> **📱 Phone Access:** After running the script, it will display your local IP address.  
> On your phone's browser, navigate to `https://<YOUR_IP>:5173`  
> Accept the self-signed certificate warning to proceed.

---

## 📋 Prerequisites

| Requirement | Version | Download |
|-------------|---------|----------|
| **Node.js** | 18+ (LTS recommended) | [nodejs.org](https://nodejs.org/) |
| **Python** | 3.9+ (for ML training only) | [python.org](https://www.python.org/downloads/) |

> Python is **optional** — you only need it if you want to train your own ML models.  
> The app ships with a pre-trained squat model that works out of the box.

---

## 🛠️ Manual Setup (Advanced)

If you prefer not to use the batch file:

```bash
# 1. Clone the repository
git clone https://github.com/haxscrz/workout_posture_ai.git
cd workout_posture_ai

# 2. Install Node.js dependencies
npm install

# 3. Start the app (HTTPS required for camera access)
npm run dev

# 4. Open in browser
# → https://localhost:5173
```

### Mobile Access (Manual)

```bash
# The Vite config already binds to 0.0.0.0
# Allow through Windows Firewall (run as admin):
netsh advfirewall firewall add rule name="FormAI" dir=in action=allow protocol=TCP localport=5173

# Find your local IP:
ipconfig
# → Connect from phone: https://<YOUR_IP>:5173
```

---

## 🎯 How It Works

FormAI uses a **hybrid AI system** for maximum accuracy:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Pose Detection** | MediaPipe Pose (Browser) | Extracts 33 body landmarks in real-time |
| **Rule-Based Engine** | JavaScript Math | Counts reps, measures angles, checks depth |
| **ML Classifier** | TensorFlow.js (Conv1D) | Detects subtle form issues (e.g., knees caving) |
| **Voice Coach** | Web Speech API / Custom TTS | Speaks corrections in real-time |

### Exercises Supported

| Exercise | Camera Position | View | Form Checks |
|----------|----------------|------|-------------|
| 🏋️ Squat | Floor, 1-1.5m in front | Front | Knee cave, back rounding, depth, heel rise |
| 💪 Bicep Curl | Floor in front | Front | Elbow drift, torso sway |
| 🤸 Push-Up | Floor to your side | Profile | Elbow flare, hip sag |

### Video Analysis Mode

You can also upload a pre-recorded video for analysis:
1. Select an exercise on the home screen
2. On the setup screen, click **"Upload Test Video"**
3. Select your `.mp4` or `.webm` file
4. The AI will analyze it exactly like a live camera feed

---

## 🧠 Training Your Own ML Model (Optional)

The app ships with a pre-trained squat model. To train models on your own videos:

### Step 1: Record Training Videos

Record 2-3 short clips (~30 seconds each) per form category:

**Squat Example:**
- ✅ `GOOD_FORM/` — Perfect squats with proper depth
- ❌ `KNEES_CAVING/` — Knees collapsing inward
- ❌ `BACK_ROUNDING/` — Excessive forward lean
- ❌ `SHALLOW_DEPTH/` — Not squatting deep enough
- ❌ `HEELS_RISING/` — Heels lifting off the ground

### Step 2: Organize Videos

Place videos into the pre-created folder structure:

```
training-server/data/uploads/
├── squat/
│   ├── GOOD_FORM/        ← drop good squat videos here
│   ├── KNEES_CAVING/     ← drop knee cave videos here
│   ├── BACK_ROUNDING/
│   ├── SHALLOW_DEPTH/
│   └── HEELS_RISING/
├── bicep-curl/
│   ├── GOOD_FORM/
│   ├── ELBOW_DRIFT/
│   └── TORSO_SWAY/
└── pushup/
    ├── GOOD_FORM/
    ├── ELBOW_FLARE/
    └── HIP_SAG/
```

### Step 3: Train

**Option A — Via Dashboard UI (recommended):**
```bash
# Start the training server
cd training-server
pip install -r requirements.txt
python server.py

# Then click "🔬 Training Dashboard" in the app
# Upload videos and click "Start Training"
```

**Option B — Via CLI:**
```bash
cd training-server
pip install -r requirements.txt
python train.py
```

Models auto-export to `public/models/<exercise>/` — the app picks them up on next page load.

### Tips for Better Accuracy

- **One mistake per video** — don't mix errors in one clip
- **Vary conditions** — different clothing, lighting, angles
- **At least 2 classes** — you need "good" + at least one "bad"
- **More data = better** — 3+ videos per category is ideal
- **Mirror augmentation** — the pipeline automatically flips videos to double your dataset

---

## 🏗️ Architecture

```
Web App (Browser)                     Python Training Server
┌─────────────────────────┐          ┌─────────────────────────┐
│  MediaPipe Pose (WASM)  │          │  Flask API (port 5000)  │
│  ↓                      │          │  ↓                      │
│  Rule-Based Analyzers   │          │  Video → MediaPipe      │
│  (exercises/*.js)       │          │  → Feature Extraction   │
│  ↓                      │          │  → Conv1D Training      │
│  ML Inference           │ ← model  │  → TF.js Export         │
│  (ml/ml-inference.js)   │ ← labels │                         │
│  ↓                      │          └─────────────────────────┘
│  UI + Voice Feedback    │
└─────────────────────────┘
```

## 📁 Project Structure

```
├── START.bat                      # One-click setup & launch script
├── app.js                         # Main app controller & detection loop
├── pose-engine.js                 # Angle math, smoothing, body measurements
├── setup-wizard.js                # Pre-workout camera positioning wizard
├── training-dashboard.js          # Dashboard UI for ML training
├── index.html                     # Main HTML entry point
├── vite.config.js                 # Vite dev server config (HTTPS + LAN)
│
├── exercises/                     # Rule-based exercise analyzers
│   ├── ExerciseBase.js            #   Base class (rep counting, scoring)
│   ├── squat.js                   #   Squat analyzer (hip-depth state machine)
│   ├── bicep-curl.js              #   Bicep curl analyzer
│   └── pushup.js                  #   Push-up analyzer
│
├── ml/                            # Browser-side ML inference
│   └── ml-inference.js            #   TF.js model loading & prediction
│
├── ui/                            # UI components
│   ├── skeleton.js                #   Canvas pose skeleton renderer
│   └── feedback.js                #   Feedback display + TTS voice coach
│
├── public/models/                 # Pre-trained TF.js models
│   └── squat/                     #   Squat model (ships with repo)
│       ├── model.json
│       ├── group1-shard1of1.bin
│       └── labels.json
│
├── training-server/               # Python ML training backend
│   ├── server.py                  #   Flask API server
│   ├── train.py                   #   CLI training script
│   ├── requirements.txt           #   Python dependencies
│   ├── data/uploads/              #   Training video folders
│   └── pipeline/
│       ├── feature_extractor.py   #   Mirror-invariant feature extraction
│       ├── video_processor.py     #   Video → sliding windows
│       ├── model.py               #   Conv1D model architecture
│       └── trainer.py             #   Training loop + TF.js export
│
├── styles/                        # CSS stylesheets
│   └── main.css                   #   Full design system
│
└── tts-server/                    # Optional realistic voice generation
    └── generate_voices.py
```

---

## 🔒 Security Notes

- The app uses a **self-signed HTTPS certificate** (generated by Vite) because camera access requires HTTPS on mobile browsers
- Your browser will show a certificate warning — this is expected and safe for local use
- The firewall rules created by `START.bat` are named `FormAI-Vite` and `FormAI-Training` for easy identification and removal
- **No data leaves your computer** — all AI processing runs locally in your browser

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
