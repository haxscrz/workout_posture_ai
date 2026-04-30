"""
generate_voices.py
──────────────────────────────────────────────────────────────────
Pre-synthesizes all gym coaching feedback phrases using Microsoft
Edge TTS (edge-tts) — the same neural engine behind Azure
Cognitive Services & Microsoft Edge Read Aloud.

Voice used: en-US-JennyNeural (warm, natural female coaching voice)
Alternative: en-US-GuyNeural, en-US-AriaNeural, en-US-DavisNeural

SETUP (one-time):
  pip install edge-tts          ← or: pip install -r requirements.txt
  python generate_voices.py

OUTPUT:
  ../assets/voices/<CODE>.mp3   ← pre-synthesized audio files
  ../assets/voices/manifest.json ← tells the browser what's available

WHY EDGE-TTS?
  - Neural voice quality identical to Azure Cognitive Services (~$16/1M chars)
  - Zero cost, no API key, works offline after generation
  - ~5MB install vs Coqui's ~2GB model download
  - No C++ build tools required (unlike Coqui TTS on Windows)
  - Runtime latency: 0ms (just plays an Audio element)
──────────────────────────────────────────────────────────────────
"""

import asyncio
import json
from pathlib import Path

VOICE      = "en-US-JennyNeural"   # Warm, natural coaching voice
OUTPUT_DIR = Path(__file__).parent.parent / "assets" / "voices"
MANIFEST   = OUTPUT_DIR / "manifest.json"

# ── All feedback phrases ────────────────────────────────────────────────────
PHRASES = {
    # ── Bicep Curl ───────────────────────────────────────────────────────────
    "BICEP_ELBOW_DRIFT":
        "Keep your elbow pinned to your side.",
    "BICEP_TORSO_SWAY":
        "Stop swinging. Control the lift.",
    "BICEP_INCOMPLETE_CURL":
        "Curl higher. Aim for a full range of motion.",
    "BICEP_INCOMPLETE_EXTENSION":
        "Fully extend your arm at the bottom.",
    "BICEP_WRIST_BREAK":
        "Keep your wrist straight.",

    # ── Push-Up ──────────────────────────────────────────────────────────────
    "PUSHUP_HIP_SAG":
        "Engage your core. Your hips are dropping.",
    "PUSHUP_HIP_PIKE":
        "Lower your hips. Don't pike up.",
    "PUSHUP_HEAD_DROOP":
        "Keep your head neutral. Look at the floor.",
    "PUSHUP_ELBOW_FLARE":
        "Tuck your elbows closer to your body.",
    "PUSHUP_PARTIAL_REP":
        "Go lower. Your chest should nearly touch the floor.",

    # ── Squat ────────────────────────────────────────────────────────────────
    "SQUAT_KNEE_CAVE_LEFT":
        "Push your left knee outward. It's caving in.",
    "SQUAT_KNEE_CAVE_RIGHT":
        "Push your right knee outward. It's caving in.",
    "SQUAT_BACK_ROUND":
        "Chest up. Your back is rounding.",
    "SQUAT_HEEL_RISE":
        "Keep your heels flat on the ground.",
    "SQUAT_INSUFFICIENT_DEPTH":
        "Go deeper. Aim for thighs parallel to the floor.",

    # ── Rep quality ──────────────────────────────────────────────────────────
    "REP_GOOD":  "Good rep!",
    "REP_FAIR":  "Fair rep.",
    "REP_POOR":  "Work on your form.",

    # ── Session start ────────────────────────────────────────────────────────
    "START_BICEP_CURL": "Starting Bicep Curls. Let's go!",
    "START_PUSHUP":     "Starting Push-Ups. Let's go!",
    "START_SQUAT":      "Starting Squats. Let's go!",

    # ── Milestone rep counts ─────────────────────────────────────────────────
    **{f"REPS_{n}": f"{n} reps! Keep it up!" for n in [5, 10, 15, 20, 25, 30]},

    # ── Visibility ───────────────────────────────────────────────────────────
    "VISIBILITY": "Make sure your full body is visible to the camera.",
}


async def generate_all():
    import edge_tts

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    print(f"Voice: {VOICE}")
    print(f"Output: {OUTPUT_DIR}\n")

    for code, text in PHRASES.items():
        out_path = OUTPUT_DIR / f"{code}.mp3"
        rel_path = f"assets/voices/{code}.mp3"

        if out_path.exists():
            print(f"  [skip] {code}")
        else:
            print(f"  [gen]  {code}: \"{text}\"")
            try:
                communicate = edge_tts.Communicate(text, VOICE)
                await communicate.save(str(out_path))
                print(f"         OK saved")
            except Exception as e:
                print(f"         ERROR: {e}")
                continue

        manifest[code] = rel_path

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! {len(manifest)} audio files generated.")
    print("  Restart the web app — realistic voice coaching is now active.")


if __name__ == "__main__":
    asyncio.run(generate_all())
