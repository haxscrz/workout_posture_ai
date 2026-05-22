"""
main.py — Entry point for Squat Posture AI.

Usage:
    python src/main.py --train                Train ML model from SQUAT_VIDEOS/
    python src/main.py --camera               Real-time analysis via webcam
    python src/main.py --camera --source URL  Real-time via IP webcam
    python src/main.py --video PATH           Analyze a video file
"""

import argparse
import sys
import os

# Force standard output and standard error streams to UTF-8 with character replacement to prevent Windows encoding crashes.
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Squat Posture AI — Detect bad squat form using MediaPipe + ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --train
  python src/main.py --camera
  python src/main.py --camera --source "http://192.168.1.5:8080/video"
  python src/main.py --video "SQUAT_VIDEOS/CORRECT SQUAT/squat_11.mp4"
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true",
                       help="Train ML model from SQUAT_VIDEOS/")
    group.add_argument("--camera", action="store_true",
                       help="Real-time analysis via camera")
    group.add_argument("--video", type=str,
                       help="Analyze a video file")

    parser.add_argument("--source", type=str, default="0",
                        help="Camera source: index (0,1,..) or IP webcam URL")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                        help="Rotate camera feed by degrees (0, 90, 180, 270)")

    args = parser.parse_args()

    if args.train:
        from src.train import run_training
        run_training()

    elif args.camera:
        from src.analyzer import run_camera
        # Parse source: int for webcam index, str for URL
        source = args.source
        try:
            source = int(source)
        except ValueError:
            pass  # Keep as string (URL)
        run_camera(source, args.rotate)

    elif args.video:
        from src.analyzer import run_video
        run_video(args.video)


if __name__ == "__main__":
    main()
