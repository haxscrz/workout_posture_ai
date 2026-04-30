#!/usr/bin/env python3
"""
train.py — Single-command training script

Usage:
    python train.py
    python train.py --data-dir ./custom_data --epochs 100

Directory structure expected:
    data/uploads/
    ├── squat/
    │   ├── GOOD_FORM/
    │   │   ├── video1.mp4
    │   │   └── video2.mp4
    │   ├── KNEES_CAVING/
    │   │   └── vid3.mp4
    │   └── BACK_ROUNDING/
    │       └── vid4.mp4
    ├── bicep-curl/
    │   ├── GOOD_FORM/
    │   └── ELBOW_DRIFT/
    └── pushup/
        ├── GOOD_FORM/
        └── HIP_SAG/
"""

import os
import sys
import argparse
import json
import queue

# Add parent dir to path for pipeline imports
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.trainer import train_model_thread


def main():
    parser = argparse.ArgumentParser(description='Train FormAI posture models')
    parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), 'data', 'uploads'),
                        help='Directory containing labeled training videos')
    parser.add_argument('--export-dir', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'public', 'models'),
                        help='Directory to export TF.js models to')
    parser.add_argument('--epochs', type=int, default=80, help='Max training epochs')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    export_dir = os.path.abspath(args.export_dir)

    print("=" * 60)
    print("  FormAI — Model Training")
    print("=" * 60)
    print(f"  Data Dir:   {data_dir}")
    print(f"  Export Dir:  {export_dir}")
    print(f"  Max Epochs:  {args.epochs}")
    print("=" * 60)
    print()

    # Check data dir exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        print(f"Create the directory and add training videos:")
        print(f"  {data_dir}/<exercise>/<label>/video.mp4")
        sys.exit(1)

    # List what's available
    exercises = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not exercises:
        print("ERROR: No exercise directories found.")
        print(f"Add directories like: {data_dir}/squat/GOOD_FORM/video.mp4")
        sys.exit(1)

    print("Found exercises:")
    for ex in exercises:
        ex_dir = os.path.join(data_dir, ex)
        labels = [d for d in os.listdir(ex_dir) if os.path.isdir(os.path.join(ex_dir, d))]
        for label in labels:
            label_dir = os.path.join(ex_dir, label)
            vids = [f for f in os.listdir(label_dir) if f.lower().endswith(('.mp4', '.mov', '.webm', '.avi'))]
            print(f"  {ex}/{label}: {len(vids)} video(s)")
    print()

    os.makedirs(export_dir, exist_ok=True)

    # Use a simple message queue to print messages
    msg_queue = queue.Queue()

    # Run training in current thread (blocking)
    import threading

    def run():
        train_model_thread(data_dir, export_dir, msg_queue, epochs=args.epochs)

    thread = threading.Thread(target=run)
    thread.start()

    # Print messages as they come
    while thread.is_alive() or not msg_queue.empty():
        try:
            msg = msg_queue.get(timeout=0.5)
            if msg['type'] == 'log':
                print(msg['message'])
            elif msg['type'] == 'epoch':
                e = msg['epoch']
                total = msg['total_epochs']
                loss = msg['loss']
                acc = msg['accuracy']
                val_loss = msg['val_loss']
                val_acc = msg['val_accuracy']
                # Progress bar
                bar_len = 30
                filled = int(bar_len * e / total)
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f"  [{bar}] Epoch {e:3d}/{total} | loss: {loss:.4f} | acc: {acc:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")
            elif msg['type'] == 'complete':
                print(f"\n✓ {msg['message']}")
            elif msg['type'] == 'error':
                print(f"\n✗ {msg['message']}")
        except queue.Empty:
            pass

    thread.join()
    print("\nDone! Trained models are ready in:", export_dir)


if __name__ == '__main__':
    main()
