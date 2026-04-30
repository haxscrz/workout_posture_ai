"""
server.py — Flask API for the Training Dashboard

Endpoints:
  POST /api/upload     — Upload a labeled training video
  GET  /api/data       — Get summary of uploaded training data
  POST /api/train      — Start model training (async)
  GET  /api/stream     — SSE stream of training telemetry

The training pipeline (TensorFlow, MediaPipe) is lazy-loaded only when
training is triggered, so the server can start without heavy ML dependencies.
"""

import os
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'uploads')
# Export models directly into the Vite app's public folder
EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'public', 'models')

# Lazy-loaded training manager
_manager = None


def get_manager():
    """Lazy-load the training manager to avoid requiring TF at startup."""
    global _manager
    if _manager is None:
        from pipeline.trainer import TrainingManager
        _manager = TrainingManager(DATA_DIR, EXPORT_DIR)
    return _manager


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload a labeled training video."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    exercise = request.form.get('exercise', 'squat')
    label = request.form.get('label')
    if not label:
        return jsonify({'error': 'No label provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    save_dir = os.path.join(DATA_DIR, exercise, label)
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, filename)
    file.save(file_path)

    # Get file size for feedback
    file_size = os.path.getsize(file_path)
    size_mb = file_size / (1024 * 1024)

    return jsonify({
        'success': True,
        'path': file_path,
        'exercise': exercise,
        'label': label,
        'size_mb': round(size_mb, 1),
    })


@app.route('/api/data', methods=['GET'])
def get_data_summary():
    """Return a summary of all uploaded training data."""
    summary = {}
    if os.path.exists(DATA_DIR):
        for exercise in sorted(os.listdir(DATA_DIR)):
            ex_dir = os.path.join(DATA_DIR, exercise)
            if os.path.isdir(ex_dir):
                summary[exercise] = {}
                for label in sorted(os.listdir(ex_dir)):
                    label_dir = os.path.join(ex_dir, label)
                    if os.path.isdir(label_dir):
                        videos = [
                            f for f in os.listdir(label_dir)
                            if f.lower().endswith(('.mp4', '.mov', '.webm', '.avi', '.mkv'))
                        ]
                        if videos:
                            summary[exercise][label] = len(videos)
    return jsonify(summary)


@app.route('/api/train', methods=['POST'])
def start_training():
    """Start model training in a background thread."""
    try:
        manager = get_manager()
    except ImportError as e:
        return jsonify({
            'success': False,
            'message': f'Missing dependencies: {e}. Run: pip install tensorflow mediapipe opencv-python tensorflowjs'
        }), 500

    if manager.start_training():
        return jsonify({'success': True, 'message': 'Training started'})
    return jsonify({'success': False, 'message': 'Training already in progress'}), 400


@app.route('/api/stream')
def stream_telemetry():
    """SSE endpoint for live training telemetry."""
    try:
        manager = get_manager()
    except ImportError:
        import json
        def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'TensorFlow not installed. Run: pip install tensorflow mediapipe opencv-python tensorflowjs'})}\n\n"
        return Response(error_stream(), mimetype='text/event-stream')

    return Response(
        manager.get_messages(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


if __name__ == '__main__':
    import sys
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    print("=" * 60)
    print("  FormAI Training Server")
    print("=" * 60)
    print(f"  Python:      {sys.executable} ({sys.version.split()[0]})")
    print(f"  Data Dir:    {os.path.abspath(DATA_DIR)}")
    print(f"  Export Dir:  {os.path.abspath(EXPORT_DIR)}")
    print(f"  Dashboard:   http://localhost:5000")
    print()

    # Eagerly check training dependencies
    try:
        from pipeline.trainer import TrainingManager
        _manager = TrainingManager(DATA_DIR, EXPORT_DIR)
        print("  [OK] TensorFlow loaded - training is available")
    except ImportError as e:
        print(f"  [WARNING] Training unavailable: {e}")
        print("    Run: pip install tensorflow mediapipe opencv-python tensorflowjs")

    print()
    print("  To upload training videos, use the web dashboard")
    print("  or drop files directly into the data directory:")
    print(f"    {os.path.abspath(DATA_DIR)}/<exercise>/<label>/video.mp4")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
