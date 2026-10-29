"""
AI Footfall — Smart Customer Analytics
Flask Application & REST API Server
--------------------------------------
Serves real-time MJPEG live camera stream, analytics dashboard, SQLite persistence,
and CSV report exports.
"""

import time
import argparse
import base64
from flask import Flask, render_template, Response, jsonify, request, send_file
import io

from database import init_db, get_stats, get_reports_data, export_csv, reset_counts
from detector import VideoPipeline, get_local_ip

app = Flask(__name__)

# Initialize database on startup
init_db()

# Global video pipeline instance
pipeline = VideoPipeline(mode="simulation")


def generate_frames():
    """Generator function that yields MJPEG video stream frames."""
    while True:
        frame_bytes = pipeline.get_stream_frame()
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS throttle


@app.route('/')
def index():
    """Render the dashboard UI."""
    return render_template('index.html')


@app.route('/mobile')
def mobile():
    """Mobile phone camera streamer page."""
    return render_template('mobile.html', local_ip=get_local_ip())


@app.route('/api/ip')
def api_ip():
    """Return LAN IP and port for mobile camera pairing."""
    ip = get_local_ip()
    return jsonify({
        "ip": ip,
        "port": 5000,
        "mobile_url": f"http://{ip}:5000/mobile"
    })


@app.route('/api/upload_frame', methods=['POST'])
def api_upload_frame():
    """Receive live camera frames from mobile browser and process detection."""
    client_ip = request.remote_addr
    frame_bytes = None

    if 'frame' in request.files:
        frame_bytes = request.files['frame'].read()
    elif request.data:
        frame_bytes = request.data
    elif request.is_json and 'image' in request.json:
        b64_data = request.json['image']
        if ',' in b64_data:
            b64_data = b64_data.split(',')[1]
        frame_bytes = base64.b64decode(b64_data)

    if frame_bytes:
        success = pipeline.update_mobile_frame(frame_bytes, client_ip=client_ip)
        stats = get_stats()
        return jsonify({
            "status": "success" if success else "failed",
            "entered": stats["entered"],
            "exited": stats["exited"],
            "occupancy": stats["occupancy"]
        })
    return jsonify({"status": "error", "message": "No frame received"}), 400


@app.route('/video_feed')
def video_feed():
    """Live streaming route for the AI Camera feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def api_stats():
    """Provide real-time statistics for the frontend charts and counter cards."""
    data = get_stats()
    data['fps'] = pipeline.fps
    data['detected'] = pipeline.active_detections_count
    return jsonify(data)


@app.route('/api/pause', methods=['POST'])
def api_pause():
    """Toggle camera pause state."""
    is_paused = pipeline.toggle_pause()
    return jsonify({"paused": is_paused})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset counts."""
    reset_counts()
    return jsonify({"status": "success", "message": "Counts reset successfully"})


@app.route('/api/report')
def api_report():
    """Return JSON table rows for reports."""
    report_type = request.args.get('type', 'Daily Footfall Report')
    rows = get_reports_data(report_type)
    return jsonify(rows)


@app.route('/api/report.csv')
def api_report_csv():
    """Download reports as a CSV file."""
    report_type = request.args.get('type', 'Daily Footfall Report')
    csv_str = export_csv(report_type)
    output = io.BytesIO(csv_str.encode('utf-8'))
    filename = f"{report_type.lower().replace(' ', '_')}.csv"
    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )


@app.route('/api/mode', methods=['GET', 'POST'])
def api_mode():
    """Switch camera mode between simulation, mobile, webcam, and ipcam."""
    if request.method == 'POST':
        data = request.json or {}
        mode = data.get('mode', 'simulation')
        source = data.get('source', None)
        pipeline.set_mode(mode, source=source)
        return jsonify({"status": "success", "mode": pipeline.mode})
    return jsonify({"mode": pipeline.mode})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI Customer Footfall Analytics System")
    parser.add_argument('--mode', type=str, default='simulation', choices=['simulation', 'mobile', 'webcam', 'ipcam'],
                        help="Video source mode ('simulation', 'mobile', 'webcam', 'ipcam')")
    parser.add_argument('--camera', type=str, default='0', help="Webcam device index or IP camera stream URL")
    parser.add_argument('--port', type=int, default=5000, help="Web server port (default 5000)")
    args = parser.parse_args()

    pipeline.mode = args.mode
    if args.camera.isdigit():
        pipeline.video_source = int(args.camera)
    else:
        pipeline.video_source = args.camera

    if args.mode not in ('simulation', 'mobile'):
        pipeline.init_camera()

    local_ip = get_local_ip()

    print("=" * 68)
    print("  AI Footfall - Smart Customer Analytics System")
    print(f"  Web Dashboard:   http://localhost:{args.port}/")
    print(f"  Mobile Camera:   http://{local_ip}:{args.port}/mobile")
    print(f"  Active Mode:     {args.mode.upper()}")
    print("=" * 68)

    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
