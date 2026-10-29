"""
Automated System Verification Script for AI Footfall Counter
"""

import sys
import os

# Add local path
sys.path.insert(0, os.path.dirname(__file__))

from database import init_db, get_stats, get_reports_data, export_csv
from tracker import CentroidTracker
from detector import VideoPipeline
from app import app


def test_tracker():
    print("Testing CentroidTracker...")
    tracker = CentroidTracker()
    line_y = 100

    # Person 1 moving down (Crossing line: from 80 to 120)
    rects1 = [(40, 60, 80, 100)]  # centroid y = 80
    objs1, events1 = tracker.update(rects1, line_y)
    assert len(objs1) == 1, "Failed to register object"

    rects2 = [(40, 100, 80, 140)]  # centroid y = 120
    objs2, events2 = tracker.update(rects2, line_y)
    assert len(events2) == 1, f"Expected 1 line crossing event, got {events2}"
    assert events2[0][0] == 'entered', f"Expected 'entered', got {events2[0][0]}"
    print("  [OK] CentroidTracker & line crossing verified.")


def test_database():
    print("Testing Database...")
    init_db()
    stats = get_stats()
    assert 'entered' in stats and 'exited' in stats and 'occupancy' in stats
    assert stats['entered'] >= 0
    print(f"  [OK] Database initialized. Current Entered: {stats['entered']}, Exited: {stats['exited']}")

    reports = get_reports_data("Daily Footfall Report")
    assert len(reports) > 0, "No report rows generated"
    print(f"  [OK] Reports generated: {len(reports)} rows")

    csv_out = export_csv("Daily Footfall Report")
    assert "Period,Entered,Exited,Net,Status" in csv_out
    print("  [OK] CSV export verified.")


def test_detector():
    print("Testing VideoPipeline & Stream Generation...")
    pipeline = VideoPipeline(mode="simulation")
    frame = pipeline.process_frame()
    assert frame is not None and frame.shape == (360, 640, 3)
    stream_bytes = pipeline.get_stream_frame()
    assert stream_bytes is not None and len(stream_bytes) > 1000
    print(f"  [OK] Frame generation & JPEG compression verified ({len(stream_bytes)} bytes/frame).")


def test_flask_api():
    print("Testing Flask Web Endpoints...")
    client = app.test_client()

    r = client.get('/')
    assert r.status_code == 200, f"Dashboard index failed: {r.status_code}"
    print("  [OK] GET / => 200 OK")

    r = client.get('/api/stats')
    assert r.status_code == 200
    json_data = r.get_json()
    assert 'entered' in json_data and 'fps' in json_data
    print(f"  [OK] GET /api/stats => 200 OK (FPS: {json_data['fps']}, Occupancy: {json_data['occupancy']})")

    r = client.post('/api/pause')
    assert r.status_code == 200
    print(f"  [OK] POST /api/pause => 200 OK ({r.get_json()})")

    r = client.get('/api/report?type=Daily%20Footfall%20Report')
    assert r.status_code == 200
    print(f"  [OK] GET /api/report => 200 OK ({len(r.get_json())} rows)")

    r = client.get('/api/report.csv?type=Daily%20Footfall%20Report')
    assert r.status_code == 200
    print(f"  [OK] GET /api/report.csv => 200 OK (Content-Type: {r.content_type})")

    r = client.get('/mobile')
    assert r.status_code == 200
    print("  [OK] GET /mobile => 200 OK (Mobile camera streamer UI)")

    r = client.get('/api/ip')
    assert r.status_code == 200
    ip_data = r.get_json()
    assert 'ip' in ip_data and 'mobile_url' in ip_data
    print(f"  [OK] GET /api/ip => 200 OK (LAN IP: {ip_data['ip']}, Mobile URL: {ip_data['mobile_url']})")

    # Test uploading a simulated JPEG frame from a phone
    import cv2
    import numpy as np
    mock_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    _, mock_jpeg = cv2.imencode('.jpg', mock_frame)
    r = client.post('/api/upload_frame', data=mock_jpeg.tobytes(), content_type='image/jpeg')
    assert r.status_code == 200
    print(f"  [OK] POST /api/upload_frame => 200 OK ({r.get_json()})")


if __name__ == '__main__':
    print("==================================================")
    print("  RUNNING COMPLETE VERIFICATION SUITE")
    print("==================================================")
    test_tracker()
    test_database()
    test_detector()
    test_flask_api()
    print("==================================================")
    print("  SUCCESS: ALL VERIFICATION TESTS PASSED!")
    print("==================================================")
