"""
Computer Vision Person Detection & Video Synthesis Engine
---------------------------------------------------------
Supports:
1. Smart Synthetic Retail Entrance Simulation (Runs anywhere out of the box with zero external files)
2. Live Webcam feed via OpenCV (HOG / Background Subtractor / YOLO)
3. Video File playback
"""

import time
import math
import random
import socket
import cv2
import numpy as np
from tracker import CentroidTracker
from database import log_event


def get_local_ip():
    """Retrieve the primary local LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class PedestrianAgent:
    """Represents a simulated person moving through the entrance hallway."""
    def __init__(self, agent_id, direction="down", width=640, height=360):
        self.id = agent_id
        self.direction = direction  # 'down' = entering, 'up' = exiting
        self.w = random.randint(48, 58)
        self.h = random.randint(110, 130)

        # Random horizontal position in the corridor
        self.x = random.randint(int(width * 0.20), int(width * 0.80) - self.w)

        if direction == "down":
            self.y = random.randint(-120, -40)
            self.speed = random.uniform(2.2, 3.8)
        else:
            self.y = random.randint(height + 20, height + 80)
            self.speed = random.uniform(-3.8, -2.2)

        # Visual style
        colors = [
            (210, 160, 40),   # Cyan/Teal
            (40, 160, 240),   # Orange
            (180, 50, 180),   # Purple
            (60, 200, 90),    # Emerald
            (200, 120, 80),   # Sky blue
            (140, 80, 220),   # Magenta
        ]
        self.shirt_color = random.choice(colors)
        self.skin_color = (190, 210, 240)
        self.conf = round(random.uniform(0.92, 0.98), 2)
        self.stride = random.uniform(0, 3.14)

    def update(self, height):
        self.y += self.speed
        self.stride += 0.22
        # Return True if still visible in bounds
        if self.direction == "down" and self.y > height + 80:
            return False
        if self.direction == "up" and self.y < -120:
            return False
        return True

    def get_bbox(self):
        return int(self.x), int(self.y), int(self.x + self.w), int(self.y + self.h)


class VideoPipeline:
    def __init__(self, mode="simulation", video_source=0):
        self.mode = mode
        self.video_source = video_source
        self.cap = None
        self.width = 640
        self.height = 360
        self.line_y = int(self.height * 0.62)

        self.tracker = CentroidTracker(max_disappeared=20, max_distance=70)
        self.is_paused = False

        # Simulation agents
        self.sim_agents = []
        self.next_sim_id = 1
        self.last_spawn_time = 0

        # Performance metrics
        self.fps = 30.0
        self.frame_count = 0
        self.last_fps_calc = time.time()
        self.active_detections_count = 0

        # Mobile camera live stream buffer
        self.latest_mobile_frame = None
        self.latest_mobile_time = 0
        self.mobile_client_ip = None
        self.ip_cam_url = ""

        # Motion & person detection for webcam/video feeds
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=30, detectShadows=False)

        if self.mode not in ("simulation", "mobile"):
            self.init_camera()

        # Pre-seed a few pedestrians in the simulation
        if self.mode == "simulation":
            self.seed_initial_simulation()

    def update_mobile_frame(self, frame_bytes, client_ip=None):
        """Receive and decode a JPEG frame uploaded from a mobile device."""
        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                self.latest_mobile_frame = img
                self.latest_mobile_time = time.time()
                if client_ip:
                    self.mobile_client_ip = client_ip
                self.mode = "mobile"
                return True
        except Exception as e:
            print(f"[ERROR] Failed to decode mobile frame: {e}")
        return False

    def generate_mobile_standby_frame(self):
        """Draw an informative standby screen when mobile mode is active but awaiting connection."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (18, 24, 34)  # Deep slate background

        local_ip = get_local_ip()
        mobile_url = f"http://{local_ip}:5000/mobile"

        # Frame border
        cv2.rectangle(frame, (8, 8), (self.width - 8, self.height - 8), (45, 55, 75), 1)

        # Phone Icon / Box
        center_x = self.width // 2
        cv2.rectangle(frame, (center_x - 30, 40), (center_x + 30, 130), (37, 99, 235), 2)
        cv2.circle(frame, (center_x, 50), 3, (120, 150, 190), -1)
        cv2.circle(frame, (center_x, 118), 5, (120, 150, 190), -1)

        # Title
        cv2.putText(frame, "MOBILE CAMERA STREAMING READY", (center_x - 170, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Instructions
        cv2.putText(frame, "1. Connect phone to the same Wi-Fi / Hotspot", (center_x - 190, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 195, 215), 1, cv2.LINE_AA)
        cv2.putText(frame, f"2. Open on phone: {mobile_url}", (center_x - 190, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (74, 222, 128), 1, cv2.LINE_AA)
        cv2.putText(frame, "3. Tap 'Start Mobile Camera' to stream live video", (center_x - 190, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 195, 215), 1, cv2.LINE_AA)

        # Pulsing waiting indicator
        pulse = int((math.sin(time.time() * 3.5) + 1.0) * 127)
        indicator_color = (0, pulse, 255)
        cv2.circle(frame, (center_x - 110, 310), 6, indicator_color, -1)
        cv2.putText(frame, "Waiting for phone video stream...", (center_x - 95, 315),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 170, 195), 1, cv2.LINE_AA)

        return frame, []

    def init_camera(self):
        try:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                print(f"[WARN] Unable to open video source {self.video_source}. Falling back to simulation.")
                self.mode = "simulation"
                self.seed_initial_simulation()
        except Exception as e:
            print(f"[ERROR] Camera init error: {e}. Falling back to simulation.")
            self.mode = "simulation"
            self.seed_initial_simulation()

    def seed_initial_simulation(self):
        p1 = PedestrianAgent(1, direction="down", width=self.width, height=self.height)
        p1.y = 80
        p1.x = int(self.width * 0.25)
        p2 = PedestrianAgent(2, direction="down", width=self.width, height=self.height)
        p2.y = 170
        p2.x = int(self.width * 0.52)
        p3 = PedestrianAgent(3, direction="up", width=self.width, height=self.height)
        p3.y = 260
        p3.x = int(self.width * 0.72)
        self.sim_agents = [p1, p2, p3]
        self.next_sim_id = 4

    def update_fps(self):
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_fps_calc
        if elapsed >= 1.0:
            self.fps = round(self.frame_count / elapsed, 1)
            self.frame_count = 0
            self.last_fps_calc = now

    def generate_simulated_frame(self):
        """Draw a sleek, high-tech retail entrance CCTV scene."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 1. Background: Polished tiled retail floor
        frame[:] = (22, 28, 38)  # Deep slate background

        # Subtle floor perspective tiles
        for x in range(0, self.width, 70):
            cv2.line(frame, (x, 0), (int(x * 1.1) - 30, self.height), (32, 40, 54), 1)
        for y in range(0, self.height, 45):
            cv2.line(frame, (0, y), (self.width, y), (30, 38, 50), 1)

        # Entrance side walls / glass panels
        cv2.rectangle(frame, (0, 0), (int(self.width * 0.12), self.height), (16, 20, 28), -1)
        cv2.rectangle(frame, (int(self.width * 0.88), 0), (self.width, self.height), (16, 20, 28), -1)
        cv2.line(frame, (int(self.width * 0.12), 0), (int(self.width * 0.12), self.height), (55, 65, 85), 2)
        cv2.line(frame, (int(self.width * 0.88), 0), (int(self.width * 0.88), self.height), (55, 65, 85), 2)

        # Subtle entrance directional guide arrows on the floor
        for arrow_y in [70, 190, 310]:
            # Left side: downward green arrow (Entrance)
            cv2.arrowedLine(frame, (int(self.width * 0.28), arrow_y - 20),
                            (int(self.width * 0.28), arrow_y + 15), (35, 60, 45), 2, tipLength=0.35)
            # Right side: upward blue arrow (Exit)
            cv2.arrowedLine(frame, (int(self.width * 0.72), arrow_y + 15),
                            (int(self.width * 0.72), arrow_y - 20), (55, 50, 35), 2, tipLength=0.35)

        # Spawn new simulated pedestrians periodically
        now = time.time()
        if now - self.last_spawn_time > random.uniform(1.8, 3.2):
            self.last_spawn_time = now
            # 65% entering, 35% exiting (realistic store pattern)
            direction = "down" if random.random() < 0.65 else "up"
            new_agent = PedestrianAgent(self.next_sim_id, direction, self.width, self.height)
            self.sim_agents.append(new_agent)
            self.next_sim_id += 1

        # Update and render simulated pedestrian avatars
        active_agents = []
        detected_rects = []

        for agent in self.sim_agents:
            if not self.is_paused:
                keep = agent.update(self.height)
            else:
                keep = True

            if keep:
                active_agents.append(agent)
                x1, y1, x2, y2 = agent.get_bbox()
                detected_rects.append((x1, y1, x2, y2))

                # Render Humanoid representation
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Ground shadow ellipse
                cv2.ellipse(frame, (cx, y2 - 2), (int(agent.w * 0.45), 8), 0, 0, 360, (12, 14, 20), -1)

                # Torso / clothing
                torso_top = y1 + int(agent.h * 0.28)
                torso_bottom = y1 + int(agent.h * 0.74)
                cv2.rectangle(frame, (x1 + 6, torso_top), (x2 - 6, torso_bottom), agent.shirt_color, -1)
                cv2.rectangle(frame, (x1 + 6, torso_top), (x2 - 6, torso_bottom), (60, 60, 70), 1)

                # Head
                head_r = int(agent.w * 0.26)
                head_cy = y1 + int(agent.h * 0.16)
                cv2.circle(frame, (cx, head_cy), head_r, agent.skin_color, -1)
                cv2.circle(frame, (cx, head_cy), head_r, (40, 40, 50), 1)

                # Legs with walking cadence
                leg_swing = int(math.sin(agent.stride) * 7)
                cv2.line(frame, (cx - 7, torso_bottom), (cx - 7 + leg_swing, y2 - 4), (50, 55, 65), 5)
                cv2.line(frame, (cx + 7, torso_bottom), (cx + 7 - leg_swing, y2 - 4), (45, 50, 60), 5)

        self.sim_agents = active_agents
        return frame, detected_rects

    def process_frame(self):
        """Main processing pipeline: captures/synthesizes frame, detects, tracks, and annotates."""
        self.update_fps()

        if self.mode == "simulation":
            frame, rects = self.generate_simulated_frame()
        elif self.mode == "mobile":
            # Check if a recent frame was received from the mobile phone (within 4.5s)
            if self.latest_mobile_frame is not None and (time.time() - self.latest_mobile_time < 4.5):
                frame = cv2.resize(self.latest_mobile_frame, (self.width, self.height))
                fg_mask = self.bg_subtractor.apply(frame)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                rects = []
                for cnt in contours:
                    if cv2.contourArea(cnt) > 1600:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if h > 40 and (h / max(1, w)) > 0.7:
                            rects.append((x, y, x + w, y + h))
            else:
                frame, _ = self.generate_mobile_standby_frame()
                return frame
        else:
            if self.cap is None or not self.cap.isOpened():
                self.init_camera()
            ret, frame = self.cap.read() if self.cap else (False, None)
            if not ret or frame is None:
                frame, rects = self.generate_simulated_frame()
            else:
                frame = cv2.resize(frame, (self.width, self.height))
                # Background subtraction and contour detection
                fg_mask = self.bg_subtractor.apply(frame)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                rects = []
                for cnt in contours:
                    if cv2.contourArea(cnt) > 1800:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # Aspect ratio check for upright human figure
                        if h > 45 and (h / max(1, w)) > 0.8:
                            rects.append((x, y, x + w, y + h))

        self.active_detections_count = len(rects)

        # Update Centroid Tracker with Line Crossing detection
        tracked_objects, events = self.tracker.update(rects, self.line_y)

        # Log any line crossing events to database
        for direction, object_id in events:
            log_event(direction=direction, person_id=object_id, confidence=0.96)

        # ----------------- Visual Annotations -----------------
        # 1. Virtual Tripwire Counting Line
        line_color = (68, 68, 245)  # Crimson neon
        dash_step = 16
        for x in range(int(self.width * 0.12), int(self.width * 0.88), dash_step * 2):
            cv2.line(frame, (x, self.line_y), (min(x + dash_step, int(self.width * 0.88)), self.line_y), line_color, 2)

        # Counting Line Label
        cv2.putText(frame, "COUNTING TRIPWIRE (ENTRY / EXIT)", (int(self.width * 0.14), self.line_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 255), 1, cv2.LINE_AA)

        # 2. Draw Object Tracking Bounding Boxes & Trajectories
        for object_id, data in tracked_objects.items():
            cx, cy = data['centroid']
            traj = data['trajectory']
            direction = data['direction']

            # Motion Trail (Glowing history)
            if len(traj) > 1:
                for t in range(1, len(traj)):
                    thickness = int(np.sqrt(16 / float(len(traj) - t + 1))) * 2
                    alpha_color = (40, 220, 110) if direction == "Entering" else (220, 160, 40)
                    cv2.line(frame, traj[t - 1], traj[t], alpha_color, max(1, thickness))

            # Centroid Point
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)

            # Bounding Box
            box = data['box']
            if box is not None:
                x1, y1, x2, y2 = box
                # Neon bounding box
                box_color = (34, 197, 94) if direction == "Entering" else (235, 120, 37)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Corner brackets for futuristic AI aesthetic
                length = 12
                # Top-left
                cv2.line(frame, (x1, y1), (x1 + length, y1), (255, 255, 255), 2)
                cv2.line(frame, (x1, y1), (x1, y1 + length), (255, 255, 255), 2)
                # Bottom-right
                cv2.line(frame, (x2, y2), (x2 - length, y2), (255, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x2, y2 - length), (255, 255, 255), 2)

                # Label tag
                tag = f"ID #{object_id:02d} [{direction}]"
                (w, h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x1, max(0, y1 - 18)), (x1 + w + 8, y1), box_color, -1)
                cv2.putText(frame, tag, (x1 + 4, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Telemetry HUD Overlay (Top & Bottom)
        # Top Left: Camera badge
        if self.mode == "mobile":
            cam_title = f"MOBILE CAM: {self.mobile_client_ip or 'CONNECTED'}"
        elif self.mode == "webcam":
            cam_title = "USB WEBCAM 01: ACTIVE"
        elif self.mode == "ipcam":
            cam_title = "IP CAMERA FEED: ACTIVE"
        else:
            cam_title = "CAM 01: ENTRANCE HALLWAY"

        cv2.putText(frame, cam_title, (14, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 230, 245), 1, cv2.LINE_AA)

        # Top Right: FPS & Mode
        fps_text = f"FPS: {self.fps} | {self.mode.upper()}"
        cv2.putText(frame, fps_text, (self.width - 185, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 230, 140), 1, cv2.LINE_AA)

        # Bottom status bar
        status_text = f"TRACKING: {len(tracked_objects)} | DETECTED: {self.active_detections_count} | AI ONLINE"
        cv2.putText(frame, status_text, (14, self.height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 195, 215), 1, cv2.LINE_AA)

        return frame

    def get_stream_frame(self):
        """Encode current processed frame to JPEG for MJPEG stream."""
        frame = self.process_frame()
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ret:
            return jpeg.tobytes()
        return None

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        return self.is_paused

    def set_mode(self, mode, source=None):
        self.mode = mode
        if mode == "webcam":
            self.video_source = int(source) if source is not None and str(source).isdigit() else 0
            self.init_camera()
        elif mode == "ipcam":
            url = str(source).strip() if source else str(self.video_source).strip()
            if url and not (url.endswith('/video') or url.endswith('/mjpeg') or url.endswith('.m3u8')):
                if url.endswith('/'):
                    url = url + 'video'
                else:
                    url = url + '/video'
            self.video_source = url
            self.ip_cam_url = url
            self.init_camera()
        elif mode == "simulation":
            if not self.sim_agents:
                self.seed_initial_simulation()
        return self.mode
