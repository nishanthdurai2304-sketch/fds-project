"""
Centroid Tracker and Movement Direction Analyzer for Customer Footfall Counting
--------------------------------------------------------------------------------
Tracks bounding boxes across consecutive video frames using Euclidean distance
matching on centroid coordinates. Detects when centroids cross the virtual
tripwire line to determine Entrance or Exit events.
"""

import math
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=60):
        """
        Initialize the tracker.
        :param max_disappeared: Number of consecutive frames an object can be missing before deregistering.
        :param max_distance: Max Euclidean pixel distance to associate a centroid with an existing ID.
        """
        self.next_object_id = 1
        self.objects = OrderedDict()       # object_id -> (cx, cy)
        self.disappeared = OrderedDict()   # object_id -> frame_count
        self.trajectories = OrderedDict()  # object_id -> list of (cx, cy)
        self.counted = OrderedDict()       # object_id -> {'entered': bool, 'exited': bool}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        """Register a newly detected object with a unique ID."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.trajectories[self.next_object_id] = [centroid]
        self.counted[self.next_object_id] = {'entered': False, 'exited': False}
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object from tracking."""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.trajectories:
            del self.trajectories[object_id]
        if object_id in self.counted:
            del self.counted[object_id]

    def update(self, rects, line_y):
        """
        Update tracker with current frame bounding boxes: rects = [(startX, startY, endX, endY), ...]
        Returns:
            tracked_objects: dict of object_id -> {
                'box': (x1, y1, x2, y2),
                'centroid': (cx, cy),
                'trajectory': [(cx, cy), ...],
                'direction': 'Entering' | 'Exiting' | 'Stationary'
            }
            events: list of ('entered' | 'exited', object_id)
        """
        events = []

        # If no bounding boxes detected in this frame
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}, events

        # Compute centroid for each bounding box
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        # If currently not tracking any objects, register all detections
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute Euclidean distance matrix between current objects and input detections
            # Shape: (len(object_centroids), len(input_centroids))
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # If distance exceeds threshold, do not associate
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                new_centroid = input_centroids[col]
                prev_centroid = self.objects[object_id]

                # Update object centroid and reset disappearance counter
                self.objects[object_id] = new_centroid
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append(new_centroid)
                if len(self.trajectories[object_id]) > 40:
                    self.trajectories[object_id].pop(0)

                # Check line crossing (tripwire)
                # Moving downwards: previous Y < line_y and current Y >= line_y -> ENTERED
                if prev_centroid[1] < line_y and new_centroid[1] >= line_y:
                    if not self.counted[object_id]['entered']:
                        self.counted[object_id]['entered'] = True
                        events.append(('entered', object_id))

                # Moving upwards: previous Y > line_y and current Y <= line_y -> EXITED
                elif prev_centroid[1] > line_y and new_centroid[1] <= line_y:
                    if not self.counted[object_id]['exited']:
                        self.counted[object_id]['exited'] = True
                        events.append(('exited', object_id))

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        # Prepare tracked objects result with movement vectors
        results = {}
        for object_id, centroid in self.objects.items():
            traj = self.trajectories[object_id]
            direction = "Stationary"
            if len(traj) >= 2:
                delta_y = traj[-1][1] - traj[0][1]
                if delta_y > 4:
                    direction = "Entering"
                elif delta_y < -4:
                    direction = "Exiting"

            # Match bounding box from rects if close
            box = None
            min_dist = float('inf')
            for r in rects:
                rcx = (r[0] + r[2]) / 2.0
                rcy = (r[1] + r[3]) / 2.0
                dist = math.hypot(centroid[0] - rcx, centroid[1] - rcy)
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    box = r

            results[object_id] = {
                'box': box,
                'centroid': centroid,
                'trajectory': traj,
                'direction': direction
            }

        return results, events
