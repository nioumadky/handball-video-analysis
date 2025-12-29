"""
tracking.py
Tracking simple par distance (centroid tracking)

Attribue des IDs persistants aux détections successives.
"""

import math


class CentroidTracker:
    def __init__(self, max_distance=60):
        self.next_id = 0
        self.objects = {}  # id -> (cx, cy)
        self.max_distance = max_distance

    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def update(self, detections):
        """
        detections: list of (x, y, w, h)
        returns: dict id -> (x, y, w, h)
        """

        updated_objects = {}

        # Calcul des centres des détections
        centers = [
            (x + w // 2, y + h // 2, (x, y, w, h))
            for (x, y, w, h) in detections
        ]

        # Associer aux objets existants
        for cx, cy, bbox in centers:
            best_id = None
            best_dist = float("inf")

            for obj_id, (ox, oy) in self.objects.items():
                d = self._distance((cx, cy), (ox, oy))
                if d < best_dist and d < self.max_distance:
                    best_dist = d
                    best_id = obj_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            updated_objects[best_id] = (cx, cy)
            self.objects[best_id] = (cx, cy)

        return {
            obj_id: bbox
            for obj_id, (_, _, bbox) in zip(
                updated_objects.keys(), centers
            )
        }
