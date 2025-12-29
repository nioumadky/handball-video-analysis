"""
tracking.py
Tracking par centroid avec stabilisation des IDs

- tolérance aux frames manquées
- IDs plus stables
- adapté MVP stage
"""

import math


class CentroidTracker:
    def __init__(self, max_distance=60, max_missing=10):
        self.next_id = 0
        self.objects = {}         # id -> (cx, cy)
        self.missing = {}         # id -> nb de frames manquées
        self.max_distance = max_distance
        self.max_missing = max_missing

    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def update(self, detections):
        """
        detections: list of (x, y, w, h)
        returns: dict id -> (x, y, w, h)
        """

        updated = {}
        used_ids = set()

        # Calcul centres
        centers = [
            (x + w // 2, y + h // 2, (x, y, w, h))
            for (x, y, w, h) in detections
        ]

        # Association détections ↔ objets existants
        for cx, cy, bbox in centers:
            best_id = None
            best_dist = float("inf")

            for obj_id, (ox, oy) in self.objects.items():
                if obj_id in used_ids:
                    continue

                d = self._distance((cx, cy), (ox, oy))
                if d < best_dist and d < self.max_distance:
                    best_dist = d
                    best_id = obj_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.objects[best_id] = (cx, cy)
                self.missing[best_id] = 0
            else:
                self.objects[best_id] = (cx, cy)
                self.missing[best_id] = 0

            used_ids.add(best_id)
            updated[best_id] = bbox

        # Gestion des objets manqués
        for obj_id in list(self.objects.keys()):
            if obj_id not in used_ids:
                self.missing[obj_id] += 1
                if self.missing[obj_id] > self.max_missing:
                    del self.objects[obj_id]
                    del self.missing[obj_id]

        return updated
