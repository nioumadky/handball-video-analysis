"""
player_detector.py
Détection des joueurs – Version 2 (plus robuste)

Hypothèses :
- caméra fixe
- vue tribune
- joueurs en mouvement
"""

import cv2


class PlayerDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=800,
            varThreshold=40,
            detectShadows=True
        )

        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detect(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)

        # Suppression des ombres
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphologie pour nettoyer le masque
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = h / float(w + 1e-5)

            # Filtrage heuristique "joueur"
            if (
                area > 1200 and
                h > 40 and
                1.2 < aspect_ratio < 4.5
            ):
                detections.append((x, y, w, h))

        return detections
