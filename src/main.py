"""
main.py
Point d’entrée du projet d’analyse vidéo handball (MVP stage)

Pipeline actuel :
- Chargement vidéo
- Détection des joueurs
- Tracking des joueurs (IDs persistants)
- Trajectoires
- Calcul de la vitesse
- Interface de lecture vidéo (play/pause + barre de progression)
- Détection des tirs (V1)
- Filtrage des doublons de tirs
"""

from pathlib import Path
from collections import defaultdict
import math
import cv2

from video.video_loader import load_video
from detection.player_detector import PlayerDetector
from detection.tracking import CentroidTracker


def main():
    # -------- Configuration --------
    video_path = Path("data/raw/match_01.mp4")

    if not video_path.exists():
        raise FileNotFoundError(
            f"Vidéo introuvable : {video_path}\n"
            "Place une vidéo dans data/raw/ et mets à jour le nom."
        )

    # -------- Chargement vidéo --------
    cap = load_video(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    detector = PlayerDetector()
    tracker = CentroidTracker(max_distance=60, max_missing=10)

    # -------- Tracking & analyse --------
    trajectories = defaultdict(list)
    prev_positions = {}
    prev_speeds = {}

    shot_candidates = {}      # obj_id -> frame du pic
    shots_detected = []       # liste des tirs validés
    last_shot_frame = {}      # obj_id -> dernier tir (anti-doublon)

    MAX_TRAJECTORY_LENGTH = 40

    # -------- Paramètres tirs (MVP) --------
    SHOOTING_ZONE_X_MIN = int(frame_width * 0.6)

    SPEED_SHOT_THRESHOLD = 10.0
    SPEED_DROP_THRESHOLD = 5.0
    SHOT_WINDOW_FRAMES = 10
    SHOT_COOLDOWN_FRAMES = 30  # anti-doublon (~1s)

    # -------- Lecture vidéo --------
    paused = False
    current_frame_idx = 0

    WINDOW_NAME = "Handball Video Analysis - MVP"
    cv2.namedWindow(WINDOW_NAME)

    def on_trackbar(val):
        nonlocal current_frame_idx
        current_frame_idx = val
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)

    cv2.createTrackbar(
        "Progress",
        WINDOW_NAME,
        0,
        total_frames - 1,
        on_trackbar
    )

    print("[INFO] Vidéo chargée avec succès.")
    print("[INFO] Play/Pause : p | Quitter : q")

    # -------- Boucle principale --------
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_idx += 1
            cv2.setTrackbarPos("Progress", WINDOW_NAME, current_frame_idx)

        # 1️⃣ Détection joueurs
        detections = detector.detect(frame)

        # 2️⃣ Tracking joueurs (IDs)
        tracked_objects = tracker.update(detections)

        # 3️⃣ Analyse par joueur
        for obj_id, (x, y, w, h) in tracked_objects.items():
            cx = x + w // 2
            cy = y + h // 2

            # --- Calcul vitesse ---
            if obj_id in prev_positions:
                px, py = prev_positions[obj_id]
                speed = math.hypot(cx - px, cy - py)
            else:
                speed = 0.0

            prev_positions[obj_id] = (cx, cy)

            # -------- Détection du tir (V1 + anti-doublon) --------
            in_shooting_zone = cx > SHOOTING_ZONE_X_MIN

            # 1️⃣ Pic de vitesse
            if in_shooting_zone and speed > SPEED_SHOT_THRESHOLD:
                shot_candidates[obj_id] = current_frame_idx

            # 2️⃣ Chute après le pic → validation
            if (
                obj_id in shot_candidates and
                current_frame_idx - shot_candidates[obj_id] <= SHOT_WINDOW_FRAMES and
                speed < SPEED_DROP_THRESHOLD
            ):
                last_frame = last_shot_frame.get(obj_id, -9999)
                if current_frame_idx - last_frame > SHOT_COOLDOWN_FRAMES:
                    shots_detected.append((obj_id, current_frame_idx))
                    last_shot_frame[obj_id] = current_frame_idx

                    cv2.putText(
                        frame,
                        "SHOT",
                        (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3
                    )

                del shot_candidates[obj_id]

            prev_speeds[obj_id] = speed

            # --- Trajectoire ---
            trajectories[obj_id].append((cx, cy))
            if len(trajectories[obj_id]) > MAX_TRAJECTORY_LENGTH:
                trajectories[obj_id].pop(0)

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ID + vitesse
            cv2.putText(
                frame,
                f"ID {obj_id} v={speed:.1f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Trajectoire
            points = trajectories[obj_id]
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 2)

        # Compteur tirs
        cv2.putText(
            frame,
            f"Tirs detectes : {len(shots_detected)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused

    # -------- Nettoyage --------
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Fin de la lecture vidéo.")


if __name__ == "__main__":
    main()
