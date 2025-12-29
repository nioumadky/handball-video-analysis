"""
main.py
Point d’entrée du projet d’analyse vidéo handball (MVP stage)

Pipeline actuel :
- Chargement vidéo
- Détection des joueurs
- Tracking des joueurs (IDs persistants)
- Visualisation
"""

from pathlib import Path
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

    detector = PlayerDetector()
    tracker = CentroidTracker(max_distance=60)

    print("[INFO] Vidéo chargée avec succès.")
    print("[INFO] Appuie sur 'q' pour quitter.")

    # -------- Boucle principale --------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1️⃣ Détection joueurs
        detections = detector.detect(frame)

        # 2️⃣ Tracking joueurs (IDs)
        tracked_objects = tracker.update(detections)

        # 3️⃣ Visualisation
        for obj_id, (x, y, w, h) in tracked_objects.items():
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {obj_id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow("Handball Video Analysis - MVP", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # -------- Nettoyage --------
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Fin de la lecture vidéo.")


if __name__ == "__main__":
    main()
