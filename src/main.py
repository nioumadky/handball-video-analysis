"""
main.py
Point d'entrée du projet d'analyse vidéo handball (MVP stage)

Pipeline actuel :
- Chargement vidéo
- Lecture frame par frame
- Affichage vidéo (validation pipeline)

Les étapes IA (détection, tracking, events) seront ajoutées progressivement.
"""

from pathlib import Path
import cv2

from video.video_loader import load_video


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

    print("[INFO] Vidéo chargée avec succès.")
    print("[INFO] Appuie sur 'q' pour quitter.")

    # -------- Boucle principale --------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # (Plus tard : détection joueurs ici)

        cv2.imshow("Handball Video Analysis - MVP", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # -------- Nettoyage --------
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Fin de la lecture vidéo.")


if __name__ == "__main__":
    main()
