import cv2
import mediapipe as mp
import csv
import os

# ------------------ CONFIGURATION ------------------
VIDEO_PATH = "data/raw_videos/test_match.mp4"  # Chemin vers ta vidéo test
OUTPUT_VIDEO_PATH = "data/processed_videos/test_match_annotated.mp4"
OUTPUT_CSV_PATH = "data/processed_videos/test_match_landmarks.csv"

# Création du dossier de sortie si nécessaire
os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# ------------------ VIDEO CAPTURE ------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Erreur : impossible d'ouvrir la vidéo {VIDEO_PATH}")
    exit()

# Récupérer les dimensions et FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Définir le writer pour sauvegarder la vidéo annotée
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# Préparer le CSV
csv_file = open(OUTPUT_CSV_PATH, mode='w', newline='')
csv_writer = csv.writer(csv_file)
header = ["frame", "landmark_id", "x", "y", "z", "visibility"]
csv_writer.writerow(header)

frame_idx = 0

# ------------------ BOUCLE PRINCIPALE ------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Conversion BGR -> RGB pour MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Annoter les landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Sauvegarder les coordonnées dans le CSV
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            csv_writer.writerow([frame_idx, i, landmark.x, landmark.y, landmark.z, landmark.visibility])

    # Afficher la frame annotée
    cv2.imshow('Tracking Handball', frame)
    out.write(frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# ------------------ NETTOYAGE ------------------
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
pose.close()

print(f"Vidéo annotée sauvegardée dans : {OUTPUT_VIDEO_PATH}")
print(f"CSV des landmarks sauvegardé dans : {OUTPUT_CSV_PATH}")
