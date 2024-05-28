import cv2
import os
import numpy as np
import random
# Tuto25[OpenCV] Lecture des panneaux de vitesse p.1 (houghcircles) 16min

size=42
video_dir = "server-ia/data/videos"

# Liste tous les fichiers dans le répertoire vidéo
l = os.listdir(video_dir)

for video in l:
    if not video.endswith("mp4"):
        continue
    cap = cv2.VideoCapture(video_dir + "/" + video)

    print("video:", video)
    while True:
        # Capture une frame de la vidéo
        ret, frame = cap.read()
        if ret is False:
            break
        
        # Redimensionne la frame pour un affichage correct
        f_w, f_h, f_c = frame.shape
        frame = cv2.resize(frame, (int(f_h / 1.5), int(f_w / 1.5)))

        # Extrait une région d'intérêt (ROI) de la frame
        image = frame[200:400, 700:1000]

        # represents the top left corner of rectangle 
        start_point = (600, 50)
        # Ending coordinate
        # represents t
        # he bottom right corner of rectangle 
        end_point = (800, 450)
        # Color in BGR 
        color = (255, 255, 255)
        # Line thickness
        thickness = 1

        # Dessine un rectangle autour de la ROI (dimensions spécifiées)
        cv2.rectangle(frame, start_point, end_point, color, thickness) 

        # Convertit l'image en niveaux de gris pour la détection des cercles
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Détection des cercles dans l'image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=60, minRadius=5, maxRadius=45)
        if circles is not None:
            circles = np.int16(np.around(circles))
            for i in circles[0, :]:
                if i[2] != 0:
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 4)
                    # Extrait le panneau de signalisation à partir de la position et du rayon du cercle
                    panneau = cv2.resize(
                        image[max(0, i[1] - i[2]):i[1] + i[2], max(0, i[0] - i[2]):i[0] + i[2]],
                        (size, size)) / 255
                    cv2.imshow("panneau", panneau)

        # Affiche le nom du fichier vidéo en cours
        cv2.putText(frame, "fichier:" + video, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        # Affiche la frame avec le rectangle et le panneau détecté
        cv2.imshow("Video", frame)

        # Attend l'appui d'une touche et traite les actions associées
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            quit()
        if key == ord('a'):
            for cpt in range(100):
                ret, frame = cap.read()
        if key == ord('f'):
            break

# Ferme toutes les fenêtres OpenCV
cv2.destroyAllWindows()
