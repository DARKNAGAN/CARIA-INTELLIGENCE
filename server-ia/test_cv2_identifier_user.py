#!/usr/bin/env python
import cv2
import pickle
import numpy as np
import time

# Configuration des tailles et couleurs
MIN_SIZE = 50
COLOR_INFO = (255, 255, 255)
COLOR_KO = (0, 0, 255)
COLOR_OK = (0, 255, 0)

# Facteur d'agrandissement pour la fenêtre d'affichage
SCALE_FACTOR = 1.3

# Chargement des classificateurs et des modèles
FACE_CASCADE_PATH = "server-ia/data/haarcascades/haarcascade_frontalface_alt2.xml"
MODEL_PATH = "server-ia/data/modeles/CV2/trainner.yml"
LABELS_PATH = "server-ia/data/modeles/CV2/labels.pickle"

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# Chargement des labels
with open(LABELS_PATH, "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# Initialisation de la capture vidéo
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.253.194:8081")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(MIN_SIZE, MIN_SIZE))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        
        if conf <= 95:
            color = COLOR_OK
            name = labels.get(id_, "Inconnu")
        else:
            color = COLOR_KO
            name = "Inconnu"
        
        label = f"{name} {conf:.2f}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, COLOR_INFO, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Afficher la résolution actuelle
    resolution_text = f"Resolution: {frame.shape[1]}x{frame.shape[0]}"
    cv2.putText(frame, resolution_text, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1, cv2.LINE_AA)
    
    # Calculer et afficher la netteté
    sharpness = calculate_sharpness(frame)
    sharpness_text = f"Sharpness: {sharpness:.2f}"
    cv2.putText(frame, sharpness_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1, cv2.LINE_AA)
    
    return frame

def main():
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la capture de la vidéo.")
            break
        
        frame = process_frame(frame)
        
        # Calculer et afficher le FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1, cv2.LINE_AA)
        
        # Redimensionner la frame pour l'affichage
        frame_resized = cv2.resize(frame, (int(frame.shape[1] * SCALE_FACTOR), int(frame.shape[0] * SCALE_FACTOR)))
        cv2.imshow('CARIA Project - Identification du client', frame_resized)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            # Capture 50 frames to avoid blocking
            for _ in range(50):
                ret, _ = cap.read()
                if not ret:
                    break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Fin")

if __name__ == "__main__":
    main()
