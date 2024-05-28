import cv2
import numpy as np
import tensorflow as tf

# Chemin vers le dossier contenant le modèle SavedModel
model_dir = "server-ia/data/modeles/traffic_signs_model/1"

# Chargement du modèle SavedModel
model = tf.keras.models.load_model(model_dir)

# Fonction de prétraitement de l'image
def preprocess_image(image):
    # Redimensionner l'image à la taille attendue par le modèle
    processed_image = cv2.resize(image, (30, 30))
    # Assurez-vous que l'image est dans le format attendu par le modèle
    processed_image = processed_image.astype("float32") / 255.0  # Normalisation
    processed_image = np.expand_dims(processed_image, axis=0)  # Ajouter une dimension pour l'axe du lot
    return processed_image

# Fonction de détection des panneaux
def detect_signs(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Prétraitement de l'image
        processed_frame = preprocess_image(frame)
        
        # Faire une prédiction avec le modèle
        predictions = model.predict(processed_frame)
        
        # Analyser les prédictions et afficher les résultats sur l'image
        # Remplacez cette partie par votre code de détection et d'affichage des panneaux
        
        # Affichage de la vidéo avec les détections
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Appel de la fonction pour détecter les panneaux dans la vidéo
video_path = "server-ia/data/videos/autoroute.mp4"
detect_signs(video_path)
