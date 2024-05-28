import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Charger le modèle sauvegardé
model = load_model("server-ia/data/modeles/RoadSign/modele_signaux_routiers.h5")

# Fonction pour charger les noms de classe à partir d'un fichier
def load_class_names(file_path):
    with open(file_path, 'r') as file:
        class_names = [line.strip() for line in file.readlines()]
    return class_names

# Définir le chemin du fichier contenant les noms de classe
class_names_file = "server-ia/data/modeles/RoadSign/class_names.txt"

# Charger les noms de classe à partir du fichier
class_names = load_class_names(class_names_file)

# Fonction pour prétraiter l'image
def preprocess_image(image):
    # Mettre à l'échelle l'image aux dimensions attendues par le modèle
    scaled_image = cv2.resize(image, (224, 224))
    scaled_image = scaled_image.astype("float") / 255.0
    
    # Ajouter une dimension pour correspondre à la forme d'entrée du modèle
    preprocessed_img = np.expand_dims(scaled_image, axis=0)
    
    return preprocessed_img

# Fonction pour obtenir les coordonnées de la boîte englobante
def obtenir_coordonnees_boite(predictions):
    # A DEFINIR
    # Supposons que predictions est une liste de [x_min, y_min, x_max, y_max] pour chaque boîte englobante
    return predictions[0]  # Retourne les coordonnées de la première boîte englobante

# Fonction pour détecter les panneaux de signalisation dans une image
def detect_sign(image):
    # Prétraiter l'image
    preprocessed_img = preprocess_image(image)
    
    # Faire une prédiction avec le modèle
    predictions = model.predict(preprocessed_img)
    
    # Obtenir l'indice de la classe prédite
    predicted_class_index = np.argmax(predictions)
    
    # Récupérer le nom de la classe prédite
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name, predictions

# Définir la source vidéo ou la caméra
# Pour une vidéo
video_path = "server-ia/data/videos/autoroute.mp4"
cap = cv2.VideoCapture(video_path)

# Pour la caméra
# cap = cv2.VideoCapture(0)
# Nombre de trames à sauter entre chaque trame lue
skip_frames = 5

# Boucle pour lire les images de la vidéo ou de la caméra
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Incrémenter le compteur de trames
    frame_count += 1
    
    # Si le compteur de trames est un multiple de skip_frames, traiter la trame et afficher
    if frame_count % skip_frames == 0:
        # Réinitialiser le compteur de trames
        frame_count = 0
        
        # Détecter les panneaux de signalisation dans l'image
        predicted_class_name, predictions = detect_sign(frame)
        
        # Obtenir les coordonnées de la boîte englobante de l'objet détecté
        x_min, y_min, x_max, y_max = obtenir_coordonnees_boite(predictions)
        
        # Dessiner un carré autour de chaque objet détecté et afficher le nom de la classe au-dessus
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, predicted_class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Afficher l'image
        cv2.imshow('Frame', frame)
        
        # Attendre la touche 'q' pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libérer la capture et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
