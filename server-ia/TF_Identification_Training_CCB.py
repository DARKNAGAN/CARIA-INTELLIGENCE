import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd

# Configuration
size = 60  # Taille des images à laquelle le modèle a été entraîné
model_path = "server-ia/data/modeles/tensorflow/tf_modele_speed_panneau.keras"
labels_csv_path = "server-ia/data/modeles/tensorflow/train_labels.csv"
confidence_threshold = 0.6  # Seuil de confiance pour considérer une prédiction comme valide

# Fonction pour charger le mapping des labels depuis le fichier CSV
def load_labels_mapping(labels_csv_path):
    df = pd.read_csv(labels_csv_path)
    unique_labels = df['label'].unique()
    labels_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"Mapping des labels chargé : {labels_mapping}")
    return labels_mapping

# Chargement du mapping des labels
labels_mapping = load_labels_mapping(labels_csv_path)

# Inversion du mapping pour obtenir l'étiquette à partir de l'index
reverse_labels_mapping = {v: k for k, v in labels_mapping.items()}

# Chargement du modèle
model = tf.keras.models.load_model(model_path)
print(f"Modèle chargé depuis {model_path}")

# Fonction pour prédire la catégorie d'une image, encadrer la zone reconnue et afficher les résultats
def predict_and_draw_bounding_box(image_path, model, labels_mapping, size, confidence_threshold):
    # Chargement de l'image originale
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Erreur de chargement de l'image : {image_path}")
    
    # Redimensionnement et normalisation pour la prédiction
    image_resized = cv2.resize(image, (size, size), cv2.INTER_LANCZOS4)
    image_normalized = image_resized.astype('float32') / 255.0  # Normalisation
    image_batch = np.expand_dims(image_normalized, axis=0)  # Ajout d'une dimension pour le batch

    # Prédiction
    predictions = model.predict(image_batch)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    # Récupération du label prédictif ou "Inconnu" si la confiance est faible
    if confidence < confidence_threshold:
        predicted_label = "Inconnu"
        print(f"Confiance faible ({confidence:.2f}). La prédiction n'est pas fiable.")
    else:
        predicted_label = reverse_labels_mapping[predicted_index]
    
    # Affichage de la prédiction dans la console
    print(f"Image: {image_path}, Prédiction: {predicted_label} (Confiance: {confidence:.2f})")

    # Encadrement de la zone reconnue (l'ensemble de l'image redimensionnée)
    height, width, _ = image.shape
    cv2.rectangle(image, (0, 0), (width, height), (0, 255, 0), 2)  # Couleur verte et épaisseur de 2

    # Ajout du texte de la prédiction sur l'image
    cv2.putText(image, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Affichage de l'image avec le cadre
    cv2.imshow("Image avec zone reconnue", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Exemple d'utilisation
image_to_classify = "server-ia/data/images/panneaux/France_road_sign_B14_(10).svg.png"  # Remplacez par le chemin exact de l'image à classifier

try:
    predict_and_draw_bounding_box(image_to_classify, model, labels_mapping, size, confidence_threshold)
except ValueError as e:
    print(e)
