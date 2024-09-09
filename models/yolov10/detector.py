import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
import streamlit as st

def inference(image_path, weight_path1, weight_path2):
    # Charger le premier modèle
    model = YOLO(weight_path1)
    pil_img = Image.open(image_path)

    # Effectuer la détection initiale
    result = model(source=pil_img, verbose=False)[0]
    result_img = result.plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # Obtenir les boîtes englobantes
    bboxes = result.boxes.xyxy.cpu().numpy()  # Format [x1, y1, x2, y2]
    x1, y1, x2, y2 = bboxes[0]
    
    # Recadrer l'image
    cropped_img = pil_img.crop((x1, y1, x2, y2))

    # Convertir l'image recadrée en format compatible pour le modèle
    cropped_img_np = np.array(cropped_img)
    cropped_img_pil = Image.fromarray(cropped_img_np)

    # Charger le second modèle et effectuer la détection sur l'image recadrée
    model_cropped = YOLO(weight_path2)
    result2 = model_cropped(source=cropped_img_pil, verbose=False)[0]
    result_img2 = result2.plot()
    result_img2 = cv2.cvtColor(result_img2, cv2.COLOR_BGR2RGB)

    # Obtenir les boîtes englobantes et les classes après recadrage
    bboxes_cropped = result2.boxes.xyxy.cpu().numpy()
    classes_cropped = result2.boxes.cls.cpu().numpy()

    # Trier les boîtes englobantes et les classes en fonction de la coordonnée x1 (de gauche à droite)
    sorted_indices = np.argsort(bboxes_cropped[:, 0])  # Trier par x1 (première colonne)
    sorted_bboxes = bboxes_cropped[sorted_indices]
    sorted_classes = classes_cropped[sorted_indices]

    # Afficher les classes triées de gauche à droite
    plate = []
    for i, bbox in enumerate(sorted_bboxes):
        x1, y1, x2, y2 = bbox
        detected_class = sorted_classes[i]
        plate.append(detected_class)
    
    st.markdown(f"Classes détectées après recadrage (de gauche à droite) : {plate}")
    print(f"Classes détectées après recadrage (de gauche à droite) : {plate}")

    return cropped_img_pil
