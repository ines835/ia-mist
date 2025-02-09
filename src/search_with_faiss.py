from io import BytesIO
import os
import numpy as np
from PIL import Image
import faiss
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from typing import List
from .extract_embedding import extract_embeddings, extract_embeddings_for_search
import sys

# Ajouter le dossier `src` au chemin
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Racine du projet IA-MIST
print(BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")  # Dossier `src`
sys.path.append(SRC_DIR)  # Ajoute `src` au chemin Python

DATASET_DIR = os.path.join(BASE_DIR, "dataset")  # Assure que dataset/ est bien à la racine
EMBEDDINGS_FOLDER = os.path.join(DATASET_DIR, "embeddings")  # Corrige l'emplacement d'embeddings

os.makedirs(os.path.join(EMBEDDINGS_FOLDER, "lost"), exist_ok=True)
os.makedirs(os.path.join(EMBEDDINGS_FOLDER, "found"), exist_ok=True)

# 📌 Charger EfficientNet-B0
base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")

def load_faiss_index(category: str):
    """
    📌 Charge l'index FAISS et les UUIDs pour une catégorie donnée.
    """
    index_path = os.path.join(EMBEDDINGS_FOLDER, category, f"faiss_{category}.idx")
    uuids_path = os.path.join(EMBEDDINGS_FOLDER, category, f"image_uuids_{category}.npy")

    print(f"📌 Chemin FAISS : {index_path}")
    print(f"📌 Chemin UUIDs : {uuids_path}")

    if not os.path.exists(index_path):
        print(f"⚠️ Index FAISS manquant pour la catégorie {category}.")
    if not os.path.exists(uuids_path):
        print(f"⚠️ Fichier des UUIDs manquant pour la catégorie {category}.")
    
    if not os.path.exists(index_path) or not os.path.exists(uuids_path):
        return None, None

    try:
        index = faiss.read_index(index_path)
        image_uuids = np.load(uuids_path, allow_pickle=True)
        return index, image_uuids
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'index FAISS : {e}")
        return None, None

def search_similar_images(category: str, query_img_bytes: bytes, top_k: int = 5) -> List[dict]:
    """
    📌 Recherche les images les plus similaires en utilisant FAISS.
    """
    # Charger l'index FAISS et les UUIDs
    index, image_uuids = load_faiss_index(category)
    if index is None or image_uuids is None:
        print(f"⚠️ Aucun index FAISS trouvé pour la catégorie '{category}'.")
        return []

    # Extraire l'embedding de la requête
    query_embedding = extract_embeddings_for_search(query_img_bytes)
    if query_embedding.size == 0:
        print("⚠️ Embedding de la requête vide. Recherche annulée.")
        return []

    try:
        # Effectuer la recherche FAISS
        distances, indices = index.search(query_embedding, top_k)
        print(f"📌 Distances retournées : {distances[0]}")
        print(f"📌 Indices retournés : {indices[0]}")

        # Filtrer les résultats valides (indices >= 0 et distances > 0)
        SIMILARITY_THRESHOLD = 0.9  # Ajustez ce seuil selon vos besoins
        valid_results = [
            {"uuid": str(image_uuids[i]), "distance": float(d)}
            for i, d in zip(indices[0], distances[0])
            if i >= 0 and d > SIMILARITY_THRESHOLD
        ]

        if not valid_results:
            print("⚠️ Aucun voisin valide trouvé au-dessus du seuil de similarité.")
            return []

        print(f"✅ Résultats valides trouvés : {valid_results}")
        return valid_results
    except Exception as e:
        print(f"❌ Erreur lors de la recherche FAISS : {e}")
        return []

if __name__ == "__main__":
    """
    📌 Exécution autonome pour tester la recherche dans FAISS.
    """
    for category in ["lost", "found"]:
        input_folder = os.path.join(DATASET_DIR, category)
        category_folder = os.path.join(EMBEDDINGS_FOLDER, category)
        embeddings_file = os.path.join(category_folder, "image_embeddings.npy")
        
        # 🔹 Vérifier et créer les dossiers
        os.makedirs(input_folder, exist_ok=True)
        print(f"📌 Chemin absolu du dossier `{category}` : {input_folder}")
        print(f"📂 Fichiers détectés : {os.listdir(input_folder)}")

        # Charger les anciens embeddings
        if os.path.exists(embeddings_file):
            embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
        else:
            embeddings_dict = {}

        print(f"📂 {category}: {len(embeddings_dict)} images déjà indexées.")
