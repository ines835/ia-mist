import os
import numpy as np
import faiss
import uuid
from typing import Tuple

# 📌 Dossiers d'output
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Récupère le chemin absolu du script
print(BASE_DIR)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")  # Assure que dataset/ est bien à la racine
EMBEDDINGS_FOLDER = os.path.join(DATASET_DIR, "embeddings")  # Corrige l'emplacement d'embeddings

os.makedirs(os.path.join(EMBEDDINGS_FOLDER, "lost"), exist_ok=True)
os.makedirs(os.path.join(EMBEDDINGS_FOLDER, "found"), exist_ok=True)

def load_embeddings(category: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    📌 Charge les embeddings et les UUIDs pour `lost` ou `found`.
    Retourne : (embeddings_matrix, image_uuids)
    """
    embeddings_file = os.path.join(EMBEDDINGS_FOLDER, category, "image_embeddings.npy")
    uuids_file = os.path.join(EMBEDDINGS_FOLDER, category, f"image_uuids_{category}.npy")

    if not os.path.exists(embeddings_file) or not os.path.exists(uuids_file) or os.path.getsize(embeddings_file) == 0:
        print(f"⚠️ Aucun embedding trouvé pour la catégorie '{category}'.")
        return np.array([]), np.array([])

    embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
    image_uuids = np.load(uuids_file, allow_pickle=True)

    if not embeddings_dict or len(image_uuids) == 0:  # Vérifie si le dictionnaire ou la liste est vide
        print(f"⚠️ Les fichiers pour la catégorie '{category}' sont vides.")
        return np.array([]), np.array([])

    embeddings_matrix = np.array(list(embeddings_dict.values())).astype("float32")

    return embeddings_matrix, image_uuids

def update_or_create_faiss_index(category: str):
    """
    📌 Met à jour ou crée un index FAISS avec les nouveaux embeddings.
    """
    category_folder = os.path.join(EMBEDDINGS_FOLDER, category)
    os.makedirs(category_folder, exist_ok=True)

    embeddings_file = os.path.join(category_folder, "image_embeddings.npy")
    uuids_file = os.path.join(category_folder, f"image_uuids_{category}.npy")
    index_path = os.path.join(category_folder, f"faiss_{category}.idx")

    # Charger les anciens embeddings et UUIDs
    if os.path.exists(embeddings_file) and os.path.exists(uuids_file):
        embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
        image_uuids = np.load(uuids_file, allow_pickle=True).tolist()
    else:
        embeddings_dict = {}
        image_uuids = []

    if len(embeddings_dict) == 0:
        print(f"⚠️ Aucun embedding trouvé pour '{category}'. FAISS ne sera pas mis à jour.")
        return

    # Charger ou créer l'index FAISS
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"📌 FAISS chargé avec {index.ntotal} embeddings.")
    else:
        index = faiss.IndexFlatIP(1280)  # 1280 = dimension des embeddings EfficientNet-B0
        print(f"📌 Création d'un nouvel index FAISS pour '{category}'.")

    # Vérifier les embeddings déjà dans l'index
    num_existing_embeddings = index.ntotal
    print(f"📌 FAISS contient déjà {num_existing_embeddings} embeddings.")

    # Récupérer uniquement les nouveaux embeddings non encore ajoutés
    
    # Sélectionner uniquement les nouveaux embeddings
    all_uuids = list(embeddings_dict.keys())
    all_embeddings = np.array(list(embeddings_dict.values())).astype("float32")


    new_uuids = all_uuids[num_existing_embeddings:]
    new_embeddings = all_embeddings[num_existing_embeddings:]

    if len(new_embeddings) > 0:
        # Normaliser les nouveaux embeddings avant ajout
        new_embeddings /= np.linalg.norm(new_embeddings, axis=1, keepdims=True)

        # Ajouter les nouveaux embeddings à l'index FAISS
        index.add(new_embeddings)
        print(f"✅ {len(new_embeddings)} nouveaux embeddings ajoutés à l'index FAISS.")
    else:
        print(f"⚠️ Aucun nouvel embedding à ajouter à l'index FAISS pour '{category}'.")

    # Sauvegarder l'index FAISS
    faiss.write_index(index, index_path)
    np.save(os.path.join(category_folder, f"image_uuids_{category}.npy"), all_uuids)
    print(f"✅ Index FAISS sauvegardé pour la catégorie '{category}'.")

if __name__ == "__main__":
    """
    📌 Exécution autonome pour mettre à jour les index FAISS de `lost` et `found`.
    """
    update_or_create_faiss_index("lost")
    update_or_create_faiss_index("found")
