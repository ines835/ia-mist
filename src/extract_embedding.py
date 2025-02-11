from io import BytesIO
import os
import uuid
from PIL import Image
import numpy as np
import tensorflow as tf
import psutil
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input 
from tensorflow.keras.preprocessing import image
from typing import List, Tuple

# 💌 Détection de la RAM et réglage du batch size
TOTAL_RAM_GB = psutil.virtual_memory().total / 1e9
BATCH_SIZE = 64 if TOTAL_RAM_GB > 16 else 16

# 💌 Détection des cœurs CPU
NUM_CORES = os.cpu_count()
NUM_WORKERS = max(4, NUM_CORES // 4)

# 💌 Charger EfficientNet-B0 sans la couche de classification
base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")

# 💌 Dossiers d'output
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Récupère le chemin absolu du script
print(BASE_DIR)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")  # Assure que dataset/ est bien à la racine
EMBEDDINGS_FOLDER = os.path.join(DATASET_DIR, "embeddings")  # Corrige l'emplacement d'embeddings

os.makedirs(os.path.join(EMBEDDINGS_FOLDER, "lost"), exist_ok=True)
os.makedirs(os.path.join(EMBEDDINGS_FOLDER, "found"), exist_ok=True)

def load_and_preprocess_image(img_bytes: bytes) -> Tuple[str, np.ndarray]:
    """
    💌 Charge et prétraite une image depuis `bytes` en conservant son format.
    - Garde le format d'origine si connu, sinon attribue `.jpg` par défaut.
    - Convertit en RGB si nécessaire pour compatibilité avec EfficientNet.
    - Renvoie : (nom fictif de l'image, image sous forme de tenseur).
    """
    try:
        # 🔹 Charger l’image en mémoire
        img = Image.open(BytesIO(img_bytes))

        # 🔹 Vérifier le format de l’image
        img_format = img.format  # Exemple : "JPEG", "PNG", "GIF", "BMP", etc.

        # 🔹 Associer une extension (sans conversion forcée)
        extension_map = {"JPEG": ".jpg", "PNG": ".png", "GIF": ".gif", "BMP": ".bmp", "TIFF": ".tiff"}
        extension = extension_map.get(img_format)

        # 🔹 Si le format est inconnu, on met `.jpg` par défaut
        if extension is None:
            print(f"⚠️ Format inconnu ({img_format}). Attribution de `.jpg` par défaut.")
            extension = ".jpg"

        # 🔹 Si l'image n'est pas en RGB, la convertir (EfficientNet exige RGB)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 🔹 Redimensionner pour EfficientNet (224x224)
        img = img.resize((224, 224))

        # 🔹 Convertir en tableau NumPy
        img_array = image.img_to_array(img)

        # 🔹 Appliquer la normalisation EfficientNet
        img_array = preprocess_input(img_array)

        # 🔹 Générer un nom fictif avec extension correcte
        return f"image_{np.random.randint(10000)}{extension}", img_array
    except Exception as e:
        print(f"❌ Erreur lors du traitement d’une image : {e}")
        return None, None

def extract_embeddings(images_bytes: List[bytes], category: str) -> Tuple[np.ndarray, List[str]]:
    """
    💌 Extrait les embeddings pour une liste d'images et enregistre les UUIDs.
    """
    category_folder = os.path.join(EMBEDDINGS_FOLDER, category)
    uuids_file = os.path.join(category_folder, f"image_uuids_{category}.npy")

    # ✅ Prétraitement des images
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(load_and_preprocess_image, images_bytes))

    valid_results = [res for res in results if res is not None and res[0] is not None]
    if not valid_results:
        return np.array([]), []

    img_arrays = []
    uuids = []

    for _, img_array in valid_results:
        img_arrays.append(img_array)
        uuids.append(str(uuid.uuid4()))  # Générer un UUID unique

    img_arrays = np.array(img_arrays)

    # ✅ Extraire les embeddings
    embeddings = base_model.predict(img_arrays, batch_size=BATCH_SIZE)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

  

    return embeddings, uuids

def extract_embeddings_for_search(image_bytes: bytes) -> np.ndarray:
    """
    💌 Extrait les embeddings pour une seule image.
    """
    # ✅ Prétraitement pour une seule image
    _, img_array = load_and_preprocess_image(image_bytes)
    if img_array is None or not isinstance(img_array, np.ndarray):
        print("⚠️ Prétraitement échoué ou résultat invalide.")
        return np.array([])  # Retourne un tableau vide en cas d'erreur

    # Vérification de la dimension de l'image
    if len(img_array.shape) != 3 or img_array.shape[:2] != (224, 224):  # Pour EfficientNet (224x224x3)
        print(f"⚠️ Image non conforme : {img_array.shape}. Redimensionnement nécessaire.")
        img_array = np.resize(img_array, (224, 224, 3))  # Redimensionner si nécessaire

    # ✅ Extraire les embeddings
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    img_embedding = base_model.predict(img_array)  # Prédiction avec le modèle
    img_embedding /= np.linalg.norm(img_embedding, axis=1, keepdims=True)  # Normalisation des embeddings

    return img_embedding

if __name__ == "__main__":
    """
    💌 Exécution autonome : extraction complète des embeddings pour `lost` et `found`.
    """
    for category in ["lost", "found"]:
        input_folder = os.path.join(DATASET_DIR, category)
        embeddings_file = os.path.join(EMBEDDINGS_FOLDER, category, "image_embeddings.npy")
        uuids_file = os.path.join(EMBEDDINGS_FOLDER, category, f"image_uuids_{category}.npy")

        # 🔹 Debug : Vérifier si le chemin généré est correct
        print(f"📈 Chemin absolu du dossier `{category}` : {input_folder}")

        # 🔹 Vérifier si le dossier existe, sinon le créer
        if not os.path.exists(input_folder):
            print(f"❌ ERREUR : Dossier '{input_folder}' manquant. Création en cours...")
            os.makedirs(input_folder, exist_ok=True)  # 🚀 Crée le dossier automatiquement

        print(f"✅ Dossier trouvé : {input_folder}")
        print(f"🗂 Fichiers détectés : {os.listdir(input_folder)}")

        # Charger les anciens embeddings et UUIDs
        if os.path.exists(embeddings_file):
            embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
        else:
            embeddings_dict = {}

        if os.path.exists(uuids_file):
            existing_uuids = np.load(uuids_file, allow_pickle=True).tolist()
        else:
            existing_uuids = []

        # 🔹 Obtenir la liste des images disponibles
        current_images = set(os.listdir(input_folder))
        existing_images = set(embeddings_dict.keys())
        new_images = list(current_images - existing_images)  # Filtrer les images déjà traitées

        print(f"🗂 {category}: {len(existing_images)} images déjà traitées")
        print(f"🔟 {category}: {len(new_images)} nouvelles images à extraire")

        # 🔥 Traitement des nouvelles images en batch
        for i in range(0, len(new_images), BATCH_SIZE):
            batch_filenames = new_images[i:i+BATCH_SIZE]
            batch_paths = [os.path.join(input_folder, img) for img in batch_filenames]

            # ✅ Lire les fichiers en mémoire (bytes)
            batch_bytes = [open(img_path, "rb").read() for img_path in batch_paths]

            # ✅ Extraire les embeddings depuis les bytes
            embeddings, uuids = extract_embeddings(batch_bytes, category)

            # ✅ Mise à jour progressive du dictionnaire d'embeddings
            for uuid_val, embedding in zip(uuids, embeddings):
                embeddings_dict[uuid_val] = embedding

            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                print("❌ Erreur : Certains embeddings contiennent NaN ou Inf.")

            # ✅ Sauvegarde progressive des embeddings et UUIDs
            np.save(embeddings_file, embeddings_dict)
            np.save(uuids_file, np.array(existing_uuids + uuids, dtype=object))

        print(f"✅ Extraction terminée pour {category} ! {len(embeddings_dict)} embeddings sauvegardés.")
