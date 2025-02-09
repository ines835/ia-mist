import os
import faiss
import gc
from .search_with_faiss import load_faiss_index  # Recharge l'index après suppression
import sys

# Ajouter le dossier `src` au chemin
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Racine du projet
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)  # Ajoute la racine du projet au chemin Python
SRC_DIR = os.path.join(BASE_DIR, "src")  # Dossier `src`
sys.path.append(SRC_DIR)  # Ajoute `src` au chemin Python
DATASET_DIR = os.path.join(BASE_DIR, "dataset")  # Assure que dataset/ est bien à la racine
EMBEDDINGS_FOLDER = os.path.join(DATASET_DIR, "embeddings")  # Corrige l'emplacement d'embeddings

print(f"📌 BASE_DIR : {BASE_DIR}")
print(f"📌 SRC_DIR : {SRC_DIR}")
print(f"📌 DATASET_DIR : {DATASET_DIR}")
print(f"📌 EMBEDDINGS_FOLDER : {EMBEDDINGS_FOLDER}")

os.makedirs(os.path.join(EMBEDDINGS_FOLDER, "lost"), exist_ok=True)
os.makedirs(os.path.join(EMBEDDINGS_FOLDER, "found"), exist_ok=True)


# Script pour les tests : supprimer les fichiers et recharger FAISS
def force_reload_faiss(category):
    """
    📌 Force FAISS à oublier l'index et à recharger depuis le disque, avec gestion des UUIDs.
    """
    category_folder = os.path.join(EMBEDDINGS_FOLDER, category)
    index_path = os.path.join(category_folder, f"faiss_{category}.idx")
    embeddings_path = os.path.join(category_folder, "image_embeddings.npy")
    uuids_path = os.path.join(category_folder, f"image_uuids_{category}.npy")  # UUIDs remplacent les noms d'images

    # 🔥 Supprimer l'index FAISS
    if os.path.exists(index_path):
        os.remove(index_path)
        print(f"✅ Index FAISS supprimé pour {category}.")

    # 🔥 Supprimer les fichiers contenant les embeddings
    if os.path.exists(embeddings_path):
        os.remove(embeddings_path)
        print(f"✅ Embeddings supprimés pour {category}.")

    # 🔥 Supprimer la liste des UUIDs
    if os.path.exists(uuids_path):
        os.remove(uuids_path)
        print(f"✅ Liste des UUIDs supprimée pour {category}.")
    else:
        print(f"⚠️ Fichier UUIDs introuvable pour suppression : {uuids_path}")

    # 🔄 Forcer la suppression de l'index en mémoire
    gc.collect()

    # 📌 Recharge FAISS depuis zéro pour validation
    index, _ = load_faiss_index(category)

    if index is None or index.ntotal == 0:
        print(f"🔄 FAISS a bien été rechargé depuis zéro pour {category}.")
    else:
        print(f"⚠️ FAISS contient encore {index.ntotal} embeddings après suppression.")

# 🔥 Exécute le reset pour toutes les catégories
for cat in ["lost", "found"]:
    force_reload_faiss(cat)
