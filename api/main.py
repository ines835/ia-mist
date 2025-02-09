from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import os
import uuid
import numpy as np
import faiss
from typing import List
from src.extract_embedding import extract_embeddings
from src.build_faiss import update_or_create_faiss_index
from src.search_with_faiss import load_faiss_index, search_similar_images
import sys

app = FastAPI()

# Configuration des dossiers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Racine du projet IA-MIST
sys.path.append(BASE_DIR)  # Ajoute la racine du projet IA-MIST au chemin Python
DATASET_DIR = os.path.join(BASE_DIR, "dataset")  # Assure que dataset/ est bien à la racine
EMBEDDINGS_FOLDER = os.path.join(DATASET_DIR, "embeddings")  # Corrige l'emplacement d'embeddings

# Middleware pour limiter la taille des requêtes
class LimitSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        max_size = 30 * 1024 * 1024  # Limite : 30 Mo
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > max_size:
            raise HTTPException(status_code=413, detail="Request too large")
        return await call_next(request)

app.add_middleware(LimitSizeMiddleware)

# Ajout d'images
@app.post("/add/")
async def add_image(category: str, files: List[UploadFile] = File(...)):
    """
    📌 Ajoute une ou plusieurs images à une catégorie FAISS et met à jour l'index.
    """
    category_folder = os.path.join(EMBEDDINGS_FOLDER, category)
    os.makedirs(category_folder, exist_ok=True)

    image_bytes = [await file.read() for file in files]
    embeddings, uuids = extract_embeddings(image_bytes, category)

    if len(embeddings) == 0:
        return {"message": "Erreur : Impossible d'extraire les embeddings."}

    # Charger les anciens embeddings
    embeddings_file = os.path.join(category_folder, "image_embeddings.npy")
    if os.path.exists(embeddings_file):
        embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
    else:
        embeddings_dict = {}

    # Charger les UUIDs existants
    uuids_file = os.path.join(category_folder, f"image_uuids_{category}.npy")
    if os.path.exists(uuids_file):
        existing_uuids = np.load(uuids_file, allow_pickle=True).tolist()
    else:
        existing_uuids = []

    # Ajouter les nouveaux embeddings et UUIDs
    for uuid_val, embedding in zip(uuids, embeddings):
        embeddings_dict[uuid_val] = embedding
    existing_uuids.extend(uuids)

    # Sauvegarder les mises à jour
    np.save(embeddings_file, embeddings_dict)
    np.save(uuids_file, np.array(existing_uuids, dtype=object))

    # Mettre à jour l'index FAISS
    update_or_create_faiss_index(category)

    return {"message": f"{len(uuids)} image(s) ajoutée(s) à {category} et index FAISS mis à jour."}

# Liste des images
@app.get("/list/")
async def list_images(category: str):
    """
    📌 Retourne la liste des UUIDs enregistrés dans l'index FAISS.
    """
    uuids_file = os.path.join(EMBEDDINGS_FOLDER, category, f"image_uuids_{category}.npy")

    if not os.path.exists(uuids_file):
        return {"message": "Aucune image enregistrée pour cette catégorie."}

    uuids = np.load(uuids_file, allow_pickle=True).tolist()
    index = faiss.read_index(os.path.join(EMBEDDINGS_FOLDER, category, f"faiss_{category}.idx"))

    return {
        "category": category,
        "total_images": len(uuids),
        "faiss_embeddings": index.ntotal,
        "uuids": uuids,
    }

# Recherche d'images
@app.post("/search/")
async def search_image(category: str, file: UploadFile = File(...)):
    """
    📌 Recherche les images les plus similaires en utilisant FAISS.
    """
    query_image_bytes = await file.read()
    query_results = search_similar_images(category, query_image_bytes, top_k=5)

    if not query_results:
        return {"message": "Aucune image similaire trouvée."}

    return query_results

# Suppression d'une image
@app.post("/delete/")
async def delete_image(category: str, uuid_to_delete: str = Form(...)):
    """
    📌 Supprime une image de l'index FAISS à l'aide de son UUID.
    """
    category_folder = os.path.join(EMBEDDINGS_FOLDER, category)
    embeddings_file = os.path.join(category_folder, "image_embeddings.npy")
    uuids_file = os.path.join(category_folder, f"image_uuids_{category}.npy")

    if not os.path.exists(embeddings_file):
        return {"message": "Erreur : Aucun embedding trouvé pour cette catégorie."}

    embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
    if uuid_to_delete not in embeddings_dict:
        return {"message": "Erreur : UUID non trouvé dans l'index."}

    # Supprimer l'embedding et l'UUID
    del embeddings_dict[uuid_to_delete]
    np.save(embeddings_file, embeddings_dict)

    uuids = np.load(uuids_file, allow_pickle=True).tolist()
    uuids.remove(uuid_to_delete)
    np.save(uuids_file, np.array(uuids, dtype=object))

    # Mettre à jour l'index FAISS
    update_or_create_faiss_index(category)

    return {"message": f"Image avec UUID {uuid_to_delete} supprimée avec succès de {category}."}


@app.post("/test-files/")
async def test_files(files: List[UploadFile] = File(...)):
    file_sizes = []
    for file in files:
        content = await file.read()
        file_sizes.append(len(content))
    return {"file_sizes": file_sizes}
