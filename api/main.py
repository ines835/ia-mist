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

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet à toutes les origines d'accéder
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les méthodes HTTP
    allow_headers=["*"],  # Permet tous les headers
)


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

EMBEDDING_DIMENSION = 1280

@app.get("/")
def root():
    return {"message": "L'API est en ligne et fonctionne correctement ! 🚀"}

@app.post("/add/")
async def add_image(category: str, files: List[UploadFile] = File(...)):
    """
    📌 Ajoute une ou plusieurs images à une catégorie FAISS et vérifie si un embedding similaire existe déjà.
    """
    category_folder = os.path.join(EMBEDDINGS_FOLDER, category)
    os.makedirs(category_folder, exist_ok=True)

    print("📂 Chargement des images et extraction des embeddings...")
    image_bytes = [await file.read() for file in files]
    embeddings, uuids = extract_embeddings(image_bytes, category)

    if len(embeddings) == 0:
        print("❌ Erreur : Aucun embedding extrait.")
        return {"message": "Erreur : Impossible d'extraire les embeddings."}

    print("✅ Embeddings extraits avec succès.")
    print(f"👉 Dimension des embeddings : {embeddings.shape}")

    # Vérification de la dimension des embeddings
    if embeddings.shape[1] != 1280: 
        print(f"❌ Erreur : Embedding généré avec une dimension incorrecte ({embeddings.shape[1]}).")
        return {"message": "Erreur embedding généré incorrect"}

    # Charger les anciens embeddings
    embeddings_file = os.path.join(category_folder, "image_embeddings.npy")
    uuids_file = os.path.join(category_folder, f"image_uuids_{category}.npy")

    if os.path.exists(embeddings_file):
        embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
    else:
        embeddings_dict = {}

    if os.path.exists(uuids_file):
        existing_uuids = np.load(uuids_file, allow_pickle=True).tolist()
    else:
        existing_uuids = []

    print("🔍 Vérification des similarités...")
    # Vérifier si un embedding similaire existe déjà
    for new_embedding, _ in zip(embeddings, uuids):
        for existing_uuid, existing_embedding in embeddings_dict.items():
            similarity = np.dot(new_embedding, existing_embedding)  # Produit scalaire pour similarité cosinus
            if similarity > 0.99:  # Contrainte de similarité (ajustez le seuil selon vos besoins)
                print(f"⚠️ Match parfait détecté avec l'UUID existant : {existing_uuid}")
                return {
                    "message": "Un match parfait existe, veuillez consulter le post.",
                    "uuid": existing_uuid,
                }
            print("Ce log ne devrait jamais s'afficher !")

    print("✅ Aucun match parfait détecté. Ajout des nouveaux embeddings.")
    # Ajouter les nouveaux embeddings et UUIDs
    for uuid_val, embedding in zip(uuids, embeddings):
        embeddings_dict[uuid_val] = embedding
    existing_uuids.extend(uuids)

    # Sauvegarder les mises à jour
    print("💾 Sauvegarde des embeddings et des UUIDs...")
    np.save(embeddings_file, embeddings_dict)
    np.save(uuids_file, np.array(existing_uuids, dtype=object))

    # Mettre à jour l'index FAISS
    print("📈 Mise à jour de l'index FAISS...")
    update_or_create_faiss_index(category)

    response = {
        "message": "Les images ont été ajoutées avec succès.",
        "uuids": uuids,
    }
    print("✅ Processus terminé avec succès.")
    return response




# Endpoint pour lister les images avec vérification des UUIDs liés aux embeddings
@app.get("/list/")
async def list_images(category: str):
    """
    📌 Retourne la liste des UUIDs enregistrés et vérifie qu'ils sont liés aux embeddings dans l'index FAISS.
    """
    category_folder = os.path.join(EMBEDDINGS_FOLDER, category)
    uuids_file = os.path.join(category_folder, f"image_uuids_{category}.npy")
    embeddings_file = os.path.join(category_folder, "image_embeddings.npy")
    index_path = os.path.join(category_folder, f"faiss_{category}.idx")

    # Vérifier si les fichiers nécessaires existent
    if not os.path.exists(uuids_file) or not os.path.exists(embeddings_file) or not os.path.exists(index_path):
        return {"message": "Aucun index FAISS ou données associées trouvées pour cette catégorie."}

    # Charger les UUIDs et les embeddings
    uuids = np.load(uuids_file, allow_pickle=True).tolist()
    embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
    index = faiss.read_index(index_path)

    # Vérifier si tous les UUIDs ont des embeddings associés
    missing_uuids = [uuid for uuid in uuids if uuid not in embeddings_dict]
    linked_uuids = [uuid for uuid in uuids if uuid in embeddings_dict]

    response = {
        "category": category,
        "total_images": len(uuids),
        "faiss_embeddings": index.ntotal,
        "uuids": linked_uuids,
        "missing_uuids": missing_uuids,  # UUIDs sans embeddings associés
        "status": "ok" if not missing_uuids else "Mismatch detected",
    }

    if missing_uuids:
        print(f"⚠️ Les UUIDs suivants n'ont pas d'embeddings associés : {missing_uuids}")

    return response

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


@app.post("/delete/")
async def delete_image(category: str, uuid_to_delete: str = Form(...)):
    """
    📌 Supprime une image de l'index FAISS à l'aide de son UUID.
    """
    category_folder = os.path.join(EMBEDDINGS_FOLDER, category)
    embeddings_file = os.path.join(category_folder, "image_embeddings.npy")
    uuids_file = os.path.join(category_folder, f"image_uuids_{category}.npy")
    index_path = os.path.join(category_folder, f"faiss_{category}.idx")

    # Vérifier que les fichiers nécessaires existent
    if not os.path.exists(embeddings_file) or not os.path.exists(uuids_file) or not os.path.exists(index_path):
        return {"message": "Erreur : Données manquantes pour cette catégorie."}

    # Charger les embeddings et les UUIDs
    embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
    uuids = np.load(uuids_file, allow_pickle=True).tolist()

    if uuid_to_delete not in embeddings_dict:
        return {"message": "Erreur : UUID non trouvé dans l'index."}

    # Supprimer l'UUID et l'embedding associé
    del embeddings_dict[uuid_to_delete]
    uuids.remove(uuid_to_delete)

    # Recréer l'index FAISS sans l'embedding supprimé
    all_embeddings = np.array(list(embeddings_dict.values())).astype("float32")
    index = faiss.IndexFlatIP(all_embeddings.shape[1])  # Recrée un nouvel index FAISS
    index.add(all_embeddings)  # Ajoute les embeddings restants au nouvel index

    # Sauvegarder le nouvel index FAISS
    faiss.write_index(index, index_path)

    # Sauvegarder les mises à jour
    np.save(embeddings_file, embeddings_dict)
    np.save(uuids_file, np.array(uuids, dtype=object))

    return {"message": f"Image avec UUID {uuid_to_delete} et son embedding associé ont été supprimés avec succès de {category}."}



@app.post("/test-files/")
async def test_files(files: List[UploadFile] = File(...)):
    file_sizes = []
    for file in files:
        content = await file.read()
        file_sizes.append(len(content))
    return {"file_sizes": file_sizes}
