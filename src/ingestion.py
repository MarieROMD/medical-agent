"""
ingestion.py — Indexation des documents médicaux
Crée deux bases vectorielles FAISS séparées :
  - data/index_patients/  → 20 dossiers patients PDF
  - data/index_medical/   → PDFs OMS/HAS (diabète, hypertension, cancer)

Lance une seule fois : python src/ingestion.py
"""

import os
import sys
from pathlib import Path

# Ajouter le dossier racine au path Python
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

# ─── CONFIGURATION ────────────────────────────────────────────────────────

# Modèle d'embeddings — supporte français, arabe, anglais
EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Dossiers source
PATIENTS_DIR  = "data/patients"       # tes 20 PDFs patients
MEDICAL_DIR   = "data/medical_docs"   # PDFs OMS/HAS

# Dossiers index FAISS (créés automatiquement)
INDEX_PAT     = "data/index_patients"
INDEX_MED     = "data/index_medical"

# Paramètres de chunking
CHUNK_SIZE    = 500    # taille d'un chunk en caractères
CHUNK_OVERLAP = 100    # chevauchement entre chunks


# ─── EMBEDDINGS ───────────────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Charge le modèle d'embeddings HuggingFace.
    Premier lancement = téléchargement ~100MB (une seule fois).
    """
    print(f"🔢 Chargement modèle embeddings : {EMBED_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},      # utilise "cuda" si GPU disponible
        encode_kwargs={"normalize_embeddings": True}
    )


# ─── CHARGEMENT DOCUMENTS ─────────────────────────────────────────────────

def load_pdfs(docs_dir: str) -> list:
    """
    Charge tous les PDFs d'un dossier.
    Retourne une liste de Documents LangChain.
    """
    pdfs = list(Path(docs_dir).glob("**/*.pdf"))

    if not pdfs:
        print(f"⚠️  Aucun PDF trouvé dans {docs_dir}/")
        return []

    print(f"📂 {len(pdfs)} PDFs trouvés dans {docs_dir}/")

    documents = []
    for pdf_path in tqdm(pdfs, desc="Chargement PDFs"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages  = loader.load()

            # Ajouter le nom du fichier dans les métadonnées
            for page in pages:
                page.metadata["source"]    = str(pdf_path)
                page.metadata["filename"]  = pdf_path.name
                page.metadata["file_stem"] = pdf_path.stem

            documents.extend(pages)
        except Exception as e:
            print(f"⚠️  Erreur chargement {pdf_path.name} : {e}")
            continue

    print(f"✅ {len(documents)} pages chargées au total")
    return documents


# ─── CHUNKING ─────────────────────────────────────────────────────────────

def split_documents(documents: list) -> list:
    """
    Découpe les documents en chunks.
    RecursiveCharacterTextSplitter respecte les paragraphes
    et ne coupe pas les phrases en plein milieu.
    """
    if not documents:
        return []

    print(f"✂️  Chunking : taille={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )

    chunks = splitter.split_documents(documents)

    # Ajouter un identifiant à chaque chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"✅ {len(chunks)} chunks créés")
    return chunks


# ─── INDEXATION FAISS ─────────────────────────────────────────────────────

def create_index(chunks: list, index_dir: str) -> FAISS:
    """
    Crée les embeddings et stocke dans FAISS.
    Sauvegarde l'index sur disque pour réutilisation.
    """
    if not chunks:
        print(f"⚠️  Aucun chunk à indexer pour {index_dir}")
        return None

    print(f"📦 Création index FAISS → {index_dir}/")

    embeddings  = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Sauvegarder sur disque
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)

    print(f"✅ Index sauvegardé → {index_dir}/")
    return vectorstore


# ─── CHARGEMENT INDEX EXISTANT ────────────────────────────────────────────

def load_index(index_dir: str) -> FAISS:
    """
    Charge un index FAISS existant depuis le disque.
    Utilisé par l'agent au démarrage.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✅ Index chargé depuis {index_dir}/")
    return vectorstore


# ─── VÉRIFICATION ─────────────────────────────────────────────────────────

def index_exists(index_dir: str) -> bool:
    """
    Vérifie si un index FAISS existe déjà sur le disque.
    Utilisé par l'agent pour éviter de réindexer inutilement.
    """
    return (
        Path(f"{index_dir}/index.faiss").exists() and
        Path(f"{index_dir}/index.pkl").exists()
    )


# ─── PIPELINE COMPLET ─────────────────────────────────────────────────────

def index_directory(docs_dir: str, index_dir: str, label: str) -> FAISS:
    """
    Pipeline complet pour un dossier :
    PDFs → chargement → chunking → embeddings → FAISS
    """
    print(f"\n{'='*50}")
    print(f"📂 Indexation : {label}")
    print(f"{'='*50}")

    # Vérifier que le dossier existe
    os.makedirs(docs_dir, exist_ok=True)

    # Étape 1 : Charger les PDFs
    documents = load_pdfs(docs_dir)
    if not documents:
        return None

    # Étape 2 : Découper en chunks
    chunks = split_documents(documents)
    if not chunks:
        return None

    # Étape 3 : Créer l'index FAISS
    vectorstore = create_index(chunks, index_dir)

    return vectorstore


# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🏥 INDEXATION AGENT MÉDICAL IA")
    print("=" * 50)

    # Créer les dossiers si nécessaires
    os.makedirs(PATIENTS_DIR, exist_ok=True)
    os.makedirs(MEDICAL_DIR,  exist_ok=True)

    # Vérifier les PDFs disponibles
    n_patients = len(list(Path(PATIENTS_DIR).glob("**/*.pdf")))
    n_medical  = len(list(Path(MEDICAL_DIR).glob("**/*.pdf")))

    print(f"\n📊 PDFs disponibles :")
    print(f"   🗂️  Patients    : {n_patients} fichiers")
    print(f"   📚 Médical OMS : {n_medical}  fichiers")

    if n_patients == 0:
        print(f"\n⚠️  Ajoute tes 20 PDFs patients dans : {PATIENTS_DIR}/")

    if n_medical == 0:
        print(f"\n⚠️  Ajoute tes PDFs OMS/HAS dans : {MEDICAL_DIR}/")
        print("   Liens de téléchargement :")
        print("   • Diabète OMS : iris.who.int/bitstream/handle/10665/254648/9789242565256-fre.pdf")
        print("   • HTA OMS     : iris.who.int/bitstream/handle/10665/364487/9789240061460-fre.pdf")
        print("   • Cancer CIRC : events.iarc.who.int/.../GC66_2_Rapport_Biennal_2022-2023.pdf")

    if n_patients == 0 and n_medical == 0:
        print("\n❌ Aucun PDF trouvé. Ajoute des fichiers et relance.")
        sys.exit(1)

    # Indexation dossiers patients
    if n_patients > 0:
        vs_patients = index_directory(PATIENTS_DIR, INDEX_PAT, "Dossiers patients")
    else:
        print("\n⚠️  Indexation patients ignorée (aucun PDF)")

    # Indexation documents médicaux OMS/HAS
    if n_medical > 0:
        vs_medical = index_directory(MEDICAL_DIR, INDEX_MED, "Documents médicaux OMS/HAS")
    else:
        print("\n⚠️  Indexation médicale ignorée (aucun PDF)")

    # Résumé final
    print(f"\n{'='*50}")
    print("🎉 INDEXATION TERMINÉE !")
    print(f"{'='*50}")

    if index_exists(INDEX_PAT):
        print(f"✅ Index patients  → {INDEX_PAT}/")
    if index_exists(INDEX_MED):
        print(f"✅ Index médical   → {INDEX_MED}/")

    print(f"\n🚀 Lance maintenant : streamlit run app.py")
    print(f"{'='*50}\n")