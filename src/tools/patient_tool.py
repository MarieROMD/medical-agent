"""
patient_tool.py — Outil 1 de l'agent médical
Cherche les informations d'un patient dans les dossiers PDF indexés.

Supporte les 3 pathologies :
  - Diabète      : glycémie, HbA1c, insuline, complications
  - Hypertension : tension, cardio, AVC, traitements
  - Cancer       : oncologie, marqueurs tumoraux, protocoles
"""

import os
import sys

# Ajouter le dossier racine au path Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_community.vectorstores import FAISS


class PatientTool:
    """
    Outil de recherche dans les dossiers patients indexés.

    Utilise la recherche vectorielle FAISS pour trouver
    les passages les plus pertinents dans les 20 dossiers PDF.

    Exemples d'utilisation :
      - "dossier Mohamed Benali"        → fiche patient P-20240001
      - "patient HbA1c 8.2%"            → patients diabétiques mal équilibrés
      - "patient hypertension sévère"   → patients HTA
      - "patient cancer sein HER2"      → patients oncologiques
    """

    # Nom et description utilisés par l'agent pour identifier l'outil
    name = "patient_search"
    description = (
        "Utilise cet outil pour chercher les informations d'un patient "
        "spécifique dans les dossiers médicaux : glycémie, HbA1c, tension "
        "artérielle, type de cancer, symptômes, traitements en cours, "
        "complications, antécédents, résultats biologiques. "
        "Fournir le nom du patient ou son numéro de dossier (ex: P-20240001)."
    )

    def __init__(self, vectorstore: FAISS):
        """
        Paramètres :
          vectorstore : base FAISS chargée depuis data/index_patients/
        """
        self.vectorstore = vectorstore

    def run(self, query: str, k: int = 5) -> str:
        """
        Cherche dans les dossiers patients et retourne
        les passages les plus pertinents.

        Paramètres :
          query : question ou nom du patient à rechercher
          k     : nombre de passages à retourner (défaut: 5)

        Retourne :
          str : passages pertinents formatés avec source et page
        """
        if not query or not query.strip():
            return "⚠️ Requête vide. Précise le nom ou l'ID du patient."

        # Recherche vectorielle dans l'index patients
        try:
            docs = self.vectorstore.similarity_search(query.strip(), k=k)
        except Exception as e:
            return f"⚠️ Erreur lors de la recherche : {str(e)}"

        if not docs:
            return (
                "Aucun dossier patient trouvé pour cette recherche.\n"
                "Vérifie que les PDFs sont bien dans data/patients/ "
                "et que l'indexation a été faite (python src/ingestion.py)."
            )

        # Formater les résultats
        results = []
        seen_sources = set()  # éviter les doublons

        for i, doc in enumerate(docs, 1):
            # Récupérer les métadonnées
            filename = doc.metadata.get("filename", "inconnu")
            page     = doc.metadata.get("page", "?")

            # Numéro de page lisible (commence à 0 dans LangChain)
            if isinstance(page, int):
                page = page + 1

            # Clé unique pour détecter les doublons
            key = f"{filename}_{page}"
            if key in seen_sources:
                continue
            seen_sources.add(key)

            results.append(
                f"[Dossier {i} | {filename} | page {page}]\n"
                f"{doc.page_content.strip()}"
            )

        if not results:
            return "Aucun résultat pertinent trouvé dans les dossiers patients."

        return "\n\n---\n\n".join(results)

    def search_by_id(self, patient_id: str) -> str:
        """
        Recherche un patient par son identifiant (ex: P-20240001).

        Paramètres :
          patient_id : identifiant du patient (ex: "P-20240001")
        """
        return self.run(f"numéro dossier {patient_id} patient")

    def search_by_pathology(self, pathology: str) -> str:
        """
        Recherche tous les patients d'une pathologie.

        Paramètres :
          pathology : "diabète", "hypertension" ou "cancer"
        """
        query_map = {
            "diabète":      "patient diabète glycémie HbA1c",
            "hypertension": "patient hypertension artérielle tension",
            "cancer":       "patient cancer tumeur chimiothérapie"
        }
        query = query_map.get(pathology.lower(), pathology)
        return self.run(query, k=8)

    def get_critical_patients(self) -> str:
        """
        Cherche les patients en situation d'urgence ou à risque élevé.
        """
        return self.run(
            "urgence risque élevé complications graves pied diabétique "
            "IC décompensée métastase rétinopathie proliférante",
            k=6
        )