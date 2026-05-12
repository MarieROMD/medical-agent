"""
medical_tool.py — Outil 2 de l'agent médical
Cherche dans les documents médicaux officiels OMS/HAS indexés.

Couvre 3 pathologies :
  - Diabète      : protocoles, valeurs cibles, médicaments
  - Hypertension : guidelines, antihypertenseurs, risque CV
  - Cancer       : chimiothérapie, immunothérapie, marqueurs
"""

import os
import sys

# Ajouter le dossier racine au path Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_community.vectorstores import FAISS


class MedicalTool:
    """
    Outil de recherche dans les guidelines médicales OMS/HAS.

    Utilise la recherche vectorielle FAISS pour trouver
    les recommandations officielles pertinentes.

    Exemples d'utilisation :
      - "critères diagnostiques diabète type 2"
      - "valeur cible HbA1c diabétique"
      - "protocole HTA résistante"
      - "chimiothérapie cancer sein HER2+"
    """

    # Nom et description utilisés par l'agent
    name = "medical_search"
    description = (
        "Utilise cet outil pour chercher des informations médicales "
        "officielles OMS/HAS : protocoles de traitement, critères "
        "diagnostiques, valeurs normales et cibles, recommandations "
        "thérapeutiques, posologies médicaments. "
        "Couvre diabète, hypertension artérielle et cancer."
    )

    def __init__(self, vectorstore: FAISS):
        """
        Paramètres :
          vectorstore : base FAISS chargée depuis data/index_medical/
        """
        self.vectorstore = vectorstore

    def run(self, query: str, k: int = 5) -> str:
        """
        Cherche dans les guidelines médicales et retourne
        les passages les plus pertinents.

        Paramètres :
          query : question médicale à rechercher
          k     : nombre de passages à retourner (défaut: 5)

        Retourne :
          str : passages pertinents formatés avec source et page
        """
        if not query or not query.strip():
            return "⚠️ Requête vide. Précise ta question médicale."

        # Recherche vectorielle dans l'index médical
        try:
            docs = self.vectorstore.similarity_search(query.strip(), k=k)
        except Exception as e:
            return f"⚠️ Erreur lors de la recherche : {str(e)}"

        if not docs:
            return (
                "Aucune information médicale trouvée pour cette recherche.\n"
                "Vérifie que les PDFs OMS/HAS sont dans data/medical_docs/ "
                "et que l'indexation a été faite (python src/ingestion.py)."
            )

        # Formater les résultats
        results = []
        seen_sources = set()  # éviter les doublons

        for i, doc in enumerate(docs, 1):
            # Récupérer les métadonnées
            filename = doc.metadata.get("filename", "inconnu")
            page     = doc.metadata.get("page", "?")

            # Numéro de page lisible
            if isinstance(page, int):
                page = page + 1

            # Clé unique pour détecter les doublons
            key = f"{filename}_{page}"
            if key in seen_sources:
                continue
            seen_sources.add(key)

            results.append(
                f"[Source médicale {i} | {filename} | page {page}]\n"
                f"{doc.page_content.strip()}"
            )

        if not results:
            return "Aucune recommandation médicale pertinente trouvée."

        return "\n\n---\n\n".join(results)

    # ── Méthodes spécialisées par domaine ─────────────────────────────────

    def get_diagnostic_criteria(self, pathology: str) -> str:
        """
        Cherche les critères diagnostiques d'une pathologie.

        Paramètres :
          pathology : "diabète", "hypertension" ou "cancer"
        """
        queries = {
            "diabète":      "critères diagnostiques diabète glycémie seuil HbA1c",
            "hypertension": "critères diagnostiques hypertension artérielle seuil tension",
            "cancer":       "critères diagnostiques cancer marqueurs tumoraux biopsie"
        }
        query = queries.get(pathology.lower(), f"critères diagnostiques {pathology}")
        return self.run(query)

    def get_treatment_protocol(self, pathology: str) -> str:
        """
        Cherche le protocole de traitement d'une pathologie.

        Paramètres :
          pathology : "diabète", "hypertension" ou "cancer"
        """
        queries = {
            "diabète":      "protocole traitement diabète metformine insuline première ligne",
            "hypertension": "protocole traitement hypertension antihypertenseur première ligne",
            "cancer":       "protocole chimiothérapie radiothérapie immunothérapie"
        }
        query = queries.get(pathology.lower(), f"protocole traitement {pathology}")
        return self.run(query)

    def get_target_values(self, pathology: str) -> str:
        """
        Cherche les valeurs cibles pour une pathologie.

        Paramètres :
          pathology : "diabète", "hypertension" ou "cancer"
        """
        queries = {
            "diabète":      "valeur cible HbA1c glycémie diabète objectif thérapeutique",
            "hypertension": "valeur cible tension artérielle objectif thérapeutique mmHg",
            "cancer":       "valeur normale marqueurs tumoraux PSA CA15-3 ACE"
        }
        query = queries.get(pathology.lower(), f"valeurs cibles {pathology}")
        return self.run(query)

    def get_drug_info(self, drug_name: str) -> str:
        """
        Cherche les informations sur un médicament.

        Paramètres :
          drug_name : nom du médicament (ex: "metformine", "amlodipine")
        """
        return self.run(
            f"{drug_name} posologie indication contre-indication effets secondaires"
        )

    def get_complications(self, pathology: str) -> str:
        """
        Cherche les complications d'une pathologie.

        Paramètres :
          pathology : "diabète", "hypertension" ou "cancer"
        """
        queries = {
            "diabète":      "complications diabète rétinopathie néphropathie neuropathie pied",
            "hypertension": "complications hypertension AVC IDM insuffisance rénale HVG",
            "cancer":       "complications cancer métastase effets secondaires chimio"
        }
        query = queries.get(pathology.lower(), f"complications {pathology}")
        return self.run(query)