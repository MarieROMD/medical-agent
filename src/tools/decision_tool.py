"""
decision_tool.py — Outil 3 de l'agent médical
Combine dossier patient + guidelines OMS/HAS
pour générer une aide à la décision clinique complète.

C'est l'outil le plus puissant de l'agent :
il cherche en parallèle dans les deux bases FAISS
et retourne un contexte structuré pour le LLM.
"""

import os
import sys

# Ajouter le dossier racine au path Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_community.vectorstores import FAISS


class DecisionTool:
    """
    Outil d'aide à la décision clinique.

    Combine automatiquement :
      1. Dossier(s) patient(s) → base FAISS patients
      2. Recommandations OMS/HAS → base FAISS médicale

    Utilisé pour les questions complexes comme :
      - "Quel traitement pour le patient Benali ?"
      - "Le traitement de Vincent est-il adapté ?"
      - "Que faire pour ce patient diabétique mal équilibré ?"
    """

    # Nom et description utilisés par l'agent
    name = "clinical_decision"
    description = (
        "Utilise cet outil quand le médecin veut une aide à la décision "
        "complète : diagnostic, choix de traitement, ajustement de "
        "médicaments, évaluation du risque, conduite à tenir. "
        "Combine automatiquement le dossier patient et les guidelines "
        "OMS/HAS pour les 3 pathologies : diabète, hypertension, cancer."
    )

    def __init__(self, patient_store: FAISS, medical_store: FAISS):
        """
        Paramètres :
          patient_store : base FAISS depuis data/index_patients/
          medical_store : base FAISS depuis data/index_medical/
        """
        self.patient_store = patient_store
        self.medical_store = medical_store

    def run(self, query: str, k: int = 4) -> str:
        """
        Cherche en parallèle dans les deux bases FAISS
        et retourne un contexte combiné structuré.

        Paramètres :
          query : question clinique du médecin
          k     : nombre de passages par base (défaut: 4)

        Retourne :
          str : contexte combiné structuré (patient + guidelines)
        """
        if not query or not query.strip():
            return "⚠️ Requête vide. Précise ta question clinique."

        query = query.strip()

        # ── Étape 1 : Chercher dans les dossiers patients ────────────
        patient_context = self._search_patients(query, k)

        # ── Étape 2 : Chercher dans les guidelines médicales ─────────
        medical_context = self._search_medical(query, k)

        # ── Étape 3 : Combiner les deux contextes ─────────────────────
        return self._format_combined_context(patient_context, medical_context)

    # ── Recherche patients ─────────────────────────────────────────────────
    def _search_patients(self, query: str, k: int) -> list:
        """Cherche dans la base patients et retourne les docs pertinents"""
        try:
            docs = self.patient_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"⚠️ Erreur recherche patients : {e}")
            return []

    # ── Recherche médicale ─────────────────────────────────────────────────
    def _search_medical(self, query: str, k: int) -> list:
        """Cherche dans la base médicale et retourne les docs pertinents"""
        try:
            docs = self.medical_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"⚠️ Erreur recherche médicale : {e}")
            return []

    # ── Formatage du contexte combiné ─────────────────────────────────────
    def _format_combined_context(
        self,
        patient_docs: list,
        medical_docs: list
    ) -> str:
        """
        Formate les résultats des deux bases en un contexte
        structuré et lisible pour le LLM.
        """

        # ── Section patient ────────────────────────────────────────────
        if patient_docs:
            patient_parts = []
            seen = set()
            for i, doc in enumerate(patient_docs, 1):
                filename = doc.metadata.get("filename", "inconnu")
                page     = doc.metadata.get("page", "?")
                if isinstance(page, int):
                    page = page + 1

                key = f"{filename}_{page}"
                if key in seen:
                    continue
                seen.add(key)

                patient_parts.append(
                    f"[Dossier patient {i} | {filename} | page {page}]\n"
                    f"{doc.page_content.strip()}"
                )
            patient_section = "\n\n".join(patient_parts)
        else:
            patient_section = "Aucun dossier patient trouvé pour cette question."

        # ── Section guidelines ─────────────────────────────────────────
        if medical_docs:
            medical_parts = []
            seen = set()
            for i, doc in enumerate(medical_docs, 1):
                filename = doc.metadata.get("filename", "inconnu")
                page     = doc.metadata.get("page", "?")
                if isinstance(page, int):
                    page = page + 1

                key = f"{filename}_{page}"
                if key in seen:
                    continue
                seen.add(key)

                medical_parts.append(
                    f"[Guideline médicale {i} | {filename} | page {page}]\n"
                    f"{doc.page_content.strip()}"
                )
            medical_section = "\n\n".join(medical_parts)
        else:
            medical_section = "Aucune recommandation OMS/HAS trouvée pour cette question."

        # ── Contexte combiné final ─────────────────────────────────────
        return (
            "╔══════════════════════════════════════════╗\n"
            "║         DOSSIER(S) PATIENT(S)            ║\n"
            "╚══════════════════════════════════════════╝\n"
            f"{patient_section}\n\n"
            "╔══════════════════════════════════════════╗\n"
            "║    RECOMMANDATIONS MÉDICALES OMS/HAS     ║\n"
            "╚══════════════════════════════════════════╝\n"
            f"{medical_section}"
        )

    # ── Méthodes spécialisées ──────────────────────────────────────────────

    def evaluate_treatment(self, patient_name: str, pathology: str) -> str:
        """
        Évalue si le traitement actuel d'un patient est adapté.

        Paramètres :
          patient_name : nom du patient
          pathology    : "diabète", "hypertension" ou "cancer"
        """
        query = (
            f"traitement {patient_name} {pathology} "
            f"adapté recommandation protocole ajustement"
        )
        return self.run(query)

    def suggest_treatment(self, symptoms: str, pathology: str) -> str:
        """
        Suggère un traitement basé sur les symptômes et la pathologie.

        Paramètres :
          symptoms  : description des symptômes
          pathology : "diabète", "hypertension" ou "cancer"
        """
        query = (
            f"{symptoms} {pathology} "
            f"traitement recommandation prise en charge"
        )
        return self.run(query)

    def assess_risk(self, patient_name: str) -> str:
        """
        Évalue le niveau de risque d'un patient.

        Paramètres :
          patient_name : nom du patient
        """
        query = (
            f"{patient_name} risque complications "
            f"score cardiovasculaire urgence surveillance"
        )
        return self.run(query, k=5)

    def compare_with_guidelines(self, patient_name: str) -> str:
        """
        Compare la situation d'un patient avec les guidelines OMS/HAS.

        Paramètres :
          patient_name : nom du patient
        """
        query = (
            f"{patient_name} valeurs biologiques constantes "
            f"par rapport recommandations objectifs thérapeutiques"
        )
        return self.run(query)