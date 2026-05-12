"""
agent.py — Vrai LangChain Agent (Option 3 — Agentic RAG)
Utilise ZERO_SHOT_REACT_DESCRIPTION pour décider :
  - quand faire une recherche (patient, medical, decision)
  - quand répondre directement (salutation, question générale)
  - quel outil utiliser parmi les 4 disponibles

Pathologies couvertes : Diabète (type 1 & 2), Hypertension, Cancer
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_ollama import OllamaLLM

from src.tools.patient_tool    import PatientTool
from src.tools.medical_tool    import MedicalTool
from src.tools.decision_tool   import DecisionTool
from src.tools.prediction_tool import PredictionTool
from src.ingestion             import load_index, index_exists, INDEX_PAT, INDEX_MED

OLLAMA_MODEL = "mistral"


class MedicalAgent:
    """
    Vrai agent LangChain ZERO_SHOT_REACT_DESCRIPTION.

    L'agent décide SEUL à chaque question :
      1. Dois-je chercher dans les dossiers patients ?
      2. Dois-je chercher dans les guidelines OMS/HAS ?
      3. Dois-je combiner patient + guidelines ?
      4. Dois-je faire une prédiction ML ?
      5. Puis-je répondre directement sans chercher ?

    Pathologies : Diabète type 1 & 2, Hypertension, Cancer
    """

    def __init__(self, ollama_model: str = OLLAMA_MODEL):
        self.llm       = OllamaLLM(model=ollama_model, temperature=0.1)
        self.pred_tool = PredictionTool()
        self._agent    = None
        self._ready    = False

        # Mémoire conversationnelle — retient les 5 derniers échanges
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True
        )

    # ──────────────────────────────────────────────────────────────────────
    # INITIALISATION
    # ──────────────────────────────────────────────────────────────────────
    def initialize(self) -> bool:
        """
        Charge les index FAISS et initialise le vrai LangChain Agent
        avec ses 4 outils enregistrés.
        """
        if not index_exists(INDEX_PAT):
            print("⚠️  Index patients manquant → python src/ingestion.py")
            return False
        if not index_exists(INDEX_MED):
            print("⚠️  Index médical manquant → python src/ingestion.py")
            return False

        # Charger les bases vectorielles FAISS
        pat_store = load_index(INDEX_PAT)
        med_store = load_index(INDEX_MED)

        # Instancier les 3 outils RAG
        patient_tool  = PatientTool(pat_store)
        medical_tool  = MedicalTool(med_store)
        decision_tool = DecisionTool(pat_store, med_store)

        # ── Enregistrer les 4 outils LangChain ────────────────────────
        langchain_tools = [

            Tool(
                name="patient_search",
                func=patient_tool.run,
                description=(
                    "Utilise cet outil pour chercher des informations "
                    "sur un patient spécifique dans les dossiers médicaux. "
                    "Couvre : diabète type 1 et type 2 (glycémie, HbA1c, insuline), "
                    "hypertension artérielle (tension, cardio, AVC), "
                    "cancer (tumeur, marqueurs, chimiothérapie). "
                    "Input : nom du patient ou numéro dossier (ex: Benali, P-20240001)."
                )
            ),

            Tool(
                name="medical_search",
                func=medical_tool.run,
                description=(
                    "Utilise cet outil pour chercher des recommandations médicales "
                    "officielles OMS/HAS. "
                    "Couvre : protocoles diabète type 1 et type 2, "
                    "guidelines hypertension artérielle, "
                    "protocoles oncologiques (cancer sein, côlon, prostate, poumon, lymphome). "
                    "Input : question médicale générale sur protocole ou traitement."
                )
            ),

            Tool(
                name="clinical_decision",
                func=decision_tool.run,
                description=(
                    "Utilise cet outil pour une aide à la décision clinique complète "
                    "qui nécessite à la fois le dossier patient ET les guidelines OMS/HAS. "
                    "Utilise pour : adapter un traitement, évaluer si une thérapie est correcte, "
                    "prise en charge d'un patient diabétique, hypertendu ou cancéreux. "
                    "Input : question de décision clinique avec nom du patient."
                )
            ),

            Tool(
                name="diabetes_prediction",
                func=self._prediction_wrapper,
                description=(
                    "Utilise cet outil UNIQUEMENT pour prédire le risque de diabète "
                    "d'un patient à partir de données numériques. "
                    "Input : 'glucose=X bmi=Y age=Z' avec les valeurs numériques. "
                    "Exemple : 'glucose=185 bmi=29.7 age=53'."
                )
            ),
        ]

        # ── Initialiser le vrai LangChain Agent REACT ─────────────────
        self._agent = initialize_agent(
            tools=langchain_tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate",
            handle_parsing_errors=True,
            agent_kwargs={
                "prefix": """Tu es un assistant médical expert conçu pour aider
les médecins généralistes dans leur pratique quotidienne.

Tu couvres 3 domaines médicaux :
- 🩸 Diabète type 1 et type 2 : glycémie, HbA1c, insuline, metformine
- ❤️  Hypertension artérielle  : tension, antihypertenseurs, risque CV
- 🎗️  Cancer                   : oncologie, chimiothérapie, marqueurs tumoraux

Tu as accès à 4 outils. À chaque question tu décides SEUL :
- Question sur un patient précis → utilise patient_search
- Question sur des recommandations médicales → utilise medical_search
- Question nécessitant patient + guidelines → utilise clinical_decision
- Prédiction de risque avec des chiffres → utilise diabetes_prediction
- Question simple (bonjour, définition connue) → réponds DIRECTEMENT sans outil

Réponds toujours en français. Cite tes sources à la fin.""",

                "format_instructions": """Pour utiliser un outil, utilise EXACTEMENT ce format :

Thought: [ton raisonnement — quel outil et pourquoi]
Action: [patient_search | medical_search | clinical_decision | diabetes_prediction]
Action Input: [la question à envoyer à l'outil]
Observation: [résultat automatiquement rempli]

Quand tu as la réponse finale :
Thought: J'ai toutes les informations nécessaires.
Final Answer: [ta réponse clinique complète en français]"""
            }
        )

        self._ready = True
        print("✅ LangChain Agent ZERO_SHOT_REACT initialisé")
        print("   🗂️  patient_search      → dossiers patients (diabète/HTA/cancer)")
        print("   📚  medical_search      → guidelines OMS/HAS")
        print("   🧠  clinical_decision   → décision combinée")
        print("   🤖  diabetes_prediction → prédiction ML")
        return True

    # ──────────────────────────────────────────────────────────────────────
    # WRAPPER PRÉDICTION
    # ──────────────────────────────────────────────────────────────────────
    def _prediction_wrapper(self, input_str: str) -> str:
        """
        Parse la string d'input et appelle prediction_tool.
        Format attendu : 'glucose=185 bmi=29.7 age=53'
        """
        patient_data = {
            "glucose":           120.0,
            "bmi":                25.0,
            "age":                40,
            "blood_pressure":     72.0,
            "pregnancies":         0,
            "skin_thickness":     20.0,
            "insulin":            80.0,
            "diabetes_pedigree":   0.5
        }

        try:
            for part in input_str.strip().split():
                if "=" in part:
                    key, val = part.split("=", 1)
                    key = key.strip().lower()
                    val = val.strip()

                    key_map = {
                        "glucose":     "glucose",
                        "glycemie":    "glucose",
                        "glycémie":    "glucose",
                        "bmi":         "bmi",
                        "imc":         "bmi",
                        "age":         "age",
                        "âge":         "age",
                        "bp":          "blood_pressure",
                        "tension":     "blood_pressure",
                        "insulin":     "insulin",
                        "insuline":    "insulin",
                        "pregnancies": "pregnancies",
                        "grossesses":  "pregnancies",
                        "dpf":         "diabetes_pedigree",
                        "pedigree":    "diabetes_pedigree"
                    }

                    mapped = key_map.get(key, key)
                    if mapped in patient_data:
                        if mapped in ["age", "pregnancies"]:
                            patient_data[mapped] = int(float(val))
                        else:
                            patient_data[mapped] = float(val)

        except Exception as e:
            return f"⚠️ Erreur parsing : {e}. Format : 'glucose=185 bmi=29.7 age=53'"

        return self.pred_tool.run(patient_data)

    # ──────────────────────────────────────────────────────────────────────
    # RÉPONSE PRINCIPALE
    # ──────────────────────────────────────────────────────────────────────
    def answer(self, question: str, patient_data: dict = None) -> dict:
        """
        L'agent LangChain REACT décide SEUL comment répondre.

        Cycle REACT :
          Thought → Action → Observation → Thought → Final Answer
        """
        if not self._ready:
            return {
                "answer":    "⚠️ Agent non initialisé. Lance : python src/ingestion.py",
                "tool_used": "aucun",
                "context":   ""
            }

        # Prédiction directe si données numériques fournies
        if patient_data and (
            "prédict" in question.lower() or
            "risque"  in question.lower()
        ):
            context = self.pred_tool.run(patient_data)
            return {
                "answer":    context,
                "tool_used": "prediction",
                "context":   context
            }

        try:
            # ── L'agent REACT décide SEUL ──────────────────────────────
            response  = self._agent.invoke({"input": question})
            answer    = response.get("output", str(response))
            tool_used = self._detect_tool_used(question)

            return {
                "answer":    answer,
                "tool_used": tool_used,
                "context":   "Raisonnement REACT : Thought → Action → Observation → Answer"
            }

        except Exception as e:
            # Fallback → réponse directe LLM
            try:
                direct = self.llm.invoke(
                    f"Tu es un assistant médical expert. Réponds en français.\n\n"
                    f"Question : {question}"
                )
                return {
                    "answer":    direct,
                    "tool_used": "direct",
                    "context":   "Réponse directe sans outil (fallback)"
                }
            except Exception:
                pass

            return {
                "answer":    f"⚠️ Erreur : {str(e)}\nVérifie qu'Ollama est lancé.",
                "tool_used": "aucun",
                "context":   ""
            }

    # ──────────────────────────────────────────────────────────────────────
    # DÉTECTION OUTIL — pour badge interface uniquement
    # ──────────────────────────────────────────────────────────────────────
    def _detect_tool_used(self, question: str) -> str:
        """
        Détecte quel outil a été utilisé pour afficher le badge.
        La vraie décision est faite par le LangChain Agent REACT.
        """
        q = question.lower()

        # ── Prédiction ML ──────────────────────────────────────────────
        prediction_kw = [
            "prédiction", "risque", "score", "predict",
            "probabilité", "estimer", "calculer"
        ]

        # ── Décision clinique ──────────────────────────────────────────
        decision_kw = [
            "traitement", "que faire", "adapter", "décision",
            "prise en charge", "conduite", "recommande",
            "est adapté", "suggère", "que proposez"
        ]

        # ── Dossier patient — 3 pathologies ───────────────────────────
        patient_kw = [
            # identifiants
            "patient", "dossier", "monsieur", "madame", "p-2024",
            # diabète type 1 et 2
            "glycémie", "hba1c", "insuline", "diabète", "diabétique",
            "hyperglycémie", "hypoglycémie", "diabète type 1",
            "diabète type 2", "metformine", "glargine",
            # hypertension
            "tension", "hypertension", "hypertendu", "hypertendue",
            "amlodipine", "lisinopril", "ramipril", "losartan",
            "hta", "pression artérielle",
            # cancer
            "cancer", "tumeur", "métastase", "chimiothérapie",
            "oncologie", "lymphome", "leucémie", "carcinome",
            "psa", "ca 15-3", "ace", "marqueur tumoral",
            "radiothérapie", "immunothérapie",
            # noms des 20 patients
            "benali", "trabelsi", "mansouri", "gharbi",
            "jebali", "baccouche", "romdhani",
            "vincent", "martin", "chaabane", "aloui",
            "zouari", "hamza", "ferchichi",
            "bernard", "ayari", "miled", "khelif",
            "dridi", "marzouki"
        ]

        # ── Guidelines médicales ───────────────────────────────────────
        medical_kw = [
            # général
            "protocole", "recommandation", "guideline",
            "oms", "has", "critère", "définition",
            "valeur normale", "valeur cible", "seuil",
            # diabète
            "insulinothérapie", "antidiabétique",
            # hypertension
            "antihypertenseur", "bêtabloquant", "diurétique",
            "iec", "ara2", "inhibiteur calcique",
            "hta résistante", "hta sévère",
            # cancer
            "folfiri", "folfox", "abvd", "tchp",
            "pembrolizumab", "trastuzumab", "bevacizumab",
            "cancer sein", "cancer côlon", "cancer poumon",
            "cancer prostate", "lymphome hodgkin"
        ]

        if any(k in q for k in prediction_kw):
            return "prediction"
        if any(k in q for k in decision_kw):
            return "decision"
        if any(k in q for k in patient_kw):
            return "patient"
        if any(k in q for k in medical_kw):
            return "medical"

        return "direct"

    # ──────────────────────────────────────────────────────────────────────
    # UTILITAIRES
    # ──────────────────────────────────────────────────────────────────────
    def clear_memory(self):
        """Efface l'historique de conversation"""
        self.memory.clear()

    def get_status(self) -> dict:
        """Retourne le statut de l'agent"""
        return {
            "ready":        self._ready,
            "model":        OLLAMA_MODEL,
            "agent_type":   "ZERO_SHOT_REACT_DESCRIPTION",
            "tools_active": 4 if self._ready else 0,
            "history_len":  len(self.memory.chat_memory.messages) if self._ready else 0
        }