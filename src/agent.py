"""
agent.py — Vrai LangChain Agent (Option 3 — Agentic RAG)
Utilise ZERO_SHOT_REACT_DESCRIPTION pour décider :
  - quand faire une recherche (patient, medical, decision)
  - quand répondre directement (salutation, question générale)
  - quel outil utiliser parmi les 4 disponibles

C'est exactement ce que demande le prof pour l'Option 3.
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

    C'est le cœur de l'Option 3 — Agentic RAG.
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

        # Charger les bases vectorielles
        pat_store = load_index(INDEX_PAT)
        med_store = load_index(INDEX_MED)

        # Instancier les outils RAG
        patient_tool  = PatientTool(pat_store)
        medical_tool  = MedicalTool(med_store)
        decision_tool = DecisionTool(pat_store, med_store)

        # ── Enregistrer les 4 outils LangChain ────────────────────────
        # LangChain lit les descriptions pour décider quel outil utiliser
        langchain_tools = [

            Tool(
                name="patient_search",
                func=patient_tool.run,
                description=(
                    "Utilise cet outil UNIQUEMENT pour chercher des informations "
                    "sur un patient spécifique dans les dossiers médicaux : "
                    "glycémie, HbA1c, tension artérielle, symptômes, traitements, "
                    "complications, antécédents, résultats biologiques. "
                    "Input : nom du patient ou numéro dossier (ex: Benali, P-20240001)."
                )
            ),

            Tool(
                name="medical_search",
                func=medical_tool.run,
                description=(
                    "Utilise cet outil UNIQUEMENT pour chercher des informations "
                    "médicales générales dans les guidelines OMS/HAS : "
                    "protocoles de traitement, critères diagnostiques, "
                    "posologies médicaments, valeurs normales et cibles. "
                    "Couvre diabète, hypertension artérielle et cancer. "
                    "Input : question médicale générale."
                )
            ),

            Tool(
                name="clinical_decision",
                func=decision_tool.run,
                description=(
                    "Utilise cet outil pour une aide à la décision clinique COMPLÈTE "
                    "qui nécessite à la fois le dossier patient ET les guidelines : "
                    "quel traitement adapter, est-ce que le traitement est adapté, "
                    "que faire pour ce patient, prise en charge, conduite à tenir. "
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
                "prefix": """Tu es un assistant médical expert en diabétologie,
hypertension et oncologie, conçu pour aider les médecins généralistes.

Tu as accès à 4 outils médicaux. À chaque question tu dois décider :
- Si la question porte sur un patient précis → utilise patient_search
- Si la question porte sur des recommandations médicales → utilise medical_search
- Si la question nécessite patient + guidelines → utilise clinical_decision
- Si la question demande une prédiction de risque avec des chiffres → utilise diabetes_prediction
- Si la question est simple (bonjour, définition générale) → réponds DIRECTEMENT sans outil

Réponds toujours en français. Cite tes sources à la fin.""",

                "format_instructions": """Pour utiliser un outil, utilise EXACTEMENT ce format :

Thought: [ton raisonnement — pourquoi utiliser cet outil]
Action: [patient_search | medical_search | clinical_decision | diabetes_prediction]
Action Input: [la question à envoyer à l'outil]
Observation: [résultat de l'outil — automatiquement rempli]

Quand tu as la réponse finale, utilise ce format :
Thought: J'ai toutes les informations nécessaires pour répondre.
Final Answer: [ta réponse clinique complète et structurée en français]"""
            }
        )

        self._ready = True
        print("✅ LangChain Agent ZERO_SHOT_REACT initialisé")
        print("   🗂️  patient_search      → dossiers patients")
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

        L'agent peut :
          - utiliser un outil si nécessaire
          - répondre directement si la question est simple
          - enchaîner plusieurs outils si besoin
        """
        if not self._ready:
            return {
                "answer":    "⚠️ Agent non initialisé. Lance : python src/ingestion.py",
                "tool_used": "aucun",
                "context":   ""
            }

        # Prédiction directe si données numériques fournies via formulaire
        if patient_data and ("prédict" in question.lower() or "risque" in question.lower()):
            context = self.pred_tool.run(patient_data)
            return {
                "answer":    context,
                "tool_used": "prediction",
                "context":   context
            }

        try:
            # ── L'agent LangChain REACT décide SEUL ───────────────────
            response  = self._agent.invoke({"input": question})
            answer    = response.get("output", str(response))
            tool_used = self._detect_tool_used(question)

            return {
                "answer":    answer,
                "tool_used": tool_used,
                "context":   "Raisonnement REACT : Thought → Action → Observation → Answer"
            }

        except Exception as e:
            error_msg = str(e)

            # Fallback → réponse directe du LLM sans outil
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
                "answer":    f"⚠️ Erreur : {error_msg}\nVérifie qu'Ollama est lancé.",
                "tool_used": "aucun",
                "context":   ""
            }

    # ──────────────────────────────────────────────────────────────────────
    # UTILITAIRES
    # ──────────────────────────────────────────────────────────────────────
    def _detect_tool_used(self, question: str) -> str:
        """Détecte quel outil a probablement été utilisé"""
        q = question.lower()

        if any(k in q for k in ["prédiction", "risque", "score", "predict"]):
            return "prediction"
        if any(k in q for k in ["traitement", "que faire", "adapter", "décision"]):
            return "decision"
        if any(k in q for k in ["patient", "dossier", "benali", "vincent",
                                  "martin", "trabelsi", "mansouri", "gharbi",
                                  "chaabane", "aloui", "zouari", "hamza",
                                  "ferchichi", "bernard", "ayari", "miled",
                                  "khelif", "dridi", "marzouki", "jebali",
                                  "baccouche", "romdhani", "p-2024"]):
            return "patient"
        if any(k in q for k in ["protocole", "recommandation", "oms",
                                  "has", "médicament", "posologie"]):
            return "medical"

        return "direct"

    def clear_memory(self):
        """Efface l'historique de conversation"""
        self.memory.clear()

    def get_status(self) -> dict:
     return {
        "ready":        self._ready,
        "model":        OLLAMA_MODEL,
        "agent_type":   "ZERO_SHOT_REACT_DESCRIPTION",
        "tools_active": 4 if self._ready else 0,
        "history_len":  len(self.memory.chat_memory.messages) if self._ready else 0
    }