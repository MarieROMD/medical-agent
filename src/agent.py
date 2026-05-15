

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferWindowMemory

from src.tools.patient_tool    import PatientTool
from src.tools.medical_tool    import MedicalTool
from src.tools.decision_tool   import DecisionTool
from src.tools.prediction_tool import PredictionTool
from src.ingestion             import load_index, index_exists, INDEX_PAT, INDEX_MED

OLLAMA_MODEL = "mistral"


class MedicalAgent:
   

    def __init__(self, ollama_model: str = OLLAMA_MODEL):
        self.llm           = OllamaLLM(model=ollama_model, temperature=0.1)
        self.patient_tool  = None
        self.medical_tool  = None
        self.decision_tool = None
        self.pred_tool     = PredictionTool()
        self._ready        = False

        
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True
        )

   
    def initialize(self) -> bool:
        if not index_exists(INDEX_PAT):
            print("⚠️  Index patients manquant → python src/ingestion.py")
            return False
        if not index_exists(INDEX_MED):
            print("⚠️  Index médical manquant → python src/ingestion.py")
            return False

        pat_store = load_index(INDEX_PAT)
        med_store = load_index(INDEX_MED)

        self.patient_tool  = PatientTool(pat_store)
        self.medical_tool  = MedicalTool(med_store)
        self.decision_tool = DecisionTool(pat_store, med_store)

        self._ready = True
        print("✅ Agent médical initialisé — 4 outils actifs")
        print("   🗂️  patient_tool    → dossiers patients")
        print("   📚  medical_tool    → guidelines OMS/HAS")
        print("   🧠  decision_tool   → décision combinée")
        print("   🤖  pred_tool       → prédiction ML")
        return True

 
    def _choose_tool(self, question: str) -> str:
       
        q = question.lower()

      
        direct_kw = [
            "bonjour", "bonsoir", "salut", "merci", "au revoir",
            "comment vas", "comment tu", "qui es-tu", "qui êtes-vous",
            "qu'est-ce que tu", "c'est quoi le rag", "c'est quoi un agent"
        ]
        if any(k in q for k in direct_kw):
            return "direct"

       
        prediction_kw = [
            "prédiction", "prédit", "predict", "prédire",
            "risque diabète", "probabilité diabète",
            "score risque", "évaluer le risque", "estimer"
        ]
        if any(k in q for k in prediction_kw):
            return "prediction"

        decision_kw = [
            "que faire", "quel traitement", "adapter le traitement",
            "ajuster", "changer", "aide à la décision",
            "est adapté", "est-ce adapté", "est-ce correct",
            "quelle thérapie", "prise en charge",
            "conduite à tenir", "que proposez"
        ]
        if any(k in q for k in decision_kw):
            return "decision"

        patient_kw = [
           
            "patient", "dossier", "monsieur", "madame",
            "p-2024", "p00", "p01", "p02", "p03",
            "glycémie", "hba1c", "insuline",
            "diabète", "diabétique", "hyperglycémie", "hypoglycémie",
            "tension", "hypertension", "hypertendu",
            "hta", "pression artérielle",
            "cancer", "tumeur", "métastase",
            "chimiothérapie", "oncologie", "lymphome",
            "psa", "ca 15-3", "marqueur",
            "benali", "trabelsi", "mansouri", "gharbi",
            "jebali", "baccouche", "romdhani",
            "vincent", "martin", "chaabane", "aloui",
            "zouari", "hamza", "ferchichi",
            "bernard", "ayari", "miled", "khelif",
            "dridi", "marzouki"
        ]
        if any(k in q for k in patient_kw):
            return "patient"

        medical_kw = [
            "protocole", "recommandation", "guideline",
            "oms", "has", "critère", "définition",
            "valeur normale", "valeur cible", "seuil",
            "metformine", "insulinothérapie",
            "antihypertenseur", "bêtabloquant",
            "folfiri", "folfox", "abvd",
            "pembrolizumab", "trastuzumab",
            "radiothérapie", "immunothérapie"
        ]
        if any(k in q for k in medical_kw):
            return "medical"

        
        return "decision"

   
    def _generate_response(self, question: str, context: str, tool_used: str) -> str:
        """
        Génère une réponse clinique structurée via Ollama
        en utilisant le contexte récupéré par l'outil.
        """
        history = ""
        messages = self.memory.chat_memory.messages
        if messages:
            lines = ["=== Historique récent ==="]
            for msg in messages[-4:]:
                role = "Médecin" if msg.type == "human" else "Agent"
                lines.append(f"{role} : {msg.content[:200]}")
            history = "\n".join(lines) + "\n\n"

       
        if tool_used == "direct":
            prompt = f"""Tu es un assistant médical expert en diabétologie,
hypertension et oncologie.
Réponds de façon naturelle et professionnelle en français.

{history}Question : {question}

Réponse :"""

        else:
            tool_labels = {
                "patient":    "dossiers patients médicaux",
                "medical":    "guidelines OMS/HAS officielles",
                "decision":   "dossiers patients + guidelines OMS/HAS",
                "prediction": "modèle ML prédiction diabète"
            }
            label = tool_labels.get(tool_used, "sources médicales")

            prompt = f"""Tu es un assistant médical expert conçu pour aider
les médecins généralistes dans leur pratique quotidienne.

Tu couvres 3 pathologies :
- 🩸 Diabète type 1 et type 2
- ❤️  Hypertension artérielle
- 🎗️  Cancer (sein, côlon, prostate, poumon, lymphome)

RÈGLES STRICTES :
- Réponds UNIQUEMENT à partir du contexte médical fourni ci-dessous
- Si une information est absente : dis "Information non disponible dans mes sources"
- Structure ta réponse avec des sections claires
- Cite les sources (nom du fichier + page) à la fin
- Signale toute urgence avec ⚠️
- Ne remplace JAMAIS le jugement clinique du médecin
- Réponds en français

{history}=== CONTEXTE MÉDICAL (source : {label}) ===
{context}

=== QUESTION DU MÉDECIN ===
{question}

=== RÉPONSE CLINIQUE ==="""

        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            return f"⚠️ Erreur Ollama : {str(e)}\nVérifie qu'Ollama est lancé avec : ollama serve"

  
    def answer(self, question: str, patient_data: dict = None) -> dict:
        """
        Répond à une question médicale du médecin.

        L'agent :
          1. Choisit l'outil approprié (_choose_tool)
          2. Récupère le contexte avec l'outil
          3. Génère la réponse via Ollama (_generate_response)
          4. Met à jour la mémoire
        """
        if not self._ready:
            return {
                "answer":    "⚠️ Agent non initialisé. Lance : python src/ingestion.py",
                "tool_used": "aucun",
                "context":   ""
            }

        tool_used = self._choose_tool(question)

        context = ""

        if tool_used == "direct":
            context = ""

        elif tool_used == "prediction":
            if patient_data:
                context = self.pred_tool.run(patient_data)
            else:
                context = (
                    "Aucune donnée numérique fournie.\n"
                    "Utilise le formulaire dans la barre latérale "
                    "pour entrer glycémie, IMC, âge, tension."
                )

        elif tool_used == "patient":
            context = self.patient_tool.run(question, k=6)

        elif tool_used == "medical":
            context = self.medical_tool.run(question, k=5)

        else: 
            context = self.decision_tool.run(question, k=4)

      
        answer = self._generate_response(question, context, tool_used)

  
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(answer[:500])

        return {
            "answer":    answer,
            "tool_used": tool_used,
            "context":   context
        }

  
    def clear_memory(self):
        """Efface l'historique de conversation"""
        self.memory.clear()

    def get_status(self) -> dict:
        """Retourne le statut de l'agent"""
        return {
            "ready":        self._ready,
            "model":        OLLAMA_MODEL,
            "agent_type":   "Agentic RAG — Option 3",
            "tools_active": 4 if self._ready else 0,
            "history_len":  len(self.memory.chat_memory.messages)
        }