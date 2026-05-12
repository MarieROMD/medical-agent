"""
prediction_tool.py — Outil 4 de l'agent médical
Prédit le risque de diabète via le modèle ML HuggingFace.

Modèle   : Oduwo/Diabetes_assessment_Model
Dataset  : Pima Indians Diabetes (standard ML médical)
Features : Pregnancies, Glucose, BloodPressure, SkinThickness,
           Insulin, BMI, DiabetesPedigreeFunction, Age
"""

import os
import sys
import numpy as np

# Ajouter le dossier racine au path Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ─── CHARGEMENT DU MODÈLE ML ──────────────────────────────────────────────
_model = None   # singleton — chargé une seule fois


def load_model():
    """
    Télécharge et charge le modèle ML depuis HuggingFace.
    Mis en cache globalement pour éviter les rechargements.
    """
    global _model

    if _model is not None:
        return _model

    try:
        from huggingface_hub import hf_hub_download
        import joblib

        print("⏳ Téléchargement modèle ML depuis HuggingFace...")
        print("   Repo : Oduwo/Diabetes_assessment_Model")

        model_path = hf_hub_download(
            repo_id="Oduwo/Diabetes_assessment_Model",
            filename="model.pkl"
        )
        _model = joblib.load(model_path)
        print("✅ Modèle ML diabète chargé avec succès")

    except ImportError as e:
        print(f"⚠️  Bibliothèque manquante : {e}")
        print("   Lance : pip install huggingface_hub joblib scikit-learn")
        _model = None

    except Exception as e:
        print(f"⚠️  Erreur chargement modèle ML : {e}")
        print("   Vérifies ta connexion internet (HuggingFace)")
        _model = None

    return _model


class PredictionTool:
    """
    Outil de prédiction ML du risque de diabète.

    Utilise un modèle scikit-learn pré-entraîné sur le dataset
    Pima Indians Diabetes pour prédire si un patient est
    diabétique ou non, avec un score de risque en pourcentage.

    Features attendues (unités standard) :
      - glucose          : glycémie à jeun en mg/dL (ex: 120)
      - bmi              : IMC en kg/m² (ex: 28.5)
      - age              : âge en années (ex: 45)
      - blood_pressure   : tension diastolique en mmHg (ex: 80)
      - pregnancies      : nombre de grossesses (0 pour les hommes)
      - skin_thickness   : épaisseur pli cutané en mm (ex: 20)
      - insulin          : insuline sérique en µU/mL (ex: 80)
      - diabetes_pedigree: score génétique 0.0 à 2.5 (ex: 0.5)
    """

    # Nom et description utilisés par l'agent
    name = "diabetes_prediction"
    description = (
        "Utilise cet outil pour prédire le risque de diabète d'un patient "
        "à partir de ses données cliniques numériques : glycémie, IMC, âge, "
        "tension artérielle, insuline. "
        "Retourne : diabétique / non diabétique + niveau de risque en %."
    )

    def __init__(self):
        """Charge le modèle ML au démarrage"""
        self.model = load_model()

    # ─── PRÉDICTION PRINCIPALE ────────────────────────────────────────────
    def predict_from_values(
        self,
        glucose:           float,
        bmi:               float,
        age:               int,
        blood_pressure:    float = 72.0,
        pregnancies:       int   = 0,
        skin_thickness:    float = 20.0,
        insulin:           float = 80.0,
        diabetes_pedigree: float = 0.5
    ) -> dict:
        """
        Fait une prédiction à partir des valeurs numériques du patient.

        Paramètres :
          glucose           : glycémie à jeun (mg/dL) — ex: 140
          bmi               : IMC — ex: 29.5
          age               : âge en années — ex: 45
          blood_pressure    : tension diastolique (mmHg) — ex: 80
          pregnancies       : nombre grossesses (0 pour hommes)
          skin_thickness    : épaisseur pli cutané (mm) — ex: 20
          insulin           : insuline sérique (µU/mL) — ex: 80
          diabetes_pedigree : score génétique (0.0 → 2.5) — ex: 0.5

        Retourne :
          dict avec :
            prediction  : "diabétique" ou "non diabétique"
            risk_level  : "ÉLEVÉ", "MODÉRÉ" ou "FAIBLE"
            risk_score  : pourcentage de risque (0-100%)
            message     : résumé complet pour le médecin
        """
        # ── Modèle non disponible ──────────────────────────────────────
        if self.model is None:
            return {
                "prediction":  "indisponible",
                "risk_level":  "inconnu",
                "risk_score":  0.0,
                "message": (
                    "⚠️ Modèle ML non disponible.\n"
                    "Vérifie ta connexion internet et relance l'application.\n"
                    "Commande : pip install huggingface_hub joblib scikit-learn"
                )
            }

        # ── Conversion glycémie g/L → mg/dL si nécessaire ─────────────
        # (les dossiers patients tunisiens utilisent g/L)
        if glucose < 20.0:
            glucose_mgdl = glucose * 100.0
        else:
            glucose_mgdl = glucose

        # ── Validation des valeurs ─────────────────────────────────────
        glucose_mgdl   = max(0.0, min(500.0, glucose_mgdl))
        bmi            = max(10.0, min(70.0, bmi))
        age            = max(1,    min(120,  age))
        blood_pressure = max(0.0,  min(200.0, blood_pressure))
        pregnancies    = max(0,    min(20,    pregnancies))
        skin_thickness = max(0.0,  min(100.0, skin_thickness))
        insulin        = max(0.0,  min(900.0, insulin))
        diabetes_pedigree = max(0.0, min(2.5, diabetes_pedigree))

        # ── Construction du vecteur de features ───────────────────────
        # Ordre exact attendu par le modèle Pima Indians :
        # [Pregnancies, Glucose, BloodPressure, SkinThickness,
        #  Insulin, BMI, DiabetesPedigreeFunction, Age]
        features = np.array([[
            pregnancies,
            glucose_mgdl,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]])

        # ── Prédiction ────────────────────────────────────────────────
        try:
            prediction_raw = self.model.predict(features)[0]
            probabilities  = self.model.predict_proba(features)[0]
            risk_score     = round(float(probabilities[1]) * 100, 1)

        except Exception as e:
            return {
                "prediction":  "erreur",
                "risk_level":  "inconnu",
                "risk_score":  0.0,
                "message":     f"⚠️ Erreur lors de la prédiction : {str(e)}"
            }

        # ── Interprétation du résultat ────────────────────────────────
        if prediction_raw == 1:
            prediction = "diabétique"
            if risk_score >= 75:
                risk_level = "TRÈS ÉLEVÉ"
                emoji      = "🔴"
            else:
                risk_level = "ÉLEVÉ"
                emoji      = "🟠"
        else:
            prediction = "non diabétique"
            if risk_score >= 40:
                risk_level = "MODÉRÉ"
                emoji      = "🟡"
            else:
                risk_level = "FAIBLE"
                emoji      = "🟢"

        # ── Message complet pour le médecin ───────────────────────────
        message = (
            f"{emoji} Prédiction ML : patient {prediction}\n"
            f"   Niveau de risque : {risk_level}\n"
            f"   Score de risque  : {risk_score}%\n"
            f"\n"
            f"   Données utilisées :\n"
            f"   • Glycémie       : {glucose_mgdl:.0f} mg/dL\n"
            f"   • IMC            : {bmi} kg/m²\n"
            f"   • Âge            : {age} ans\n"
            f"   • Tension dias.  : {blood_pressure:.0f} mmHg\n"
            f"   • Insuline       : {insulin:.0f} µU/mL\n"
            f"   • Score génét.   : {diabetes_pedigree}\n"
            f"\n"
            f"   ⚠️ Cette prédiction est indicative.\n"
            f"   Elle ne remplace pas le diagnostic clinique du médecin."
        )

        return {
            "prediction":  prediction,
            "risk_level":  risk_level,
            "risk_score":  risk_score,
            "message":     message
        }

    # ─── MÉTHODE PRINCIPALE (utilisée par l'agent) ────────────────────────
    def run(self, patient_data: dict) -> str:
        """
        Point d'entrée principal utilisé par l'agent.
        Accepte un dictionnaire avec les données cliniques du patient.

        Paramètres :
          patient_data : dict avec les clés :
            - glucose           (obligatoire)
            - bmi               (obligatoire)
            - age               (obligatoire)
            - blood_pressure    (optionnel — défaut: 72)
            - pregnancies       (optionnel — défaut: 0)
            - skin_thickness    (optionnel — défaut: 20)
            - insulin           (optionnel — défaut: 80)
            - diabetes_pedigree (optionnel — défaut: 0.5)

        Retourne :
          str : message de prédiction formaté
        """
        if not patient_data:
            return (
                "⚠️ Aucune donnée patient fournie pour la prédiction.\n"
                "Utilise le formulaire dans la barre latérale de l'interface."
            )

        result = self.predict_from_values(
            glucose           = float(patient_data.get("glucose",           120)),
            bmi               = float(patient_data.get("bmi",                25)),
            age               = int(patient_data.get("age",                  40)),
            blood_pressure    = float(patient_data.get("blood_pressure",     72)),
            pregnancies       = int(patient_data.get("pregnancies",           0)),
            skin_thickness    = float(patient_data.get("skin_thickness",     20)),
            insulin           = float(patient_data.get("insulin",            80)),
            diabetes_pedigree = float(patient_data.get("diabetes_pedigree", 0.5)),
        )
        return result["message"]

    # ─── MÉTHODE UTILITAIRE ───────────────────────────────────────────────
    def is_available(self) -> bool:
        """Vérifie si le modèle ML est disponible"""
        return self.model is not None

    def get_model_info(self) -> str:
        """Retourne des informations sur le modèle"""
        if self.model is None:
            return "❌ Modèle ML non disponible"

        model_type = type(self.model).__name__
        return (
            f"✅ Modèle ML chargé\n"
            f"   Type    : {model_type}\n"
            f"   Source  : Oduwo/Diabetes_assessment_Model (HuggingFace)\n"
            f"   Dataset : Pima Indians Diabetes\n"
            f"   Features: 8 (glucose, IMC, âge, tension, insuline...)"
        )