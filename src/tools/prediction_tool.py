

import os
import re
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

_model = None

def load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from huggingface_hub import hf_hub_download
        import joblib
        print("⏳ Chargement modèle ML diabète depuis HuggingFace...")
        model_path = hf_hub_download(
            repo_id="Oduwo/Diabetes_assessment_Model",
            filename="model.pkl"
        )
        _model = joblib.load(model_path)
        print("✅ Modèle ML diabète chargé")
    except Exception as e:
        print(f"⚠️ Modèle ML non disponible : {e}")
        _model = None
    return _model


class PredictionTool:
    """
    Outil de prédiction pour 3 pathologies.
    Compatible avec le format exact de tes PDFs patients.
    """

    name = "risk_prediction"
    description = (
        "Prédit le risque pour diabète, hypertension ou cancer "
        "selon les données cliniques du patient."
    )

    def __init__(self, patients_dir="data/patients"):
        self.model = load_model()
        self.patients_dir = Path(patients_dir)
        self._patients_cache = None

   
    
    def _load_all_patients(self) -> List[Dict]:
        """Lit tous les PDFs patients et extrait les données."""
        if self._patients_cache is not None:
            return self._patients_cache

        patients = []

        if not self.patients_dir.exists():
            return patients

        try:
            from pypdf import PdfReader
        except ImportError:
            print("⚠️ pypdf non installé : pip install pypdf")
            return patients

        for pdf_path in sorted(self.patients_dir.glob("*.pdf")):
            try:
                reader = PdfReader(str(pdf_path))
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

                patient = self._extract_patient_data(text, pdf_path.name)
                if patient and patient.get("nom"):
                    patients.append(patient)
            except Exception as e:
                print(f"⚠️ Erreur {pdf_path.name}: {e}")

        self._patients_cache = patients
        print(f"📊 {len(patients)} patients chargés")
        return patients

    def _extract_patient_data(self, text: str, filename: str) -> Dict:
        """
        Extrait les données d'un patient depuis le texte PDF.
        FIX: Compatible avec le format exact de tes PDFs :
          - 'Nom complet: Jacques VINCENT'
          - 'Tension arterielle: 152/86 mmHg'
          - 'Glycemie a jeun: 194 mg/dL'
          - 'HbA1c: 7.3 %'
          - 'Tabagisme: Oui (39 PA)'
        """

     
        nom_match = re.search(
            r"Nom complet:\s*([A-Za-zÀ-ÿ]+)\s+([A-Za-zÀ-ÿ]+)",
            text
        )
        nom = (
            f"{nom_match.group(1)} {nom_match.group(2)}"
            if nom_match else filename.replace(".pdf", "")
        )

      
        age_match = re.search(r"\((\d+)\s*ans\)", text)
        if not age_match:
            age_match = re.search(r"(\d+)\s*ans", text)
        age = int(age_match.group(1)) if age_match else None

        if re.search(r"Hypertension", text, re.IGNORECASE):
            pathologie = "HTA"
        elif re.search(r"Diab[eè]te?\s+de\s+type\s+1", text, re.IGNORECASE):
            pathologie = "Diabete_type1"
        elif re.search(r"Diab[eè]te?", text, re.IGNORECASE):
            pathologie = "Diabete_type2"
        elif re.search(r"Cancer", text, re.IGNORECASE):
            pathologie = "Cancer"
        else:
            pathologie = "Inconnue"

       
        ta_match = re.search(
            r"Tension\s+art[eé]rielle\s*:\s*(\d+)/(\d+)",
            text
        )
        ta_sys = int(ta_match.group(1)) if ta_match else None
        ta_dia = int(ta_match.group(2)) if ta_match else None

     
        imc_match = re.search(r"IMC:\s*(\d+[,.]?\d*)", text)
        imc = (
            float(imc_match.group(1).replace(",", "."))
            if imc_match else None
        )

        gly_match = re.search(
            r"Glyc[eé]mie[^:]*:\s*(\d+[,.]?\d*)\s*(g/L|mg/dL)",
            text
        )
        if gly_match:
            glycemie = float(gly_match.group(1).replace(",", "."))
            if gly_match.group(2) == "g/L":
                glycemie = glycemie * 100 
        else:
            glycemie = None

       
        hba1c_match = re.search(r"HbA1c:\s*(\d+[,.]?\d*)\s*%", text)
        hba1c = (
            float(hba1c_match.group(1).replace(",", "."))
            if hba1c_match else None
        )

        
        tabac_match = re.search(r"Tabagisme:\s*Oui\s*\((\d+)\s*PA\)", text)
        tabac = int(tabac_match.group(1)) if tabac_match else 0

       
        bpco = bool(re.search(r"BPCO", text, re.IGNORECASE))

     
        dossier_match = re.search(r"(P-\d{8})", text)
        numero_dossier = dossier_match.group(1) if dossier_match else None

        return {
            "nom":            nom,
            "age":            age,
            "pathologie":     pathologie,
            "ta_sys":         ta_sys,
            "ta_dia":         ta_dia,
            "imc":            imc,
            "glycemie":       glycemie,
            "hba1c":          hba1c,
            "tabac":          tabac,
            "bpco":           bpco,
            "numero_dossier": numero_dossier,
            "filename":       filename
        }

    def refresh_cache(self):
        """Force le rechargement des PDFs."""
        self._patients_cache = None
        return self._load_all_patients()


    def run(self, input_data) -> str:
        """
        Point d'entrée principal.
        Accepte un dict ou une str (ex: 'glucose=185 age=53 imc=29.7').
        """
        if isinstance(input_data, str):
            data = self._parse_string(input_data)
            if not data:
                return self._help()
        elif isinstance(input_data, dict):
            data = {k.lower(): v for k, v in input_data.items()}
        else:
            return "⚠️ Format non reconnu. Utilise un dict ou une chaîne."

     
        patho = data.get("patho", data.get("pathologie", "")).lower()
        if not patho:
            patho = self._detect_patho(data)

        if "diab" in patho:
            return self._predict_diabetes(data)
        elif "hta" in patho or "hypert" in patho:
            return self._predict_hta(data)
        elif "cancer" in patho or "oncol" in patho:
            return self._predict_cancer(data)
        else:
            return self._predict_auto(data)


    def _detect_patho(self, data: dict) -> str:
        if "glucose" in data or "hba1c" in data or "glycemie" in data:
            return "diabete"
        elif "ta_sys" in data or "ta_dia" in data:
            return "hta"
        else:
            return "cancer"

    def _predict_auto(self, data: dict) -> str:
        results = []
        if "glucose" in data or "hba1c" in data or "glycemie" in data:
            results.append(self._predict_diabetes(data))
        if "ta_sys" in data:
            results.append(self._predict_hta(data))
        if not results:
            results.append(self._predict_cancer(data))
        return "\n\n".join(results)

    
    def _predict_diabetes(self, data: dict) -> str:
       
        glucose     = float(data.get("glucose", data.get("glycemie", 120)))
        bmi         = float(data.get("bmi", data.get("imc", 25)))
        age         = int(data.get("age", 40))
        bp          = float(data.get("ta_dia", data.get("blood_pressure", 72)))
        pregnancies = int(data.get("pregnancies", data.get("grossesses", 0)))
        insulin     = float(data.get("insulin", data.get("insuline", 80)))
        dpf         = float(data.get("diabetes_pedigree", 0.5))
        hba1c       = float(data.get("hba1c", 0))
        nom         = data.get("nom", "Patient")

       
        if glucose < 20:
            glucose = glucose * 100

        
        if glucose < 100:
            gly_eval = "🟢 Normal"
        elif glucose < 126:
            gly_eval = "🟡 Pré-diabète (100–125 mg/dL)"
        else:
            gly_eval = "🔴 Diabète probable (≥126 mg/dL)"

       
        hba1c_line = ""
        if hba1c > 0:
            if hba1c < 5.7:
                hba1c_line = f"   HbA1c           : 🟢 {hba1c}% — Normal\n"
            elif hba1c < 6.5:
                hba1c_line = f"   HbA1c           : 🟡 {hba1c}% — Pré-diabète\n"
            elif hba1c < 7.0:
                hba1c_line = f"   HbA1c           : 🟢 {hba1c}% — Bien contrôlé\n"
            elif hba1c < 8.0:
                hba1c_line = f"   HbA1c           : 🟠 {hba1c}% — Contrôle modéré\n"
            else:
                hba1c_line = f"   HbA1c           : 🔴 {hba1c}% — Mal contrôlé\n"

        ml_lines = ""
        if self.model is not None:
            try:
                features = np.array([[
                    pregnancies, glucose, bp, 20,
                    insulin, bmi, dpf, age
                ]])
                pred  = self.model.predict(features)[0]
                proba = self.model.predict_proba(features)[0]
                score = round(float(proba[1]) * 100, 1)

                if pred == 1:
                    emoji  = "🔴" if score >= 70 else "🟠"
                    result = "diabétique"
                    niveau = "ÉLEVÉ" if score >= 70 else "MODÉRÉ"
                else:
                    emoji  = "🟢"
                    result = "non diabétique"
                    niveau = "FAIBLE"

                ml_lines = (
                    f"\n   🤖 Prédiction ML    : {emoji} {result}\n"
                    f"   📊 Score de risque  : {score}%\n"
                    f"   ⚠️  Niveau           : {niveau}"
                )
            except Exception as e:
                ml_lines = f"\n   ⚠️ Erreur ML : {e}"
        else:
            ml_lines = "\n   ⚠️ Modèle ML non chargé — évaluation clinique seule"

        return (
            f"🩸 PRÉDICTION DIABÈTE — {nom}\n"
            f"{'='*45}\n"
            f"   Glycémie à jeun  : {glucose:.0f} mg/dL — {gly_eval}\n"
            f"{hba1c_line}"
            f"   IMC              : {bmi} kg/m²\n"
            f"   Âge              : {age} ans\n"
            f"{ml_lines}\n"
            f"\n   ⚠️ Ne remplace pas le diagnostic clinique du médecin."
        )

   
    def _predict_hta(self, data: dict) -> str:
        ta_sys = float(data.get("ta_sys", data.get("tension_sys", 120)))
        ta_dia = float(data.get("ta_dia", data.get("tension_dia", 80)))
        age    = int(data.get("age", 50))
        imc    = float(data.get("bmi", data.get("imc", 25)))
        tabac  = float(data.get("tabac", 0))
        nom    = data.get("nom", "Patient")

        
        if ta_sys < 120 and ta_dia < 80:
            ta_classe = "🟢 Optimale"
        elif ta_sys < 130 and ta_dia < 85:
            ta_classe = "🟢 Normale"
        elif ta_sys < 140 or ta_dia < 90:
            ta_classe = "🟡 Normale haute"
        elif ta_sys < 160 or ta_dia < 100:
            ta_classe = "🟠 HTA grade 1"
        elif ta_sys < 180 or ta_dia < 110:
            ta_classe = "🔴 HTA grade 2"
        else:
            ta_classe = "🔴🔴 HTA grade 3 — URGENCE HYPERTENSIVE"

       
        score_cv  = 0
        score_cv += max(0, (age - 40)) * 0.08
        score_cv += max(0, (ta_sys - 120)) * 0.025
        if tabac > 0:
            score_cv += min(tabac * 0.05, 3.0)
        if imc > 30:
            score_cv += (imc - 30) * 0.1
        score_cv = max(0, round(score_cv, 1))

        if score_cv < 2.5:
            risque = "🟢 FAIBLE (<2.5%)"
        elif score_cv < 5:
            risque = "🟡 MODÉRÉ (2.5–5%)"
        elif score_cv < 10:
            risque = "🟠 ÉLEVÉ (5–10%)"
        else:
            risque = "🔴 TRÈS ÉLEVÉ (>10%)"

        if ta_sys < 130:
            reco = "Surveillance + hygiène de vie"
        elif ta_sys < 140:
            reco = "Envisager monothérapie si risque élevé"
        elif ta_sys < 160:
            reco = "💊 Monothérapie antihypertensive recommandée"
        elif ta_sys < 180:
            reco = "💊💊 Bithérapie antihypertensive recommandée"
        else:
            reco = "⚠️ Trithérapie urgente — risque AVC immédiat"

        return (
            f"❤️  PRÉDICTION HTA — {nom}\n"
            f"{'='*45}\n"
            f"   Tension artérielle : {ta_sys:.0f}/{ta_dia:.0f} mmHg\n"
            f"   Classification ESC : {ta_classe}\n"
            f"   IMC                : {imc} kg/m²\n"
            f"   Tabagisme          : {tabac} PA\n"
            f"   Âge                : {age} ans\n"
            f"\n"
            f"   📊 Score CV SCORE2  : {score_cv}%\n"
            f"   ⚠️  Niveau de risque : {risque}\n"
            f"   💊 Recommandation   : {reco}\n"
            f"\n   ⚠️ Ne remplace pas le diagnostic clinique du médecin."
        )

   
    def _predict_cancer(self, data: dict) -> str:
        age   = int(data.get("age", 50))
        imc   = float(data.get("bmi", data.get("imc", 25)))
        tabac = float(data.get("tabac", 0))
        bpco  = bool(data.get("bpco", False))
        nom   = data.get("nom", "Patient")

        score    = 0
        facteurs = []

        if tabac >= 30:
            score += 4
            facteurs.append(f"🔴 Tabagisme élevé ({tabac} PA) — risque ×15")
        elif tabac >= 15:
            score += 2
            facteurs.append(f"🟠 Tabagisme modéré ({tabac} PA) — risque ×5")
        elif tabac > 0:
            score += 1
            facteurs.append(f"🟡 Tabagisme faible ({tabac} PA)")
        else:
            facteurs.append("🟢 Non fumeur")

        if age >= 60:
            score += 2
            facteurs.append(f"🟠 Âge ≥60 ans ({age} ans)")
        elif age >= 45:
            score += 1
            facteurs.append(f"🟡 Âge 45–60 ans ({age} ans)")
        else:
            facteurs.append(f"🟢 Âge <45 ans ({age} ans)")

        if imc >= 35:
            score += 2
            facteurs.append(f"🟠 Obésité sévère (IMC {imc} ≥35)")
        elif imc >= 30:
            score += 1
            facteurs.append(f"🟡 Obésité (IMC {imc} ≥30)")
        else:
            facteurs.append(f"🟢 IMC normal ({imc})")

        if bpco:
            score += 2
            facteurs.append("🔴 BPCO — risque cancer poumon ×5")

        if score <= 2:
            risque = "🟢 FAIBLE"
            surv   = "Dépistage standard recommandé"
        elif score <= 4:
            risque = "🟡 MODÉRÉ"
            surv   = "Dépistage précoce recommandé"
        elif score <= 6:
            risque = "🟠 ÉLEVÉ"
            surv   = "Scanner thoracique annuel recommandé"
        else:
            risque = "🔴 TRÈS ÉLEVÉ"
            surv   = "⚠️ Suivi oncologique rapproché urgent"

        return (
            f"🎗️  PRÉDICTION CANCER — {nom}\n"
            f"{'='*45}\n"
            f"   Âge   : {age} ans\n"
            f"   IMC   : {imc} kg/m²\n"
            f"   Tabac : {tabac} PA\n"
            f"   BPCO  : {'Oui ⚠️' if bpco else 'Non'}\n"
            f"\n"
            f"   📊 Score de risque : {score}/10\n"
            f"   ⚠️  Niveau          : {risque}\n"
            f"   🏥 Surveillance    : {surv}\n"
            f"\n"
            f"   Facteurs identifiés :\n"
            + "\n".join(f"   {f}" for f in facteurs) +
            f"\n\n   ⚠️ Ne remplace pas le diagnostic clinique du médecin."
        )


    def _parse_string(self, text: str) -> dict:
        """Parse une chaîne type 'glucose=185 age=53 imc=29.7'"""
        text = text.lower()
        data = {}
        patterns = {
            "glucose":  r'glucose[=:]\s*(\d+\.?\d*)',
            "glycemie": r'glyc[eé]mie[=:]\s*(\d+\.?\d*)',
            "ta_sys":   r'ta_sys[=:]\s*(\d+\.?\d*)',
            "ta_dia":   r'ta_dia[=:]\s*(\d+\.?\d*)',
            "tension":  r'tension[=:]\s*(\d+)[/\s]*(\d*)',
            "age":      r'age[=:]\s*(\d+\.?\d*)',
            "imc":      r'imc[=:]\s*(\d+\.?\d*)',
            "bmi":      r'bmi[=:]\s*(\d+\.?\d*)',
            "tabac":    r'tabac[=:]\s*(\d+\.?\d*)',
            "hba1c":    r'hba1c[=:]\s*(\d+\.?\d*)',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                if key == "tension":
                    data["ta_sys"] = float(match.group(1))
                    if match.group(2):
                        data["ta_dia"] = float(match.group(2))
                else:
                    data[key] = float(match.group(1))
        return data

    def _help(self) -> str:
        return (
            "## 🔮 PRÉDICTION — Aide\n\n"
            "**Diabète :** `glucose=185 age=53 imc=29.7`\n"
            "**HTA :** `ta_sys=152 ta_dia=86 age=60`\n"
            "**Cancer :** `tabac=42 age=62`\n\n"
            "Ou utilise le formulaire dans la barre latérale."
        )

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> str:
        return (
            f"✅ Prédiction 3 pathologies\n"
            f"   🩸 Diabète      : ML HuggingFace\n"
            f"   ❤️  Hypertension : Score SCORE2 ESC 2023\n"
            f"   🎗️  Cancer       : Score facteurs de risque\n"
            f"   ML disponible  : {'✅' if self.model else '❌'}"
        )