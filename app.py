"""
app.py — Interface Streamlit de l'agent médical
3 pathologies : diabète, hypertension, cancer
Lance avec : streamlit run app.py
"""

import streamlit as st
import sys
import os
import time

# Ajouter le dossier racine au path Python
sys.path.insert(0, os.path.dirname(__file__))

from src.agent import MedicalAgent
from src.ingestion import index_exists, INDEX_PAT, INDEX_MED

# ─── CONFIG PAGE ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agent Médical IA",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* En-tête principal */
    .main-header {
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
    }

    /* Badge outil utilisé */
    .tool-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .badge-patient    { background: #d5f5e3; color: #1e8449; }
    .badge-medical    { background: #fdebd0; color: #935116; }
    .badge-decision   { background: #d6eaf8; color: #1a5276; }
    .badge-prediction { background: #e8daef; color: #6c3483; }
    .badge-aucun      { background: #f2f3f4; color: #717d7e; }

    /* Cartes pathologies sidebar */
    .patho-card {
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 13px;
        font-weight: 500;
    }
    .patho-diabete      { background: #d5f5e3; border-left: 4px solid #1e8449; }
    .patho-hypertension { background: #d6eaf8; border-left: 4px solid #1a5276; }
    .patho-cancer       { background: #e8daef; border-left: 4px solid #6c3483; }

    /* Boîte d'avertissement */
    .warning-box {
        background: #fef9e7;
        border: 1px solid #f0b429;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    /* Boîte urgence */
    .urgent-box {
        background: #fdedec;
        border: 1px solid #e74c3c;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    /* Contexte consulté */
    .context-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.8rem;
        font-family: monospace;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)


# ─── CHARGEMENT AGENT (mis en cache) ─────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Chargement de l'agent médical...")
def load_agent():
    agent = MedicalAgent()
    ok = agent.initialize()
    return agent, ok


agent, agent_ready = load_agent()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:

    # Statut du système
    st.markdown("## ⚙️ Statut du système")
    col1, col2 = st.columns(2)
    with col1:
        if index_exists(INDEX_PAT):
            st.success("✅ Patients")
        else:
            st.error("❌ Patients")
    with col2:
        if index_exists(INDEX_MED):
            st.success("✅ Docs OMS")
        else:
            st.error("❌ Docs OMS")

    if not agent_ready:
        st.error("⚠️ Lance d'abord :\n`python src/ingestion.py`")

    st.divider()

    # Pathologies couvertes
    st.markdown("### 🏥 Pathologies couvertes")
    st.markdown("""
<div class="patho-card patho-diabete">🩸 Diabète — 7 patients</div>
<div class="patho-card patho-hypertension">❤️ Hypertension — 7 patients</div>
<div class="patho-card patho-cancer">🎗️ Cancer — 6 patients</div>
""", unsafe_allow_html=True)

    st.divider()


  

    # Boutons utilitaires
    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        if agent_ready:
            agent.clear_memory()
        st.rerun()

    # Statut agent
    if agent_ready:
       status = agent.get_status()
st.caption(f"🤖 Modèle : {status.get('model', 'mistral')}")
st.caption(f"🔧 Outils actifs : {status.get('tools_active', status.get('tools', 4))}/4")
st.caption(f"💬 Type agent : {status.get('agent_type', 'REACT')}")


# ─── EN-TÊTE PRINCIPAL ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:1.8rem">🩺 Agent Médical IA</h1>
    <p style="margin:0.3rem 0 0; opacity:0.88; font-size:0.95rem">
        Diabète · Hypertension · Cancer &nbsp;|&nbsp;
        4 outils autonomes &nbsp;|&nbsp;
        20 dossiers patients &nbsp;|&nbsp;
        Guidelines OMS/HAS &nbsp;|&nbsp;
        Prédiction ML
    </p>
</div>
""", unsafe_allow_html=True)

# Avertissement médical
st.markdown("""
<div class="warning-box">
⚠️ <strong>Usage clinique uniquement.</strong>
Cet agent aide à la décision médicale mais ne remplace pas
le jugement du médecin. Consultez toujours un spécialiste
pour toute décision thérapeutique.
</div>
""", unsafe_allow_html=True)


# ─── QUESTIONS EXEMPLES ───────────────────────────────────────────────────
with st.expander("💡 Questions exemples — cliquer pour utiliser", expanded=False):
    col1, col2 = st.columns(2)

    questions_diabete = [
        "Quel est l'état du patient Benali Mohamed ?",
        "HbA1c et traitement du patient P-20240003 ?",
        "Quel traitement adapter pour un diabétique HbA1c 8.2% ?",
        "Quels patients ont des complications graves ?",
    ]
    questions_hta = [
        "Quel est l'état de Jacques Vincent (P-20240008) ?",
        "Le traitement de Chaabane est-il adapté ?",
        "Protocole HTA résistante selon OMS ?",
        "Quels patients ont une HVG ?",
    ]
    questions_cancer = [
        "Quel est le protocole de Bernard Claire (cancer sein) ?",
        "Quels patients cancer sont en urgence ?",
        "Protocole ABVD pour lymphome de Hodgkin ?",
        "Résultats biologiques du patient Marzouki ?",
    ]

    with col1:
        st.markdown("**🩸 Diabète**")
        for q in questions_diabete:
            if st.button(q, key=f"d_{q}", use_container_width=True):
                st.session_state.prefill = q

        st.markdown("**❤️ Hypertension**")
        for q in questions_hta:
            if st.button(q, key=f"h_{q}", use_container_width=True):
                st.session_state.prefill = q

    with col2:
        st.markdown("**🎗️ Cancer**")
        for q in questions_cancer:
            if st.button(q, key=f"c_{q}", use_container_width=True):
                st.session_state.prefill = q


# ─── INITIALISATION MESSAGES ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Labels et styles des outils
TOOL_INFO = {
    "patient":    ("🗂️ Dossier patient",    "badge-patient"),
    "medical":    ("📚 Guidelines OMS/HAS", "badge-medical"),
    "decision":   ("🧠 Décision clinique",  "badge-decision"),
    "prediction": ("🤖 Prédiction ML",      "badge-prediction"),
    "aucun":      ("⚙️ Non initialisé",     "badge-aucun"),
}


# ─── AFFICHAGE HISTORIQUE ─────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):

        # Badge outil (messages assistant uniquement)
        if msg["role"] == "assistant" and "tool_used" in msg:
            label, css = TOOL_INFO.get(
                msg["tool_used"],
                ("🔧 Outil", "badge-decision")
            )
            st.markdown(
                f'<span class="tool-badge {css}">{label}</span>',
                unsafe_allow_html=True
            )

        st.markdown(msg["content"])

        # Contexte consulté (si disponible)
        if msg["role"] == "assistant" and "context" in msg and msg["context"]:
            with st.expander("📄 Contexte consulté par l'agent"):
                ctx = msg["context"]
                st.markdown(
                    f'<div class="context-box">'
                    f'{ctx[:2500]}{"..." if len(ctx) > 2500 else ""}'
                    f'</div>',
                    unsafe_allow_html=True
                )


# ─── ZONE DE SAISIE ───────────────────────────────────────────────────────
if not agent_ready:
    st.warning("⚠️ Agent non initialisé. Lance `python src/ingestion.py` d'abord.")
    st.stop()

# Récupérer question pré-remplie (depuis boutons exemples)
prefill    = st.session_state.pop("prefill", "")
user_input = st.chat_input(
    "Posez votre question médicale... (ex: Quel traitement pour le patient Benali ?)"
)

# ─── TRAITEMENT DE LA QUESTION ────────────────────────────────────────────
if user_input or prefill:
    question = user_input or prefill

    # 1. Afficher la question
    st.session_state.messages.append({
        "role":    "user",
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    # 2. Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("🔍 L'agent analyse et choisit ses outils..."):
            start  = time.time()
            result = agent.answer(question)
            elapsed = time.time() - start

        # Badge outil utilisé
        label, css = TOOL_INFO.get(
            result["tool_used"],
            ("🔧 Outil", "badge-decision")
        )
        st.markdown(
            f'<span class="tool-badge {css}">{label}</span>',
            unsafe_allow_html=True
        )

        # Réponse principale
        st.markdown(result["answer"])

        # Métriques
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"⏱️ Temps de réponse : {elapsed:.1f}s")
        with col_b:
            st.caption(f"🔧 Outil utilisé : {label}")

        # Contexte consulté
        if result["context"]:
            with st.expander("📄 Contexte consulté par l'agent"):
                ctx = result["context"]
                st.markdown(
                    f'<div class="context-box">'
                    f'{ctx[:2500]}{"..." if len(ctx) > 2500 else ""}'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # 3. Sauvegarder dans l'historique
        st.session_state.messages.append({
            "role":      "assistant",
            "content":   result["answer"],
            "tool_used": result["tool_used"],
            "context":   result["context"]
        })