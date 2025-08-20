import os
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import requests
from requests.adapters import HTTPAdapter
from streamlit_extras.switch_page_button import switch_page


# ---------------------- your config & modules ----------------------

# Must define: OPENAI_API_KEY, OPENAI_REALTIME_MODEL, DEFAULT_REALTIME_VOICE, REALTIME_VOICES,
# APP_NAME, VERSION, DATE, PAGE_CONFIG, THEME_COLORS, PAGES, FEATURES, MODELS,
# validate_api_keys, get_system_info
from config import *

# Local modules (unchanged)
from lib import chatpdf, chatgeneration, sidebar  # noqa: F401
from lib.enhanced_usage_logger import log_page_visit, log_session_action, migrate_from_csv


# -----------------------------------------------------------------------------
# Manage page tracking and associated session state
# -----------------------------------------------------------------------------
THIS_PAGE = "chatgpt"
st.session_state.cur_page = THIS_PAGE
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Change page names within the file:
# ----------------------------------
# Based on https://stackoverflow.com/a/74418483
pages = st.source_util.get_pages('chatgpt.py')

# New page names
new_page_names = {
  'chatgpt': '🏠 Home',
  'coding_companion': '🖥 Coding Companion',
  'project_coach': '👩‍🏫 Project Coach',
  'exam_ally': '📝 Exam Ally',
  'interview_mentor': '👔 Interview Mentor',
  'ai_comparisons': '⚖️ AI Comparisons',
}

for key, page in pages.items():
  if page['page_name'] in new_page_names:
    page['page_name'] = new_page_names[page['page_name']]


# ---------------------- Streamlit setup ----------------------
st.set_page_config(**PAGE_CONFIG)

miami_css = f"""
<style>
    .stApp {{ background-color: {THEME_COLORS['background']}; }}
    .stButton > button {{
        background-color: {THEME_COLORS['primary']}; color: white; border: none; border-radius: 4px;
    }}
    .stButton > button:hover {{ background-color: {THEME_COLORS['secondary']}; color: white; }}
    .stSelectbox > div > div {{ background-color: {THEME_COLORS['background']}; border: 1px solid {THEME_COLORS['primary']}; }}
    .stSidebar {{ background-color: #f8f9fa; }}
    h1, h2, h3 {{ color: {THEME_COLORS['primary']}; }}
    .stInfo {{ background-color: rgba(195, 20, 45, 0.1); border: 1px solid {THEME_COLORS['primary']}; }}
    .stWarning {{ background-color: rgba(255, 160, 122, 0.2); }}
</style>
"""

st.markdown(miami_css, unsafe_allow_html=True)

load_dotenv(override=True)


# ---------------------- pooled HTTP session (bursts) ----------------------
_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=32, pool_maxsize=128)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)
OPENAI_RT_SESS_URL = "https://api.openai.com/v1/realtime/sessions"

def mint_realtime_client_secret(model: str, voice: str, instructions: str | None = None) -> dict:
    """
    Server-side mint of ephemeral Realtime session token.
    Returns: {client_secret, expires_at, session_id, model, voice}
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing (set in config.py or .env)")

    default_instructions = (
        "You are a helpful AI assistant for ChatISA, an educational platform. "
        "Speak naturally and professionally. Keep responses clear and concise. "
        "You are designed to help students with their educational needs."
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "voice": voice,
        "modalities": ["audio", "text"],
        "instructions": instructions or default_instructions,
    }
    resp = _session.post(OPENAI_RT_SESS_URL, headers=headers, json=payload, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"OpenAI Realtime session error {resp.status_code}: {resp.text}")

    j = resp.json()
    return {
        "client_secret": (j.get("client_secret", {}) or {}).get("value") or j.get("client_secret"),
        "expires_at": j.get("expires_at"),
        "session_id": j.get("id"),
        "model": model,
        "voice": voice,
    }


# ---------------------- minimal WebRTC launcher (front-end) ----------------------
RTC_HTML = """
<script>
async function startRealtime(clientSecret, model) {
  // Peer connection
  const pc = new RTCPeerConnection();
  const audioEl = document.getElementById("realtime-audio");
  pc.ontrack = (event) => { audioEl.srcObject = event.streams[0]; };

  // Mic
  const ms = await navigator.mediaDevices.getUserMedia({ audio: true });

  for (const track of ms.getTracks()) pc.addTrack(track, ms);

  // Optional data channel (events/control)
  const dc = pc.createDataChannel("oai-events");
  dc.onmessage = (e) => console.log("OAI:", e.data);

  // SDP offer
  const offer = await pc.createOffer({ offerToReceiveAudio: true });
  await pc.setLocalDescription(offer);

  // SDP exchange with OpenAI Realtime (Authorization: Bearer <clientSecret>)
  const sdpResponse = await fetch("https://api.openai.com/v1/realtime?model=" + encodeURIComponent(model), {
    method: "POST",
    headers: {
      "Authorization": "Bearer " + clientSecret,
      "Content-Type": "application/sdp"
    },
    body: offer.sdp
  });

  if (!sdpResponse.ok) {
    const errText = await sdpResponse.text();
    throw new Error("Realtime SDP exchange failed: " + sdpResponse.status + " " + errText);
  }

  const answerSdp = await sdpResponse.text();
  await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });

  // Expose for debugging/stop
  window._oai_pc = pc;
}

async function stopRealtime() {
  const pc = window._oai_pc;

  if (pc) {
    pc.getSenders().forEach(s => { try { s.track && s.track.stop(); } catch(_){} });
    pc.getReceivers().forEach(r => { try { r.track && r.track.stop(); } catch(_){} });
    pc.close();
    window._oai_pc = null;
  }
}
</script>


<div style="display:flex;gap:8px;align-items:center;">
  <button id="startBtn">Start Realtime</button>
  <button id="stopBtn">Stop</button>
  <audio id="realtime-audio" autoplay></audio>
</div>

<script>
(() => {
  const startBtn = document.getElementById("startBtn");
  const stopBtn  = document.getElementById("stopBtn");
  startBtn.onclick = async () => {
    const payload = JSON.parse(decodeURIComponent(window.location.hash.slice(1)));
    await startRealtime(payload.client_secret, payload.model).catch(err => alert(err.message));
  };
  stopBtn.onclick = async () => { await stopRealtime(); };
})();
</script>
"""


# ---------------------- main/home UI ----------------------
def home():
    log_page_visit("home", {
        "version": VERSION,
        "available_models": len(validate_api_keys()["available_models"]),
        "features_enabled": sum(1 for f in FEATURES.values() if f)
    })

    st.markdown(f"## Welcome to {APP_NAME}")
    st.markdown(f"*Version {VERSION} - {DATE}*")

    st.markdown("""
    **Your AI-powered educational assistant** featuring five specialized tools designed to enhance your academic journey. 
    Developed at Miami University's Farmer School of Business, ChatISA provides free access to cutting-edge AI technology 
    to support student success in programming, projects, exam preparation, interview practice, and AI model comparison.
    """)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.markdown("### 💻\n**Coding Companion**\n*Programming help & tutorials*")
    with col2: st.markdown("### 🎯\n**Project Coach**\n*Team project guidance*")
    with col3: st.markdown("### 📝\n**Exam Ally**\n*Study materials & practice*")
    with col4: st.markdown("### 👔\n**Interview Mentor**\n*Speech-based interview prep*")
    with col5: st.markdown("### ⚖️\n**AI Comparisons**\n*Compare AI responses side-by-side*")

    st.markdown("---")

    key_validation = validate_api_keys()
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Available Models", f"{len(key_validation['available_models'])}/7")
    with c2: st.metric("Available Modules", "5")
    with c3:
        status = "🟢 Ready" if key_validation["all_keys_present"] else "🟡 Limited"
        st.metric("System Status", status)

    with st.expander("Available Models & Pricing", expanded=False):
        st.markdown("### Model Summary")
        rows = []
        for _, cfg in MODELS.items():
            rows.append({
                "Model": cfg["display_name"],
                "Provider": cfg["provider"].title(),
                "Input ($/1M tokens)": f"${cfg['cost_per_1k_input']*1000:.2f}",
                "Output ($/1M tokens)": f"${cfg['cost_per_1k_output']*1000:.2f}",
                "Max Tokens": f"{cfg['max_tokens']:,}",
                "Vision": "Yes" if cfg["supports_vision"] else "No",
                "Functions": "Yes" if cfg["supports_function_calling"] else "No",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Select one of the following options to start:")

    # Select the page to switch to (updated for five pages)
    selected = option_menu(
        menu_title=None,
        options=["Coding Companion", "Project Coach", "Exam Ally", "Interview Mentor", "AI Comparisons"],
        icons=["filetype-py", "kanban", "list-task", "briefcase", "arrows-expand"],
        menu_icon="list",
        default_index=0,
        orientation="horizontal",
    )

    if selected == 'Coding Companion':
        st.info("""
            📚 The coding companion can help you with coding-related questions, taking into account your educational background and coding styles used at Miami University. 
            
            Here, you can select the model you want to chat with, input your query, and view the conversation. You can also export the entire conversation to a PDF.""")
        if st.button("Go to Coding Companion"):
            switch_page("🖥 Coding Companion")

    if selected == 'Project Coach':
        st.info("""
        📚 The Project Coach can help you with project-related questions, where the AI can take one of four roles:  
          - **Premortem Coach** to help the team perform a project premortem by encouraging them to envision possible failures and how to avoid them.  
          - **Team Structuring Coach** to help the team recognize and make use of the resources and expertise within the team.  
          - **Devil's Advocate** to challenge your ideas and assumptions at various stages of your project.  
          - **Reflection Coach** to assist the team in reflecting on their experiences in a structured way to derive lessons and insights. 
          
          Here, you can select the model you want to chat with, input your query, and view the conversation. You can also export the entire conversation to a PDF. """
        )
        if st.button("Go to Project Coach"):
            switch_page("👩‍🏫 Project Coach")

    if selected == 'Exam Ally':
        st.info("""
        📚 The Exam Ally can help you prepare for exams by generating exam questions based on information extracted from a PDF that you upload and your choice of exam question type. 
        
        Here, you can select the model you want to chat with and type of exam questions. Note that the LLM grades and feedback can be wrong, so always double-check the answers. You can also export the entire conversation to a PDF.
        
        P.S.: We do not store any of your data on our servers.
        """)
        if st.button("Go to Exam Ally"):
            switch_page("📝 Exam Ally")

    if selected == 'Interview Mentor':
        st.info("""
        📚 The Interview Mentor is designed to help you prepare for technical interviews by generating interview questions based on information extracted from: (a) a job description that you will provide, and (b) a PDF of your resume. 
        
        Here, you can select the model you want to chat with, input your query, and view the conversation. You can also export the entire conversation to a PDF.
        
        P.S.: We do not store any of your data on our servers.
        """)
        if st.button("Go to Interview Mentor"):
            switch_page("👔 Interview Mentor")

    if selected == 'AI Comparisons':
        st.info("""
        📚 The AI Comparisons tool allows you to compare responses from multiple AI models side-by-side for the same query.
        
        Here, you can select multiple models, input your query, and view responses from different models simultaneously. This helps you understand different AI perspectives and choose the best response for your needs.
        
        🧪 **Experimental feature** - Results may vary across models.
        """)
        if st.button("Go to AI Comparisons"):
            switch_page("⚖️ AI Comparisons")

    sidebar.render_sidebar()



# ---------------------- run ----------------------
migrate_from_csv()
system_info = get_system_info()

if system_info["missing_api_keys"]:
    st.warning(f"⚠️ Missing API keys: {', '.join(system_info['missing_api_keys'])}")

# Call home() directly
home()