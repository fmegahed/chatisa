import os
import json
import time
import warnings
from pathlib import Path

# Suppress Pydantic V1 compatibility warning (LangChain internal - not fixable in user code)
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import requests
from requests.adapters import HTTPAdapter
from typing import Optional


# ---------------------- your config & modules ----------------------

# Must define: OPENAI_API_KEY, OPENAI_REALTIME_MODEL, DEFAULT_REALTIME_VOICE, REALTIME_VOICES,
# APP_NAME, VERSION, DATE, PAGE_CONFIG, THEME_COLORS, PAGES, FEATURES, MODELS,
# validate_api_keys, get_system_info
from config import *

# Local modules (unchanged)
from lib import chatpdf, chatgeneration, sidebar  # noqa: F401
from lib.ui import apply_theme_css
from lib.enhanced_usage_logger import log_page_visit, log_session_action, migrate_from_csv


# -----------------------------------------------------------------------------
# Manage page tracking and associated session state
# -----------------------------------------------------------------------------
THIS_PAGE = "chatgpt"
st.session_state.cur_page = THIS_PAGE
# -----------------------------------------------------------------------------

# ---------------------- Streamlit setup ----------------------
# st.set_page_config(**PAGE_CONFIG)
st.set_page_config(page_title="ChatISA Home", page_icon='assets/favicon.png')
apply_theme_css()

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
        "total_models": len(MODELS),
        "available_models": len(validate_api_keys()["available_models"]),
        "features_enabled": sum(1 for f in FEATURES.values() if f)
    })

    st.markdown(f'<h2 style="color: {THEME_COLORS["primary"]};">Welcome to {APP_NAME}</h2>', unsafe_allow_html=True)
    st.markdown(f"*Version {VERSION} - {DATE}*")

    st.markdown("""
    **Your AI-powered educational assistant** featuring six specialized modules designed to enhance your academic journey.
    Developed at Miami University's Farmer School of Business, ChatISA provides free access to cutting-edge AI technology
    to support student success in programming, projects, exam preparation, interview practice, code execution, and AI model comparison.
    """)

    st.markdown("#### Select a module to continue:")

    # Select the page to switch to (updated for six pages)
    selected = option_menu(
        menu_title=None,
        options=[
            "💻 Coding Companion",
            "📋 Project Coach",
            "📝 Exam Ally",
            "🎤 Interview Mentor",
            "🧪 AI Sandbox",
            "📊 AI Comparisons",
        ],
        icons=["💻", "📋", "📝", "🎤", "🧪", "📊"],
        menu_icon="list",
        default_index=0,
        orientation="horizontal",
    )

    def navigate_to(page_label: str) -> Optional[str]:
        page_map = {
            "💻 Coding Companion": "pages/01_coding_companion.py",
            "📋 Project Coach": "pages/02_project_coach.py",
            "📝 Exam Ally": "pages/03_exam_ally.py",
            "🎤 Interview Mentor": "pages/04_interview_mentor.py",
            "🧪 AI Sandbox": "pages/05_ai_sandbox.py",
            "📊 AI Comparisons": "pages/06_ai_comparisons.py",
        }
        return page_map.get(page_label)

    if st.button(f"Open {selected}", width="stretch"):
        target = navigate_to(selected)
        if target and hasattr(st, "switch_page"):
            st.switch_page(target)

    if selected == '💻 Coding Companion':
        st.info("""
            The coding companion helps with programming questions and explanations tailored to your coursework.

            Select a model, ask a question, and export the conversation to PDF if needed.""")
        st.markdown("Use the left navigation to adjust models or switch tools.")

    if selected == '📋 Project Coach':
        st.info("""
        The Project Coach supports team projects by taking one of four roles:
          - **Premortem Coach** to anticipate risks and mitigations
          - **Team Structuring Coach** to surface resources and responsibilities
          - **Devil's Advocate** to challenge assumptions
          - **Reflection Coach** to synthesize lessons learned

          Select a model, ask a question, and export the conversation to PDF if needed.""")
        st.markdown("Use the left navigation to adjust models or switch tools.")

    if selected == '📝 Exam Ally':
        st.info("""
        The Exam Ally helps you prepare by generating questions from your uploaded PDF and selected question type.

        Select a model, choose a question type, and export the conversation to PDF if needed.""")
        st.markdown("Use the left navigation to adjust models or switch tools.")

    if selected == '🎤 Interview Mentor':
        st.info("""
        The Interview Mentor generates questions based on a job description and your uploaded resume PDF.

        Select a model, ask a question, and export the conversation to PDF if needed.""")
        st.markdown("Use the left navigation to adjust models or switch tools.")

    if selected == '🧪 AI Sandbox':
        st.info("""
        The AI Sandbox provides a secure Python code execution environment for problem-solving and data analysis.
          - Upload data files (CSV, Excel, etc.) and analyze them
          - Create charts and visualizations
          - Perform calculations with step-by-step explanations

          Enter a prompt, upload files, and review results in a notebook-style interface.""")
        st.markdown("Use the left navigation to adjust models or switch tools.")

    if selected == '📊 AI Comparisons':
        st.info("""
        The AI Comparisons tool lets you compare multiple model responses side-by-side.

        Upload images or PDFs to see how different models interpret visual content.""")
        st.markdown("Use the left navigation to adjust models or switch tools.")

    st.markdown("---")

    key_validation = validate_api_keys()
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Models", len(MODELS))
    with c2: st.metric("Available Modules", "6")
    with c3:
        status = "Ready" if key_validation["all_keys_present"] else "Limited"
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
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    sidebar.render_sidebar()



# ---------------------- run ----------------------
migrate_from_csv()
system_info = get_system_info()

if system_info["missing_api_keys"]:
    st.warning(f"Warning: Missing API keys: {', '.join(system_info['missing_api_keys'])}")

# Call home() directly
home()
