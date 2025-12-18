import os
import tempfile
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
import streamlit as st
import openai
import pymupdf4llm  # PyMuPDF4LLM

from config import (
    OPENAI_API_KEY,
    OPENAI_REALTIME_MODEL,
    DEFAULT_REALTIME_VOICE,
    REALTIME_VOICES,
    PAGES,  # for max pdf pages
)

# ------------ constants ------------
OPENAI_RT_SESS_URL = "https://api.openai.com/v1/realtime/sessions"
MAX_PDF_PAGES = int(PAGES.get("interview_mentor", {}).get("max_pdf_pages", 6))

# ------------ pooled HTTP session ------------
_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=32, pool_maxsize=128)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

# ------------ session state helpers ------------
def ensure_interview_state_defaults():
    st.session_state.setdefault("submitted_speech", False)
    st.session_state.setdefault("interview_active", False)
    st.session_state.setdefault("interview_submission", {})
    st.session_state.setdefault("_rt_token", None)

def check_realtime_ready() -> bool:
    return bool(OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"))

# ------------ PDF -> markdown (PyMuPDF4LLM) ------------
def pdf_to_markdown(pdf_bytes: bytes) -> str:
    """
    Convert uploaded resume PDF bytes to markdown using PyMuPDF4LLM.
    Only the first MAX_PDF_PAGES pages are processed.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        pages = list(range(MAX_PDF_PAGES))
        return pymupdf4llm.to_markdown(tmp_path, pages=pages)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

# ------------ Realtime: mint ephemeral client_secret ------------
def mint_realtime_client_secret(
    model: Optional[str] = None,
    voice: Optional[str] = None,
    instructions: Optional[str] = None,
) -> dict:
    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")

    model = model or OPENAI_REALTIME_MODEL
    voice = (voice or DEFAULT_REALTIME_VOICE)
    if voice not in REALTIME_VOICES:
        voice = DEFAULT_REALTIME_VOICE

    default_instructions = (
        "You are an expert interview mentor for students. "
        "Speak naturally and professionally. Ask one question at a time. "
        "Keep responses concise but helpful."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
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
        raise RuntimeError(f"OpenAI Realtime error {resp.status_code}: {resp.text}")

    j = resp.json()
    return {
        "client_secret": (j.get("client_secret", {}) or {}).get("value") or j.get("client_secret"),
        "expires_at": j.get("expires_at"),
        "session_id": j.get("id"),
        "model": model,
        "voice": voice,
    }

# ------------ STT / TTS (OpenAI) ------------
def transcribe_audio(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        with open(path, "rb") as f:
            transcript = openai.audio.transcriptions.create(model="whisper-1", file=f)
        return transcript.text
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

def generate_speech(response_text: str) -> bytes:
    response_audio = openai.audio.speech.create(
        model="tts-1",
        voice=st.session_state.get("chosen_voice", DEFAULT_REALTIME_VOICE),
        input=response_text,
    )
    return response_audio.content
