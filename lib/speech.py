import tempfile
import os
import streamlit as st
import openai


# Speech-to-Text Function (Using OpenAI Whisper)
def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name
    
    with open(temp_audio_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
    
    os.remove(temp_audio_path)
    return transcript.text

# Text-to-Speech Function (Using OpenAI TTS)
def generate_speech(response_text):
    response_audio = openai.audio.speech.create(
        model="tts-1",
        voice=st.session_state.chosen_voice,
        input=response_text
    )
    return response_audio.content
