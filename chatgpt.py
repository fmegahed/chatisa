

# About this code:
# ----------------
# This code performs the following tasks:
  # 1. It creates a Streamlit app that allows users to interact with an LLM of their choosing to generate responses to their queries:  
  #   - The app allows users to select from a list of LLMs and input their queries.
  #   - The LLMs can be changed within a session. 
  #   - The app displays the conversation between the user and the LLM.
  #   - The app allows users to export the conversation to a PDF.
  # 2. Students can intereact with the app, without having to create API keys for each LLM.
  # 3. The app is designed to be used for educational purposes only.
# -----------------------------------------------------------------------------


# Import required libraries:
# --------------------------
import os # Standard Library
from collections import OrderedDict

# Our Own Modules
from lib import chatpdf, chatgeneration, sidebar

# Third-Party Libraries
from dotenv import load_dotenv

import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page

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
}

for key, page in pages.items():
  if page['page_name'] in new_page_names:
    page['page_name'] = new_page_names[page['page_name']]


# Models:
# -------
models = [
  'gpt-4-turbo-preview', 'gpt-3.5-turbo', 
  'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307',
  'command-r-plus',
  'llama3-8b-8192', 'llama3-70b-8192',
  'gemma-7b-it'
  ]
# -----------------------------------------------------------------------------


# Load Environment Variables:
# ---------------------------
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
# -----------------------------------------------------------------------------


# The Streamlit Interface:
# ------------------------

# Streamlit Text
# ----------------
st.set_page_config(page_title = "ChatISA", layout = "centered",page_icon='🤖')

st_lottie('https://lottie.host/49ad1924-ffe8-4fc0-895c-78fb5a5c8223/wsQgGsWJuV.json', speed=1, key='welcome',loop=True, quality="high", height=100)

st.markdown("## to Your ChatISA Assistant 🤖")

st.markdown("""
ChatISA is your personal, free, and prompt-engineered chatbot, where you can chat with one of nine LLMs.
The chatbot consists of **four main pages:** (a) Coding Companion, (b) Project Coach, (c) Exam Ally, and (d) Interview Mentor.

They can be accessed by clicking on the buttons below or by toggling their names on the sidebar.
""")

st.markdown("#### Select one of the following options to start chatting!")

# Select the page to switch to:
# -----------------------------
# Based on https://github.com/jiatastic/GPTInterviewer/blob/main/Homepage.py
selected = option_menu(
        menu_title= None,
        options=["Coding Companion", "Project Coach", "Exam Ally", "Interview Mentor"],
        icons = ["filetype-py", "kanban", "list-task", "briefcase"],
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

# -----------------------------------------------------------------------------
# Render the sidebar
# -----------------------------------------------------------------------------
sidebar.render_sidebar()
# -----------------------------------------------------------------------------
