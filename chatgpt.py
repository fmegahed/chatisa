

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

# Our Own Modules
from lib import chatpdf, chatgeneration

# Third-Party Libraries
from dotenv import load_dotenv

import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page

# -----------------------------------------------------------------------------

# Change page names within the file:
# ----------------------------------
# Based on https://stackoverflow.com/a/74418483
pages = st.source_util.get_pages('chatgpt.py')
new_page_names = {
  'chatgpt': '🤖 ChatISA',
  'coding_companion': '🖥 Coding Companion',
  'project_coach': '👩‍🏫 Project Coach'
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

# Streamlit Title:
# ----------------

st.set_page_config(page_title = "ChatISA", layout = "centered",page_icon='🤖')
st_lottie('https://lottie.host/49ad1924-ffe8-4fc0-895c-78fb5a5c8223/wsQgGsWJuV.json', speed=1, key='welcome',loop=True, quality="high", height=100)
st.markdown("## to Your ChatISA Assistant 🤖")
# st.markdown("""
# ChatISA is your personal, free, and prompt-engineered chatbot, where you can chat with one of nine LLMs.
# The chatbot consists of two main pages:
#   1. **Coding Companion:** A chatbot that helps you with coding-related questions, taking into account your educational background and coding sytles used at Miami University.  
#   2. **Project Coach:** A chatbot that helps you with project-related questions, where the AI can take one of four roles:  
#       - **Premortem Coach** to help the team perform a project premortem by encouraging them to envision possible failures and how to avoid them.  
#       - **Team Structuring Coach** to help the team recognize and make use of the resources and expertise within the team.  
#       - **Devil's Advocate** to challenge your ideas and assumptions at various stages of your project.  
#       - **Reflection Coach** to assist the team in reflecting on their experiences in a structured way to derive lessons and insights.
# 
# For each page, you can select the model you want to chat with, input your query, and view the conversation. You can also export the entire conversation to a PDF.
# """)
st.markdown("""
ChatISA is your personal, free, and prompt-engineered chatbot, where you can chat with one of nine LLMs.
The chatbot consists of **two main pages:** (a) Coding Companion, and (b) Project Coach. 

They can be accessed by clicking on the buttons below or by toggling their names on the sidebar.
""")

st.markdown("#### Select one of the following options to start chatting!")

# Select the page to switch to:
# -----------------------------
# Based on https://github.com/jiatastic/GPTInterviewer/blob/main/Homepage.py
selected = option_menu(
        menu_title= None,
        options=["Coding Companion", "Project Coach"],
        icons = ["filetype-py", "kanban"],
        menu_icon="list",
        default_index=0,
        orientation="horizontal",
    )
if selected == 'Coding Companion':
    st.info("""
        📚 The coding companion can help you with coding-related questions, taking into account your educational background and coding sytles used at Miami University. 
        
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


# Sidebar Markdown:
# -----------------
st.sidebar.markdown("""
### Mantained By 
  - [Fadel Megahed](https://miamioh.edu/fsb/directory/?up=/directory/megahefm)   
  - [Joshua Ferris](https://miamioh.edu/fsb/directory/?up=/directory/ferrisj2)

### Version 
  1.2.0 (April 30, 2024)

### Key Features
  - Free to use
  - Chat with one of nine LLMs
  - Export chat to PDF
  
### Support & Funding
  - Farmer School of Business
  - US Bank
""")

# A Button to show/hide Disclaimers and References:
if 'show_info' not in st.session_state:
    st.session_state.show_info = False

# Toggle button
if st.sidebar.button('Toggle Disclaimers & References'):
    st.session_state.show_info = not st.session_state.show_info  # Toggle the state

# Conditionally display the Markdown
if st.session_state.show_info:
    st.sidebar.markdown("""
    ### Disclaimers
    - ChatISA is designed for educational purposes only.
    - If you plan to use ChatISA for classwork, you must obtain approval from your instructor beforehand.
    - Neither Miami University nor the Farmer School of Business can be held responsible for any content generated by this app.
    - Always use ChatISA at your own risk and evaluate the accuracy of the generated answers.

    ### References
    - **Prompt Engineering:** Adapted from [Assigning AI by Mollick and Mollic 2023](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4475995)
    - **Streamlit App:** Adapted from [ChatGPT Apps with Streamlit](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-chatgpt-like-app)
    - **Our Code Repo:** [![Click here to access](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/fmegahed/chatisa)
    """)

# -----------------------------------------------------------------------------
