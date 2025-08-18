import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

# Local modules (keep as in your project)
from lib import chatpdf, chatgeneration, sidebar  # noqa: F401  (imported for side effects / future use)

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="ChatISA", layout="centered", page_icon="🤖")

# Load environment variables from a local .env (for local/dev). On Streamlit Cloud,
# prefer st.secrets and do NOT commit .env files.
load_dotenv()  # keep this if you rely on .env locally

# -----------------------------------------------------------------------------
# Declare pages with the NEW navigation API so they're always registered.
# -----------------------------------------------------------------------------

coding = st.Page("pages/01_coding_companion.py", title="Coding Companion", icon="🖥")
coach  = st.Page("pages/02_project_coach.py",   title="Project Coach",    icon="🎯")
exam   = st.Page("pages/03_exam_ally.py",       title="Exam Ally",        icon="📝")
mentor = st.Page("pages/04_interview_mentor.py",title="Interview Mentor", icon="👔")


def home():
    """Home page content (callable page)."""
    st.session_state.cur_page = "chatgpt"

    st.markdown("## Welcome to Your ChatISA Assistant 🤖")
    st.markdown(
        """
        ChatISA is your personal, prompt-engineered chatbot where you can chat with one of several LLMs.
        The chatbot consists of **four main pages:** (a) Coding Companion, (b) Project Coach, (c) Exam Ally, and (d) Interview Mentor.

        Use the menu below or the sidebar to navigate.
        """
    )

    st.markdown("#### Select one of the following options to start:")

    selected = option_menu(
        menu_title=None,
        options=["Coding Companion", "Project Coach", "Exam Ally", "Interview Mentor"],
        icons=["filetype-py", "kanban", "list-task", "briefcase"],
        menu_icon="list",
        default_index=0,
        orientation="horizontal",
    )

    # Helper to switch to a registered page (use Page object to avoid path issues)
    def go(target_page: st.Page) -> None:
        st.switch_page(target_page)

    if selected == "Coding Companion":
        st.info(
            """
            🧑‍💻 **Coding Companion** helps you debug, document, and scaffold code.
            Choose your preferred model, ask for code, and export the conversation to PDF.
            """
        )
        if st.button("Go to Coding Companion"):
            go(coding)

    elif selected == "Project Coach":
        st.info(
            """
            🎯 **Project Coach** supports project planning and research assistance.
            Upload context, iterate on deliverables, and keep citations handy.
            """
        )
        if st.button("Go to Project Coach"):
            go(coach)

    elif selected == "Exam Ally":
        st.info(
            """
            📚 **Exam Ally** generates practice questions from PDFs you provide.
            Select question types, review answers, and export your session to PDF.
            """
        )
        if st.button("Go to Exam Ally"):
            go(exam)

    elif selected == "Interview Mentor":
        st.info(
            """
            💼 **Interview Mentor** crafts interview questions using (a) your job description
            and (b) your résumé PDF. Practice answers and track progress.
            """
        )
        if st.button("Go to Interview Mentor"):
            go(mentor)

    # Sidebar (as in your previous app)
    sidebar.render_sidebar()


# Register a callable "Home" page so it shows in the nav and can be switched to
home_page = st.Page(home, title="🏠 Home", icon="🤖")

# Build navigation and run the selected page
nav = st.navigation([home_page, coding, coach, exam, mentor])
nav.run()


# -----------------------------------------------------------------------------
# Sidebar (unchanged)
# -----------------------------------------------------------------------------
sidebar.render_sidebar()
