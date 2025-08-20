import os
import threading
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
import uvicorn

# Import centralized configuration
from config import *

# Local modules
from lib import chatpdf, chatgeneration, sidebar  # noqa: F401
from lib.enhanced_usage_logger import log_page_visit, log_session_action, migrate_from_csv

# Import FastAPI app for embedding
try:
    from realtime_server import app as fastapi_app
    FASTAPI_AVAILABLE = True
except ImportError as e:
    st.error(f"FastAPI server import failed: {e}")
    FASTAPI_AVAILABLE = False

# -----------------------------------------------------------------------------
# App setup with centralized configuration
# -----------------------------------------------------------------------------
st.set_page_config(**PAGE_CONFIG)

# Apply Miami University color scheme
miami_css = f"""
<style>
    .stApp {{
        background-color: {THEME_COLORS['background']};
    }}
    
    .stButton > button {{
        background-color: {THEME_COLORS['primary']};
        color: white;
        border: none;
        border-radius: 4px;
    }}
    
    .stButton > button:hover {{
        background-color: {THEME_COLORS['secondary']};
        color: white;
    }}
    
    .stSelectbox > div > div {{
        background-color: {THEME_COLORS['background']};
        border: 1px solid {THEME_COLORS['primary']};
    }}
    
    .stSidebar {{
        background-color: #f8f9fa;
    }}
    
    h1, h2, h3 {{
        color: {THEME_COLORS['primary']};
    }}
    
    .stInfo {{
        background-color: rgba(195, 20, 45, 0.1);
        border: 1px solid {THEME_COLORS['primary']};
    }}
    
    .stWarning {{
        background-color: rgba(255, 160, 122, 0.2);
    }}
</style>
"""
st.markdown(miami_css, unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# -----------------------------------------------------------------------------
# FastAPI Server Embedding via Threading
# -----------------------------------------------------------------------------
def start_fastapi_server():
    """Start FastAPI server in background thread."""
    try:
        uvicorn.run(
            fastapi_app, 
            host="127.0.0.1", 
            port=5050, 
            log_level="error",
            access_log=False
        )
    except Exception as e:
        st.error(f"Failed to start FastAPI server: {e}")

# Start FastAPI server as background thread (singleton)
if FASTAPI_AVAILABLE and 'fastapi_server_started' not in st.session_state:
    try:
        fastapi_thread = threading.Thread(target=start_fastapi_server, daemon=True)
        fastapi_thread.start()
        st.session_state.fastapi_server_started = True
        st.success("🎤 Speech-to-speech server started successfully")
    except Exception as e:
        st.error(f"Failed to start embedded FastAPI server: {e}")
        st.session_state.fastapi_server_started = False

# Migrate old CSV logs to JSON format if needed
migrate_from_csv()

# Check system configuration on startup
system_info = get_system_info()
if system_info["missing_api_keys"]:
    st.warning(f"⚠️ Missing API keys: {', '.join(system_info['missing_api_keys'])}")

# -----------------------------------------------------------------------------  
# Page routing for compatibility with Streamlit 1.48.1
# -----------------------------------------------------------------------------

# Initialize page routing in session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

def home():
    """Home page content using centralized configuration."""
    st.session_state.cur_page = "chatgpt"
    
    # Log page visit
    log_page_visit("home", {
        "version": VERSION,
        "available_models": len(validate_api_keys()["available_models"]),
        "features_enabled": sum(1 for f in FEATURES.values() if f)
    })

    st.markdown(f"## Welcome to {APP_NAME}")
    st.markdown(f"*Version {VERSION} - {DATE}*")
    
    st.markdown(
        """
        **Your AI-powered educational assistant** featuring five specialized tools designed to enhance your academic journey. 
        Developed at Miami University's Farmer School of Business, ChatISA provides free access to cutting-edge AI technology 
        to support student success in programming, projects, exam preparation, interview practice, and AI model comparison.
        """
    )
    
    # Feature highlights
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("### 💻\n**Coding Companion**\n*Programming help & tutorials*")
    with col2:
        st.markdown("### 🎯\n**Project Coach**\n*Team project guidance*")
    with col3:
        st.markdown("### 📝\n**Exam Ally**\n*Study materials & practice*")
    with col4:
        st.markdown("### 👔\n**Interview Mentor**\n*Speech-based interview prep*")
    with col5:
        st.markdown("### ⚖️\n**AI Comparisons**\n*Compare AI responses side-by-side*")
    
    st.markdown("---")
    
    # System status
    key_validation = validate_api_keys()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Available Models", f"{len(key_validation['available_models'])}/7")
    with col2:
        st.metric("Available Modules", "5")
    with col3:
        status = "🟢 Ready" if key_validation["all_keys_present"] else "🟡 Limited"
        st.metric("System Status", status)

    # Model listing and summary (simplified)
    with st.expander("Available Models & Pricing", expanded=False):
        st.markdown("### Model Summary")
        
        # Create summary table
        model_data = []
        for model_name, config in MODELS.items():
            # Convert costs to per million tokens for display
            input_cost_per_million = config["cost_per_1k_input"] * 1000
            output_cost_per_million = config["cost_per_1k_output"] * 1000
            
            model_data.append({
                "Model": config["display_name"],
                "Provider": config["provider"].title(),
                "Input ($/1M tokens)": f"${input_cost_per_million:.2f}",
                "Output ($/1M tokens)": f"${output_cost_per_million:.2f}",
                "Max Tokens": f"{config['max_tokens']:,}",
                "Vision": "Yes" if config["supports_vision"] else "No",
                "Functions": "Yes" if config["supports_function_calling"] else "No"
            })
        
        df = pd.DataFrame(model_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    if not key_validation["all_keys_present"]:
        with st.expander("⚠️ Setup Required", expanded=False):
            st.markdown(f"""
            **Missing API Keys:** {', '.join(key_validation["missing_keys"])}
            
            To enable all features, add the missing API keys to your `.env` file:
            ```
            OPENAI_API_KEY=your_key_here
            ANTHROPIC_API_KEY=your_key_here
            COHERE_API_KEY=your_key_here
            GROQ_API_KEY=your_key_here
            ```
            """)

    st.markdown("#### Select one of the following options to start:")

    # Build options list dynamically from 4 core pages
    options = []
    icons = []
    descriptions = {}
    
    for page_key, page_config in PAGES.items():
        if page_key in ["coding_companion", "project_coach", "exam_ally", "interview_mentor", "ai_comparisons"]:
            title = page_config["title"]
            options.append(title)
            icons.append(page_config["icon"])
            descriptions[title] = page_config["description"]

    selected = option_menu(
        menu_title=None,
        options=options,
        icons=icons,
        menu_icon="list",
        default_index=0,
        orientation="horizontal",
    )

    # Dynamic page navigation based on selection
    page_mapping = {
        "Coding Companion": "pages/01_coding_companion.py",
        "Project Coach": "pages/02_project_coach.py", 
        "Exam Ally": "pages/03_exam_ally.py",
        "Interview Mentor": "pages/04_interview_mentor.py",
        "AI Comparisons": "pages/05_ai_comparisons.py"
    }

    # Show selected page info and navigation button
    if selected in descriptions:
        st.info(f"**{descriptions[selected]}**")
        
        # Special handling for interview mentor (now speech-based)
        if selected == "Interview Mentor":
            if st.session_state.get('fastapi_server_started', False):
                st.success("🎤 **Speech-to-Speech Feature** - Server running on single port")
            else:
                st.warning("🎤 **Speech-to-Speech Feature** - Server startup failed")
        
        # Special handling for AI model comparison (experimental)
        if selected == "AI Comparisons":
            st.warning("🧪 **Experimental Feature** - Compare AI model responses side-by-side with support for text, images, and PDFs")
        
        if st.button(f"Go to {selected}", type="primary"):
            st.session_state.current_page = page_mapping[selected]
            st.rerun()

    # Sidebar (as in your previous app)
    sidebar.render_sidebar()


# Route to the appropriate page
def route_page():
    """Route to the appropriate page based on session state."""
    current_page = st.session_state.get("current_page", "home")
    
    if current_page == "home":
        home()
    elif current_page.endswith('.py'):
        # For Streamlit pages, we need to use st.switch_page
        st.switch_page(current_page)
    else:
        home()

# Run the appropriate page
route_page()
