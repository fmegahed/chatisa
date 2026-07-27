import streamlit as st

from config import THEME_COLORS


def apply_theme_css(include_code_styles: bool = False) -> None:
    """Apply shared UI styles across pages."""
    css = f"""
<style>
    .stApp {{ background-color: {THEME_COLORS['background']}; }}
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
    .stSidebar {{ background-color: #f8f9fa; }}
    h1, h2, h3 {{ color: {THEME_COLORS['primary']}; }}
    .stInfo {{
        background-color: rgba(195, 20, 45, 0.1);
        border: 1px solid {THEME_COLORS['primary']};
    }}
    .stWarning {{ background-color: rgba(255, 160, 122, 0.2); }}

    .tool-card {{
        background-color: white;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        min-height: 110px;
    }}
    .tool-title {{
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 4px;
    }}
    .tool-subtitle {{
        color: #4b5563;
        font-size: 0.9rem;
        line-height: 1.3;
    }}
</style>
"""

    if include_code_styles:
        css += f"""
<style>
    .code-block {{
        background-color: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 6px;
        padding: 16px;
        margin: 10px 0;
        font-family: "Courier New", monospace;
    }}

    .tool-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin: 5px 0;
        background-color: #0969da;
        color: white;
    }}

    .output-section {{
        background-color: #f8f9fa;
        border-left: 3px solid {THEME_COLORS['primary']};
        padding: 12px;
        margin: 10px 0;
    }}
</style>
"""

    st.markdown(css, unsafe_allow_html=True)
