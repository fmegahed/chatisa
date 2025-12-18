import streamlit as st
from config import (
    APP_NAME, VERSION, DATE, MODELS, DEFAULT_MODELS,
    get_model_display_name, validate_api_keys,
    SIDEBAR_CONFIG, THEME_COLORS
)
from lib.enhanced_usage_logger import log_model_selection, log_session_action


def render_sidebar(skip_navigation=False):
    """Render clean sidebar with essential information.

    Args:
        skip_navigation: If True, skip rendering navigation (useful when caller renders it separately)
    """

    # App information
    key_validation = validate_api_keys()
    available_models = len(key_validation["available_models"])

    if not skip_navigation:
        render_navigation()

    # No app header to maximize sidebar space.

    # Model Selection (if enabled)
    if SIDEBAR_CONFIG.get("show_model_selector", True) and hasattr(st.session_state, 'cur_page'):
        render_model_selector()

    # Simple cost tracking (if enabled)
    if SIDEBAR_CONFIG.get("show_cost_calculator", True):
        render_simple_cost_info()

    # Clear Conversation button for chat-based pages
    render_clear_conversation_button()

    st.sidebar.markdown(f"""
<h3 style="color: {THEME_COLORS['primary']};">System Status</h3>

<div style="background-color: {THEME_COLORS['success']}15; padding: 8px; border-radius: 5px; border-left: 3px solid {THEME_COLORS['success']};">
    <strong>{available_models}</strong> models available
</div>

<h3 style="color: {THEME_COLORS['primary']};">Key Features</h3>

<ul style="padding-left: 20px;">
    <li>Free to use for students</li>
    <li>6 specialized AI modules</li>
    <li>Multiple LLM providers</li>
    <li>Speech-to-speech interview prep</li>
    <li>Model comparison tools</li>
</ul>

<h3 style="color: {THEME_COLORS['primary']};">Support & Funding</h3>

<div style="background-color: {THEME_COLORS['info']}10; padding: 8px; border-radius: 5px; font-size: 0.9em;">
    &bull; Farmer School of Business<br>
    &bull; US Bank Foundation<br>
    &bull; Raymond E. Glos Professorship
</div>

<h3 style="color: {THEME_COLORS['primary']}; margin-top: 20px;">Maintained By</h3>

<ul style="padding-left: 20px;">
    <li><a href="https://miamioh.edu/fsb/directory/?up=/directory/megahefm" style="color: {THEME_COLORS['primary']};">Fadel Megahed</a></li>
    <li><a href="https://miamioh.edu/fsb/directory/?up=/directory/ferrisj2" style="color: {THEME_COLORS['primary']};">Joshua Ferris</a></li>
</ul>
""", unsafe_allow_html=True)

    # Essential system information only
    render_essential_info()



def render_navigation():
    # Render page navigation with icons.
    st.sidebar.markdown(
        "<style>[data-testid='stSidebarNav']{display:none;}</style>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        f'<h3 style="color: {THEME_COLORS["primary"]};">Navigation</h3>',
        unsafe_allow_html=True
    )

    pages = [
        ("chatgpt.py", "🏠 Home"),
        ("pages/01_coding_companion.py", "💻 Coding Companion"),
        ("pages/02_project_coach.py", "📋 Project Coach"),
        ("pages/03_exam_ally.py", "📝 Exam Ally"),
        ("pages/04_interview_mentor.py", "🎤 Interview Mentor"),
        ("pages/05_ai_sandbox.py", "🧪 AI Sandbox"),
        ("pages/06_ai_comparisons.py", "📊 AI Comparisons"),
    ]

    if hasattr(st.sidebar, "page_link"):
        for page_path, label in pages:
            st.sidebar.page_link(page_path, label=label)
    else:
        for _, label in pages:
            st.sidebar.markdown(f"- {label}")

    st.sidebar.markdown("")

def render_model_selector():
    """Render model selection interface with improved grouping."""
    st.sidebar.markdown("")
    st.sidebar.markdown(
        f'<h3 style="color: {THEME_COLORS["primary"]};">Model Selection</h3>',
        unsafe_allow_html=True
    )

    # Get available models for current page
    if hasattr(st.session_state, 'cur_page'):
        page = st.session_state.cur_page
        # Special handling for interview mentor - only show realtime model
        if page == "interview_mentor":
            available_models = ["gpt-4o-realtime-preview-2025-06-03"]
            if "gpt-4o-realtime-preview-2025-06-03" not in MODELS:
                available_models = []
        # Special handling for AI Sandbox - only models with "sandbox" tag (uses Responses API)
        elif page == "ai_sandbox":
            available_models = [
                m for m in MODELS
                if "sandbox" in MODELS[m].get("tags", [])
            ]
        else:
            # Filter available models based on API keys for other pages
            key_validation = validate_api_keys()
            available_models = [
                m for m in key_validation["available_models"]
                if m in MODELS and not MODELS[m].get("realtime_only", False)
            ]
    else:
        available_models = list(MODELS.keys())

    if not available_models:
        st.sidebar.error("No models available. Please check your API keys.")
        return

    # Special message for interview mentor page
    if hasattr(st.session_state, 'cur_page') and st.session_state.cur_page == "interview_mentor":
        st.sidebar.info(
            "Speech Interview Mode\n"
            "Only real-time speech models are available for interview conversations."
        )
    # Special message for AI Sandbox page
    elif hasattr(st.session_state, 'cur_page') and st.session_state.cur_page == "ai_sandbox":
        st.sidebar.info(
            "Only OpenAI API-based models are available (as we use the Responses API with code interpreter)."
        )
    else:
        # Show legend for capability indicators
        st.sidebar.markdown(f"""
<div style="background-color: {THEME_COLORS['info']}10; padding: 6px; border-radius: 4px; font-size: 0.85em; margin-bottom: 10px;">
    <strong>Legend:</strong> Vision | Open Source | Free Tier
</div>
""", unsafe_allow_html=True)

    # Create two-level dropdown: Category -> Model
    from config import MODEL_CATEGORIES

    # Build available categories and their models
    available_categories = {}
    for category_key, category_info in MODEL_CATEGORIES.items():
        category_models = [m for m in category_info['models'] if m in available_models]
        if category_models:
            available_categories[category_key] = {
                'info': category_info,
                'models': category_models
            }

    if not available_categories:
        # Fallback to simple selection if no categories available
        model_options = {get_model_display_name(m): m for m in available_models}
        use_two_level = False
    else:
        use_two_level = True

    # Initialize session state keys
    current_page = getattr(st.session_state, 'cur_page', 'unknown')
    page_category_key = f"model_category_{current_page}"
    page_model_key = f"model_choice_{current_page}"
    page_default = DEFAULT_MODELS.get(current_page)

    if use_two_level:
        # Two-level dropdown: Category -> Model
        if page_category_key not in st.session_state:
            # Find category that contains default model
            default_category = None
            if page_default:
                for cat_key, cat_data in available_categories.items():
                    if page_default in cat_data['models']:
                        default_category = cat_key
                        break
            st.session_state[page_category_key] = default_category or list(available_categories.keys())[0]

        # Category selection dropdown
        category_options = {
            cat_data['info']['display_name']: cat_key
            for cat_key, cat_data in available_categories.items()
        }

        current_category_key = st.session_state[page_category_key]
        current_category_display = available_categories[current_category_key]['info']['display_name']
        category_index = list(category_options.keys()).index(current_category_display)

        selected_category_display = st.sidebar.selectbox(
            "Choose Category:",
            options=list(category_options.keys()),
            index=category_index,
            key=f"category_selector_{current_page}",
            help="Select model category"
        )

        selected_category_key = category_options[selected_category_display]
        if selected_category_key != current_category_key:
            st.session_state[page_category_key] = selected_category_key
            # Reset model choice when category changes
            st.session_state[page_model_key] = available_categories[selected_category_key]['models'][0]

        # Model selection dropdown within selected category
        category_models = available_categories[selected_category_key]['models']
        model_options_in_category = {}

        for model in category_models:
            model_config = MODELS[model]
            display_name = model_config['display_name']

            # Add capability indicators
            indicators = []
            if model_config.get('supports_vision', False):
                indicators.append("[Vision]")
            if model_config.get('open_weight', False):
                indicators.append("[Open]")
            if model_config.get('cost_per_1k_input', 1) == 0:
                indicators.append("[Free]")

            indicator_text = " " + " ".join(indicators) if indicators else ""
            full_display_name = f"{display_name}{indicator_text}"
            model_options_in_category[full_display_name] = model

        # Initialize model choice if not set or not in current category
        if (
            page_model_key not in st.session_state
            or st.session_state[page_model_key] not in category_models
        ):
            if page_default and page_default in category_models:
                st.session_state[page_model_key] = page_default
            else:
                st.session_state[page_model_key] = category_models[0]

        # Model selection
        current_model = st.session_state[page_model_key]
        current_model_display = None
        for display, model in model_options_in_category.items():
            if model == current_model:
                current_model_display = display
                break

        model_index = (
            list(model_options_in_category.keys()).index(current_model_display)
            if current_model_display
            else 0
        )

        selected_model_display = st.sidebar.selectbox(
            "Choose Model:",
            options=list(model_options_in_category.keys()),
            index=model_index,
            key=f"model_selector_{current_page}",
            help="Select the LLM model for this session"
        )

        selected_model = model_options_in_category[selected_model_display]

        # Update session state if model has changed
        if current_model != selected_model:
            log_model_selection(
                page=current_page,
                model=selected_model,
                previous_model=current_model
            )
            st.session_state[page_model_key] = selected_model

        st.session_state.model_choice = selected_model

    else:
        # Fallback to single dropdown for compatibility
        # Initialize model choice
        if page_model_key not in st.session_state:
            if page_default and page_default in available_models:
                st.session_state[page_model_key] = page_default
            else:
                st.session_state[page_model_key] = available_models[0] if available_models else None

        # Ensure model is available
        if st.session_state[page_model_key] not in available_models:
            if page_default and page_default in available_models:
                st.session_state[page_model_key] = page_default
            else:
                st.session_state[page_model_key] = available_models[0] if available_models else None

        st.session_state.model_choice = st.session_state[page_model_key]

        current_model = st.session_state.model_choice
        current_display = get_model_display_name(current_model)
        current_index = (
            list(model_options.keys()).index(current_display)
            if current_display in model_options.keys()
            else 0
        )

        selected_display = st.sidebar.selectbox(
            "Choose Model:",
            options=list(model_options.keys()),
            index=current_index,
            key=f"model_selector_fallback_{current_page}",
            help="Select the LLM model for this session"
        )

        selected_model = model_options[selected_display]
        if current_model != selected_model:
            log_model_selection(
                page=current_page,
                model=selected_model,
                previous_model=current_model
            )
            st.session_state.model_choice = selected_model
            st.session_state[page_model_key] = selected_model

    # Show enhanced model info (common for both approaches)
    if hasattr(st.session_state, 'model_choice') and st.session_state.model_choice:
        selected_model = st.session_state.model_choice
        model_config = MODELS[selected_model]
        input_cost_per_million = model_config['cost_per_1k_input'] * 1000
        output_cost_per_million = model_config['cost_per_1k_output'] * 1000

        # Build capability indicators
        capabilities = []
        if model_config.get('supports_vision', False):
            capabilities.append("Vision")
        if model_config.get('supports_function_calling', False):
            capabilities.append("Functions")
        if model_config.get('open_weight', False):
            capabilities.append("Open Source")

        capabilities_text = " | ".join(capabilities) if capabilities else "Text Only"

        # Determine cost display
        if input_cost_per_million == 0 and output_cost_per_million == 0:
            cost_html = "<strong>Free Tier</strong>"
        else:
            cost_html = (
                "<strong>Cost per 1M tokens:</strong><br>"
                f"&bull; Input: ${input_cost_per_million:.2f}<br>"
                f"&bull; Output: ${output_cost_per_million:.2f}"
            )

        st.sidebar.markdown(f"""
<div style="background-color: {THEME_COLORS['primary']}10; padding: 12px; border-radius: 8px; border-left: 4px solid {THEME_COLORS['primary']};">
    <strong>Provider:</strong> {model_config['provider'].title()}<br>
    <strong>Capabilities:</strong> {capabilities_text}<br>
    {cost_html}<br>
    <strong>Max Output:</strong> {model_config['max_tokens']:,}<br>
    <strong>Context Window:</strong> {model_config['context_window']:,}
</div>
""", unsafe_allow_html=True)


def render_clear_conversation_button():
    """Render Clear Conversation button for chat-based pages."""
    if not hasattr(st.session_state, 'cur_page'):
        return

    current_page = st.session_state.cur_page

    # Pages that have chat functionality with st.session_state.messages
    chat_pages = ["coding_companion", "project_coach", "exam_ally"]

    # Only show for chat-based pages (not ai_sandbox or ai_comparisons which have their own)
    if current_page not in chat_pages:
        return

    # Only show if there are messages to clear
    if "messages" not in st.session_state or len(st.session_state.messages) <= 1:
        return

    st.sidebar.markdown("")
    if st.sidebar.button("Clear Conversation", use_container_width=True, key="clear_conv_sidebar"):
        # Keep only the system message if present
        if st.session_state.messages and st.session_state.messages[0].get("role") == "system":
            st.session_state.messages = [st.session_state.messages[0]]
        else:
            st.session_state.messages = []
        st.rerun()


def render_simple_cost_info():
    """Render simplified cost information."""
    # Try to get page-specific costs first, fall back to global costs
    current_page = getattr(st.session_state, 'cur_page', 'unknown')
    page_costs_key = f'total_costs_{current_page}'

    costs_to_show = None
    if page_costs_key in st.session_state and st.session_state[page_costs_key]:
        costs_to_show = st.session_state[page_costs_key]
        cost_label = f"{current_page.replace('_', ' ').title()} Cost"
    elif "total_costs" in st.session_state and st.session_state.total_costs:
        costs_to_show = st.session_state.total_costs
        cost_label = "Total Session Cost"

    if not costs_to_show:
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<h3 style="color: {THEME_COLORS["primary"]};">Session Cost</h3>',
        unsafe_allow_html=True
    )

    total_session_cost = 0.0
    for model, cost in costs_to_show.items():
        if cost > 0:
            total_session_cost += cost

    if total_session_cost > 0:
        st.sidebar.markdown(f"""
<div style="background-color: {THEME_COLORS['warning']}15; padding: 8px; border-radius: 5px; border-left: 3px solid {THEME_COLORS['warning']};">
    <strong>{cost_label}:</strong> ${total_session_cost:.4f}
</div>
""", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"""
<div style="background-color: {THEME_COLORS['success']}15; padding: 8px; border-radius: 5px; border-left: 3px solid {THEME_COLORS['success']};">
    No cost incurred yet
</div>
""", unsafe_allow_html=True)


def render_essential_info():
    """Render essential system information only."""
    st.sidebar.markdown("---")

    # A Button to show/hide the disclaimers and references
    if "show_info" not in st.session_state:
        st.session_state.show_info = False

    button_style = f"""
    <style>
    div.stButton > button:first-child {{
        background-color: {THEME_COLORS['secondary']};
        color: white;
        border: none;
        border-radius: 4px;
        width: 100%;
        font-weight: bold;
    }}
    div.stButton > button:first-child:hover {{
        background-color: {THEME_COLORS['primary']};
        color: white;
    }}
    </style>
    """
    st.sidebar.markdown(button_style, unsafe_allow_html=True)

    if st.sidebar.button("Info & References"):
        st.session_state.show_info = not st.session_state.show_info

    # Conditionally display the disclaimers and references
    if st.session_state.show_info:
        st.sidebar.markdown(f"""
<h4 style="color: {THEME_COLORS['primary']};">Disclaimers</h4>
<div style="background-color: {THEME_COLORS['warning']}10; padding: 10px; border-radius: 5px; font-size: 0.9em;">
    &bull; ChatISA is designed for educational purposes only<br>
    &bull; Get instructor approval before using for classwork<br>
    &bull; Use at your own risk and evaluate accuracy
</div>
""", unsafe_allow_html=True)

        st.sidebar.markdown(f"""
<h4 style="color: {THEME_COLORS['primary']}; margin-top: 15px;">References</h4>
<div style="background-color: {THEME_COLORS['info']}10; padding: 10px; border-radius: 5px; font-size: 0.85em;">
    <strong>Prompt Engineering:</strong><br>
    <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4475995" style="color: {THEME_COLORS['primary']};">Assigning AI by Mollick & Mollick 2023</a><br><br>
    <strong>ChatISA Code Repository:</strong><br>
    <a href="https://github.com/fmegahed/chatisa" style="color: {THEME_COLORS['primary']};">GitHub Repository</a><br><br>
    <strong>ChatISA Research Paper:</strong><br>
    <a href="https://arxiv.org/pdf/2407.15010" style="color: {THEME_COLORS['primary']};">ArXiv Publication</a>
</div>
""", unsafe_allow_html=True)
