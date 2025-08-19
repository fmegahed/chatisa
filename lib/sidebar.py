import streamlit as st
from config import (
    APP_NAME, VERSION, DATE, MODELS, MODEL_GROUPS, DEFAULT_MODELS,
    get_model_display_name, calculate_cost, validate_api_keys,
    SIDEBAR_CONFIG
)
from lib.enhanced_usage_logger import log_model_selection, log_session_action

def render_sidebar():
    """Render clean sidebar with essential information."""
    
    # App information
    key_validation = validate_api_keys()
    available_models = len(key_validation["available_models"])
    total_models = len(MODELS)
    
    st.sidebar.markdown(f"""
### {APP_NAME} v{VERSION}
**Educational AI Assistant**  
*{DATE}*
""")
    
    # Model Selection (if enabled) - placed before Status section
    if SIDEBAR_CONFIG.get("show_model_selector", True) and hasattr(st.session_state, 'cur_page'):
        render_model_selector()
    
    st.sidebar.markdown(f"""
### Status
✅ **{available_models}/{total_models}** models available

### Maintained By 
  - [Fadel Megahed](https://miamioh.edu/fsb/directory/?up=/directory/megahefm)   
  - [Joshua Ferris](https://miamioh.edu/fsb/directory/?up=/directory/ferrisj2)

### Key Features
  - Free to use for students  
  - 4 specialized AI modules
  - Multiple LLM providers
  - Speech-to-speech interview prep

### Support & Funding
  - Farmer School of Business
  - US Bank  
  - Raymond E. Glos Professorship
""")
    
    # Simple cost tracking (if enabled)
    if SIDEBAR_CONFIG.get("show_cost_calculator", True):
        render_simple_cost_info()
    
    # Essential system information only
    render_essential_info()

def render_model_selector():
    """Render model selection interface."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Selection")
    
    # Get available models for current page
    if hasattr(st.session_state, 'cur_page'):
        page = st.session_state.cur_page
        # Special handling for interview mentor - only show realtime model
        if page == "interview_mentor":
            available_models = ["gpt-4o-realtime-preview-2025-06-03"]
            if "gpt-4o-realtime-preview-2025-06-03" not in MODELS:
                available_models = []
        else:
            # Filter available models based on API keys for other pages
            key_validation = validate_api_keys()
            available_models = [m for m in key_validation["available_models"] if m in MODELS and not MODELS[m].get("realtime_only", False)]
    else:
        available_models = list(MODELS.keys())
    
    if not available_models:
        st.sidebar.error("No models available. Please check your API keys.")
        return
    
    # Special message for interview mentor page
    if hasattr(st.session_state, 'cur_page') and st.session_state.cur_page == "interview_mentor":
        st.sidebar.info("🎤 **Speech Interview Mode**\nOnly real-time speech models are available for interview conversations.")
    
    # Model selection with display names
    model_options = {get_model_display_name(m): m for m in available_models}
    
    # Initialize model_choice with page-specific defaults, preserving user choices per page
    current_page = getattr(st.session_state, 'cur_page', 'unknown')
    page_default = DEFAULT_MODELS.get(current_page)
    
    # Use page-specific session state keys to preserve model choice per page
    page_model_key = f"model_choice_{current_page}"
    
    # Initialize page-specific model choice if it doesn't exist
    if page_model_key not in st.session_state:
        # Use page-specific default if available and in the available models
        if page_default and page_default in available_models:
            st.session_state[page_model_key] = page_default
        else:
            # Fallback to first available model
            st.session_state[page_model_key] = available_models[0] if available_models else None
    
    # Ensure the page-specific model is available, fallback if not
    if st.session_state[page_model_key] not in available_models:
        if page_default and page_default in available_models:
            st.session_state[page_model_key] = page_default
        else:
            st.session_state[page_model_key] = available_models[0] if available_models else None
    
    # Set the current model_choice to the page-specific choice
    st.session_state.model_choice = st.session_state[page_model_key]
    
    # Get current index for the selectbox
    current_model = st.session_state.model_choice
    current_display = get_model_display_name(current_model) if current_model in model_options.values() else list(model_options.keys())[0]
    current_index = list(model_options.keys()).index(current_display) if current_display in model_options.keys() else 0
    
    selected_display = st.sidebar.selectbox(
        "Choose Model:",
        options=list(model_options.keys()),
        index=current_index,
        key="model_selector_display",
        help="Select the LLM model for this session"
    )
    
    if selected_display:
        selected_model = model_options[selected_display]
        # Update session state if model has changed and log model change
        if current_model != selected_model:
            log_model_selection(
                page=getattr(st.session_state, 'cur_page', 'unknown'),
                model=selected_model,
                previous_model=current_model
            )
            st.session_state.model_choice = selected_model
            # Also update the page-specific model choice
            st.session_state[page_model_key] = selected_model
        
        # Show model info with costs per million tokens
        model_config = MODELS[selected_model]
        input_cost_per_million = model_config['cost_per_1k_input'] * 1000
        output_cost_per_million = model_config['cost_per_1k_output'] * 1000
        
        st.sidebar.info(
            f"**Provider:** {model_config['provider'].title()}  \n"
            f"**Cost per 1M tokens:**  \n"
            f"• Input: \${input_cost_per_million:.2f}  \n"
            f"• Output: \${output_cost_per_million:.2f}  \n"
            f"**Max Output Tokens:** {model_config['max_tokens']:,}  \n"
            f"**Context Window:** {model_config['context_window']:,}"
        )

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
    st.sidebar.markdown("### 💰 Session Cost")
    
    total_session_cost = 0.0
    for model, cost in costs_to_show.items():
        if cost > 0:
            total_session_cost += cost
    
    if total_session_cost > 0:
        st.sidebar.metric(cost_label, f"${total_session_cost:.4f}")
    else:
        st.sidebar.info("No cost incurred yet")

def render_essential_info():
    """Render essential system information only."""
    st.sidebar.markdown("---")
    
    # A Button to show/hide the disclaimers and references
    if "show_info" not in st.session_state:
        st.session_state.show_info = False
    if st.sidebar.button("📋 Info & References"):
        st.session_state.show_info = not st.session_state.show_info
    
    # Conditionally display the disclaimers and references
    if st.session_state.show_info:
        st.sidebar.markdown("""
### Disclaimers
- ChatISA is designed for educational purposes only
- Get instructor approval before using for classwork
- Use at your own risk and evaluate accuracy

### References
- **Prompt Engineering:** [Assigning AI by Mollick & Mollic 2023](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4475995)
- **ChatISA Code Repo:** [GitHub](https://github.com/fmegahed/chatisa)
- **ChatISA Paper:** [ArXiv](https://arxiv.org/pdf/2407.15010)
""")

