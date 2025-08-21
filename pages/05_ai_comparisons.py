# About this code:
# ----------------
# AI Comparisons: Compare AI model responses side-by-side with proper vision and PDF support.
# Uses correct LangChain init_chat_model approach for handling images and PDFs.
# No logging or PDF export functionality - purely for educational comparison.
# -----------------------------------------------------------------------------

# Import required libraries:
# --------------------------
import os
import streamlit as st
import tempfile
import time
import base64
import io
from typing import List, Dict, Any
from dotenv import load_dotenv
from PIL import Image

# LangChain imports for proper model handling
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Our Own Modules
from lib import sidebar
from config import MODELS, get_model_display_name

# -----------------------------------------------------------------------------

# Load Environment Variables:
# ---------------------------
load_dotenv()

# Constants:
# ----------
TEMPERATURE = 0.3
MAX_TOKENS = 2000

# Available models for comparison:
ALL_AVAILABLE_MODELS = [m for m in MODELS.keys() if not MODELS[m].get('realtime_only', False)]

# Default selected models - main models from each provider:
DEFAULT_MODELS = ['gpt-5-chat-latest', 'claude-sonnet-4-20250514']

# -----------------------------------------------------------------------------
# Manage page tracking and session state
# -----------------------------------------------------------------------------
THIS_PAGE = "ai_comparisons"
if "cur_page" not in st.session_state:
    st.session_state.cur_page = THIS_PAGE

# Clear any existing messages when switching to this page
if (st.session_state.cur_page != THIS_PAGE) and ("comparison_messages" in st.session_state):
    del st.session_state.comparison_messages

# Initialize comparison-specific session state
if "comparison_messages" not in st.session_state:
    st.session_state.comparison_messages = []

if "comparison_responses" not in st.session_state:
    st.session_state.comparison_responses = {}

if "selected_models" not in st.session_state:
    st.session_state.selected_models = DEFAULT_MODELS.copy()

st.session_state.cur_page = THIS_PAGE
# -----------------------------------------------------------------------------

def get_model_display_info(model: str) -> Dict[str, str]:
    """Get display information for a model."""
    if model in MODELS:
        config = MODELS[model]
        # Color mapping for different providers
        provider_colors = {
            'openai': '#00A67E',
            'anthropic': '#D4A574', 
            'cohere': '#39CCCC',
            'meta (via groq)': '#4285F4',
            'groq': '#4285F4',
            'huggingface_inference': '#FF6B35'
        }
        
        return {
            'name': config['display_name'],
            'provider': config['provider'].title(),
            'color': provider_colors.get(config['provider'].lower(), '#888888'),
            'supports_vision': config.get('supports_vision', False)
        }
    
    return {'name': model, 'provider': 'Unknown', 'color': '#888888', 'supports_vision': False}

def get_langchain_model_name(model: str) -> str:
    """Convert our model names to LangChain init_chat_model format."""
    model_mapping = {
        'gpt-5-chat-latest': 'openai:gpt-5-chat-latest',
        'gpt-5-mini-2025-08-07': 'openai:gpt-5-mini-2025-08-07',
        'claude-sonnet-4-20250514': 'anthropic:claude-sonnet-4-20250514',
        'command-a-03-2025': 'cohere:command-a-03-2025',
        'llama-3.3-70b-versatile': 'groq:llama-3.3-70b-versatile',
        'llama-3.1-8b-instant': 'groq:llama-3.1-8b-instant'
    }
    return model_mapping.get(model, model)

def encode_image_to_base64(uploaded_file) -> tuple:
    """Convert uploaded image file to base64 string with proper format validation."""
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        # Use PIL to properly handle and validate the image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary (for JPEG compatibility)
        if image.mode in ("RGBA", "P", "LA"):
            # Convert to RGB for JPEG, keep RGBA for PNG
            if uploaded_file.name.lower().endswith(('.jpg', '.jpeg')):
                # Convert RGBA to RGB with white background for JPEG
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = rgb_image
        
        # Determine the best format based on the original file
        original_format = image.format
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        # Choose appropriate format and MIME type
        if file_extension in ['png'] or original_format == 'PNG' or image.mode == 'RGBA':
            output_format = 'PNG'
            mime_type = 'image/png'
        elif file_extension in ['jpg', 'jpeg'] or original_format == 'JPEG':
            output_format = 'JPEG'
            mime_type = 'image/jpeg'
        elif file_extension in ['gif'] or original_format == 'GIF':
            output_format = 'PNG'  # Convert GIF to PNG for better compatibility
            mime_type = 'image/png'
        elif file_extension in ['webp'] or original_format == 'WEBP':
            output_format = 'PNG'  # Convert WebP to PNG for better compatibility
            mime_type = 'image/png'
        else:
            # Default to JPEG for unknown formats
            output_format = 'JPEG'
            mime_type = 'image/jpeg'
            if image.mode == 'RGBA':
                # Convert RGBA to RGB for JPEG
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])
                image = rgb_image
        
        # Convert to bytes in the chosen format
        img_buffer = io.BytesIO()
        image.save(img_buffer, format=output_format, quality=95 if output_format == 'JPEG' else None)
        img_buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        
        # Reset original file pointer for potential reuse
        uploaded_file.seek(0)
        
        return img_base64, mime_type
        
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        # Reset file pointer on error
        uploaded_file.seek(0)
        return None, None

def encode_pdf_to_base64(uploaded_file) -> str:
    """Convert uploaded PDF file to base64 string."""
    try:
        # Read the uploaded file
        pdf_bytes = uploaded_file.read()
        # Reset file pointer for potential reuse
        uploaded_file.seek(0)
        
        # Encode to base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        return pdf_base64
    except Exception as e:
        st.error(f"Error encoding PDF: {str(e)}")
        return None

def get_file_type(filename: str) -> str:
    """Determine file type from extension."""
    extension = filename.lower().split('.')[-1]
    
    # Image extensions
    if extension in ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff']:
        return 'image'
    # PDF extensions
    elif extension in ['pdf']:
        return 'pdf'
    # Document extensions
    elif extension in ['doc', 'docx', 'txt', 'md', 'rtf']:
        return 'document'
    # Code extensions
    elif extension in ['r', 'py', 'js', 'html', 'css', 'java', 'cpp', 'c', 'go', 'rs', 'php', 'rb', 'swift']:
        return 'code'
    # Data extensions
    elif extension in ['csv', 'json', 'xml', 'yaml', 'yml']:
        return 'data'
    else:
        return 'other'

def create_message_for_model(prompt: str, uploaded_files: List, model: str) -> Dict[str, Any]:
    """Create properly formatted message for specific model with files."""
    model_config = MODELS.get(model, {})
    provider = model_config.get('provider', '')
    supports_vision = model_config.get('supports_vision', False)
    
    # Start with text content
    content = [{"type": "text", "text": prompt}]
    
    # Process each uploaded file based on its type
    for uploaded_file in uploaded_files:
        if uploaded_file is None:
            continue
            
        file_type = get_file_type(uploaded_file.name)
        
        if file_type == 'image' and supports_vision:
            img_result = encode_image_to_base64(uploaded_file)
            if img_result[0]:  # If encoding succeeded
                img_base64, mime_type = img_result
                
                # Show processing info with more details
                st.caption(f"📷 Converted {uploaded_file.name} to {mime_type} (size: {len(img_base64)//1024}KB base64) for {provider}")
                
                if provider == 'anthropic':
                    # Claude format - exactly as per LangChain docs
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": img_base64
                        }
                    })
                else:
                    # OpenAI format
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img_base64}"
                        }
                    })
        
        elif file_type == 'pdf':
            # Both OpenAI and Claude can read PDFs directly - use correct format for each
            pdf_base64 = encode_pdf_to_base64(uploaded_file)
            if pdf_base64:
                if provider == 'anthropic':
                    # Claude's document format
                    content.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_base64
                        }
                    })
                elif provider == 'openai':
                    # OpenAI's file format (requires filename)
                    content.append({
                        "type": "file",
                        "source_type": "base64",
                        "data": pdf_base64,
                        "mime_type": "application/pdf",
                        "filename": uploaded_file.name
                    })
        
        elif file_type in ['document', 'code', 'data']:
            # For text-based files, read content and add as text
            try:
                uploaded_file.seek(0)
                content_text = uploaded_file.read().decode('utf-8')
                prompt += f"\n\n--- Content from {uploaded_file.name} ---\n{content_text}\n--- End of {uploaded_file.name} ---"
                # Update the text content
                content[0]['text'] = prompt
            except Exception as e:
                prompt += f"\n\n[Error reading {uploaded_file.name}: {str(e)}]"
                content[0]['text'] = prompt
        
        else:
            # For unsupported file types or models without required capabilities
            if file_type == 'image' and not supports_vision:
                prompt += f"\n\n[Image file uploaded: {uploaded_file.name} - This model cannot process images]"
            elif file_type == 'pdf' and provider not in ['anthropic', 'openai']:
                prompt += f"\n\n[PDF file uploaded: {uploaded_file.name} - This model cannot read PDFs directly]"
            else:
                prompt += f"\n\n[File uploaded: {uploaded_file.name} - File type not supported by this model]"
            
            content[0]['text'] = prompt
    
    return {
        "role": "user",
        "content": content
    }

def get_model_response_proper(model: str, uploaded_files: List, prompt: str) -> Dict[str, Any]:
    """Get response from a single model using proper LangChain approach."""
    try:
        # Create model-specific message
        message = create_message_for_model(prompt, uploaded_files, model)
        
        # Get model config
        model_config = MODELS.get(model, {})
        provider = model_config.get('provider', '')
        
        # Initialize the model directly with the appropriate class
        if provider == 'anthropic':
            llm = ChatAnthropic(
                model=model,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
        elif provider == 'openai':
            llm = ChatOpenAI(
                model=model,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
        elif provider == 'huggingface_inference':
            # Use chatgeneration module for HF models since it has proper setup
            from lib.chatgeneration import generate_chat_completion
            # Convert message format for chatgeneration
            messages_for_hf = [message]
            response_text, input_tokens, output_tokens = generate_chat_completion(
                model=model,
                messages=messages_for_hf,
                temp=TEMPERATURE,
                max_num_tokens=MAX_TOKENS
            )
            
            # Return in expected format
            return {
                "model": model,
                "response": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "success": True,
                "error": None,
                "response_time_ms": 0  # Time tracking handled in chatgeneration
            }
        else:
            # Fallback to init_chat_model for other providers
            langchain_model = get_langchain_model_name(model)
            llm = init_chat_model(
                langchain_model,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
        
        # Generate response
        start_time = time.time()
        response = llm.invoke([message])
        response_time_ms = (time.time() - start_time) * 1000
        
        # Extract response text
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Try to extract token usage if available
        input_tokens = None
        output_tokens = None
        if hasattr(response, 'response_metadata'):
            metadata = response.response_metadata
            # Different providers store token info differently
            if 'token_usage' in metadata:
                input_tokens = metadata['token_usage'].get('prompt_tokens')
                output_tokens = metadata['token_usage'].get('completion_tokens')
            elif 'usage' in metadata:
                input_tokens = metadata['usage'].get('input_tokens')
                output_tokens = metadata['usage'].get('output_tokens')
        
        return {
            "model": model,
            "response": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "success": True,
            "error": None,
            "response_time_ms": response_time_ms
        }
    except Exception as e:
        # Add more detailed error information for debugging
        error_msg = str(e)
        if "base64" in error_msg.lower() or "media_type" in error_msg.lower():
            error_msg += " (Possible image encoding/format issue)"
        
        return {
            "model": model,
            "response": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "success": False,
            "error": error_msg,
            "response_time_ms": 0
        }

# -----------------------------------------------------------------------------
# The Streamlit Interface:
# ------------------------

# Streamlit Page Configuration:
# -----------------------------
st.set_page_config(page_title="ChatISA: AI Comparisons", layout="wide", page_icon='assets/favicon.png')

# Import theme colors
from config import THEME_COLORS

st.markdown(f'<h2 style="color: {THEME_COLORS["primary"]};">ChatISA: AI Comparisons</h2>', unsafe_allow_html=True)
st.markdown("*Compare AI model responses side-by-side with vision and document support*")

# Info box
st.info(
    "🧪 **Experimental Feature**: Ask questions and compare how different AI models respond. "
    "Upload images, PDFs, or simple datasets for a comperative analysis of the capabilities of different models. Vision models and data will be directly processed via the provider's API, i.e., no code interpreter sessions or VMs will be called by ChatISA."
)

# -----------------------------------------------------------------------------
# Custom sidebar for model selection
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚖️ AI Comparison Settings")
    
    # Model selection with grouped checkboxes
    st.markdown("#### Select Models to Compare:")
    st.markdown("*Choose models from different categories using checkboxes*")
    
    from config import MODEL_CATEGORIES
    
    # Initialize selected models if not in session state
    if 'checkbox_selected_models' not in st.session_state:
        st.session_state.checkbox_selected_models = DEFAULT_MODELS.copy()
    
    selected_models = []
    
    # Create vertical layout with headers and checkboxes
    st.markdown("---")
    
    for category_key, category_info in MODEL_CATEGORIES.items():
        # Check if this category has any available models
        category_models = [m for m in category_info['models'] if m in MODELS and m in ALL_AVAILABLE_MODELS]
        if not category_models:
            continue
            
        # Category header
        st.markdown(f"### {category_info['display_name']}")
        st.markdown(f"*{category_info['description']}*")
        
        # Create columns for better layout (2 columns for models)
        cols = st.columns(2)
        
        for i, model in enumerate(category_models):
            model_config = MODELS[model]
            model_display = model_config['display_name']
            
            # Add indicators for special capabilities
            indicators = []
            if model_config.get('supports_vision', False):
                indicators.append("👁️ Vision")
            if model_config.get('open_weight', False):
                indicators.append("🔓 Open")
            if model_config.get('cost_per_1k_input', 1) == 0:
                indicators.append("💰 Free")
            
            indicator_text = f" ({', '.join(indicators)})" if indicators else ""
            
            # Check if this model should be selected by default
            default_selected = model in st.session_state.checkbox_selected_models
            
            # Alternate between columns
            with cols[i % 2]:
                if st.checkbox(
                    f"{model_display}{indicator_text}",
                    value=default_selected,
                    key=f"model_checkbox_{model}",
                    help=f"{model_config['description']} | Provider: {model_config['provider'].title()}"
                ):
                    selected_models.append(model)
        
        # Add some spacing between categories
        st.markdown("---")
    
    # Update session state with newly selected models
    if selected_models != st.session_state.checkbox_selected_models:
        st.session_state.checkbox_selected_models = selected_models
        st.session_state.selected_models = selected_models
    elif not selected_models:
        # Keep the previous selection if no checkboxes are currently selected
        selected_models = st.session_state.selected_models
    
    # Show selected models info with vision capability
    if st.session_state.selected_models:
        st.markdown("#### Selected Models:")
        vision_models = []
        for model in st.session_state.selected_models:
            info = get_model_display_info(model)
            vision_indicator = " 👁️" if info['supports_vision'] else ""
            st.write(f"• {info['name']} ({info['provider']}){vision_indicator}")
            if info['supports_vision']:
                vision_models.append(info['name'])
        
        if vision_models:
            st.success(f"📸 Vision-capable: {', '.join(vision_models)}")
    
    st.markdown("---")
    st.markdown("#### Generation Settings:")
    temperature = st.slider("Temperature", 0.0, 1.0, TEMPERATURE, 0.1)
    max_tokens = st.slider("Max Tokens", 500, 4000, MAX_TOKENS, 100)
    
    # Update global settings
    globals()['TEMPERATURE'] = temperature
    globals()['MAX_TOKENS'] = max_tokens
# -----------------------------------------------------------------------------

# File upload section:
# -------------------
st.markdown("### 📁 Upload Files (Optional)")

uploaded_files = st.file_uploader(
    "Upload files (images, PDFs, documents, code):",
    type=['png', 'jpg', 'jpeg', 'pdf', 'doc', 'docx', 'txt', 'md', 'r', 'py', 'js', 'css', 'json', 'csv'],
    accept_multiple_files=True,
    help="Upload any supported file type - processing will be automatically determined by file extension"
)

# Display uploaded content:
# -------------------------
if uploaded_files:
    st.markdown("#### 📋 Uploaded Files:")
    
    vision_models = [get_model_display_info(m)['name'] for m in st.session_state.selected_models 
                    if get_model_display_info(m)['supports_vision']]
    
    pdf_capable_models = [get_model_display_info(m)['name'] for m in st.session_state.selected_models 
                         if MODELS[m]['provider'] in ['anthropic', 'openai']]
    
    for uploaded_file in uploaded_files:
        file_type = get_file_type(uploaded_file.name)
        
        if file_type == 'image':
            st.image(uploaded_file, caption=f"📸 {uploaded_file.name}", width=200)
            if vision_models:
                st.success(f"👁️ **Vision models** ({', '.join(vision_models)}) will analyze this image")
            else:
                st.warning("⚠️ **No vision models selected** - Add GPT-5 or Claude to analyze images")
        
        elif file_type == 'pdf':
            st.info(f"📄 **PDF**: {uploaded_file.name}")
            if pdf_capable_models:
                st.success(f"📖 **PDF-capable models** ({', '.join(pdf_capable_models)}) can read this PDF directly")
            else:
                st.warning("⚠️ **No PDF-capable models selected** - Add GPT-5 or Claude to read PDFs directly")
        
        elif file_type in ['document', 'code', 'data']:
            icon = {'document': '📝', 'code': '💻', 'data': '📊'}[file_type]
            st.info(f"{icon} **{file_type.title()}**: {uploaded_file.name} - Content will be included as text")
        
        else:
            st.warning(f"❓ **Unknown type**: {uploaded_file.name} - May not be processed by all models")

# Main question input:
# -------------------
st.markdown("### 💬 Ask Your Question")

# Use chat_input for consistency with other modules (Enter to submit)
if prompt := st.chat_input("Ask a question to compare AI responses (e.g., 'Explain quantum computing' or 'What do you see in this image?')"):
    # Check if we can generate responses
    can_generate = prompt.strip() and len(st.session_state.selected_models) >= 1
    
    if can_generate:
        # Clear previous responses
        st.session_state.comparison_responses = {}
        
        # Files will be processed per model in get_model_response_proper
        files_to_process = uploaded_files if uploaded_files else []
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate responses sequentially
        total_models = len(st.session_state.selected_models)
        for i, model in enumerate(st.session_state.selected_models):
            model_info = get_model_display_info(model)
            status_text.text(f"Getting response from {model_info['name']}...")
            
            result = get_model_response_proper(model, files_to_process, prompt)
            st.session_state.comparison_responses[model] = result
            
            # Update progress
            progress = (i + 1) / total_models
            progress_bar.progress(progress)
            
            # Small delay to show progress
            time.sleep(0.2)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Determine content info
        content_info = "Text"
        if uploaded_files:
            file_types = set(get_file_type(f.name) for f in uploaded_files)
            if 'image' in file_types:
                content_info += " + Image"
            if 'pdf' in file_types:
                content_info += " + PDF"
            if any(ft in file_types for ft in ['document', 'code', 'data']):
                content_info += " + Files"
        
        # Add to message history
        st.session_state.comparison_messages.append({
            "prompt": prompt,
            "content_info": content_info,
            "responses": st.session_state.comparison_responses.copy()
        })
    else:
        # Show warning if cannot generate
        if not st.session_state.selected_models:
            st.warning("⚠️ Please select at least one model from the sidebar to compare responses.")
        elif not prompt.strip():
            st.warning("⚠️ Please enter a question to compare responses.")

# Display results section:
# -----------------------
if st.session_state.comparison_responses and st.session_state.selected_models:
    st.markdown("---")
    st.markdown("### 📊 Comparison Results")
    
    # Dynamic column layout
    num_models = len(st.session_state.selected_models)
    
    if num_models <= 3:
        cols = st.columns(num_models)
        
        for i, model in enumerate(st.session_state.selected_models):
            model_info = get_model_display_info(model)
            result = st.session_state.comparison_responses.get(model, {})
            
            with cols[i]:
                # Model header with custom styling
                vision_badge = " 👁️" if model_info['supports_vision'] else ""
                st.markdown(f"""
                    <div style="
                        background-color: {model_info['color']}20;
                        border-left: 4px solid {model_info['color']};
                        padding: 10px;
                        margin-bottom: 10px;
                        border-radius: 5px;
                    ">
                        <h4 style="margin: 0; color: {model_info['color']};">
                            {model_info['name']}{vision_badge}
                        </h4>
                        <small style="color: #666;">{model_info['provider']}</small>
                    </div>
                """, unsafe_allow_html=True)
                
                if result.get("success"):
                    st.markdown(result["response"])
                    
                    # Show usage info
                    if result.get("input_tokens") or result.get("output_tokens"):
                        with st.expander("📊 Usage Info", expanded=False):
                            if result.get("input_tokens"):
                                st.write(f"Input: {result['input_tokens']:,}")
                            if result.get("output_tokens"):
                                st.write(f"Output: {result['output_tokens']:,}")
                            if result.get("response_time_ms"):
                                st.write(f"Time: {result['response_time_ms']:.0f}ms")
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
    else:
        # For 4+ models, use expandable sections
        st.markdown("**Showing results in expandable sections:**")
        
        for model in st.session_state.selected_models:
            model_info = get_model_display_info(model)
            result = st.session_state.comparison_responses.get(model, {})
            
            vision_badge = " 👁️" if model_info['supports_vision'] else ""
            with st.expander(f"{model_info['name']}{vision_badge} ({model_info['provider']})", expanded=True):
                if result.get("success"):
                    st.markdown(result["response"])
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

# Chat history section:
# --------------------
if st.session_state.comparison_messages:
    st.markdown("---")
    st.markdown("### 📝 Recent Comparisons")
    
    # Show last 3 comparisons
    recent_messages = st.session_state.comparison_messages[-3:]
    
    for i, message in enumerate(reversed(recent_messages)):
        question_preview = message['prompt'][:80] + ("..." if len(message['prompt']) > 80 else "")
        content_badge = f" [{message['content_info']}]" if message.get('content_info') != "Text" else ""
        
        with st.expander(f"Q{len(recent_messages) - i}: {question_preview}{content_badge}", expanded=False):
            st.markdown(f"**Question:** {message['prompt']}")
            
            if message.get('content_info') != "Text":
                st.caption(f"📁 Content: {message['content_info']}")
            
            # Show abbreviated responses
            models_used = list(message['responses'].keys())
            for model in models_used[:3]:  # Show max 3 in history
                model_info = get_model_display_info(model)
                result = message['responses'].get(model, {})
                
                if result.get("success"):
                    response_preview = result["response"][:200] + ("..." if len(result["response"]) > 200 else "")
                    st.markdown(f"**{model_info['name']}:** {response_preview}")
                else:
                    st.markdown(f"**{model_info['name']}:** ❌ Error")
            
            if len(models_used) > 3:
                st.caption(f"... and {len(models_used) - 3} more models")

# Clear history button:
if st.session_state.comparison_messages:
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.comparison_messages = []
        st.session_state.comparison_responses = {}
        st.rerun()


# -----------------------------------------------------------------------------