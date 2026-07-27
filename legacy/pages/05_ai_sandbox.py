# About this code:
# ----------------
# AI Sandbox: Interactive code execution sandbox using OpenAI's Responses API.
# Focuses on data analysis, math computation, and visualizations with Python code execution.
# Supports file uploads (CSV, images, etc.) and allows downloading generated graphs.
# Displays code execution like a notebook with inline outputs and images.
# -----------------------------------------------------------------------------

# Import required libraries:
# --------------------------
import os
import streamlit as st
import tempfile
import time
import base64
import io
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI

# Our Own Modules
from lib import sidebar
from config import MODELS, OPENAI_API_KEY
from lib.ui import apply_theme_css
from lib.enhanced_usage_logger import log_enhanced_usage

# -----------------------------------------------------------------------------

# Load Environment Variables:
# ---------------------------
load_dotenv(override=True)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Constants:
# ----------
DEFAULT_MODEL = "gpt-5.2-2025-12-11"  # Default OpenAI model
TEMPERATURE = 0.7
MAX_TOKENS = 4000

def get_current_model():
    """Get the currently selected model from session state."""
    return st.session_state.get('model_choice', DEFAULT_MODEL)

# System instructions for the AI Sandbox
SYSTEM_INSTRUCTIONS = """
You are an advanced AI assistant with Python code execution capabilities in a secure sandbox environment.

When users ask you to:
- Analyze data, create visualizations, or perform calculations -> Write and execute Python code
- Upload files (CSV, Excel, etc.) -> Read and analyze them with pandas, matplotlib, etc.
- Solve math problems -> Show step-by-step solutions with code

Always:
1. Write clear, well-commented Python code
2. Use appropriate libraries (pandas, matplotlib, numpy, scipy, etc.)
3. Explain your approach before coding
4. Provide insights after showing results
5. For visualizations, save figures to files so users can download them

IMPORTANT for matplotlib plots:
- When generating matplotlib plots, save the figure OR display it, but NOT both
- If saving, always call plt.close() after plt.savefig() to prevent duplicates
- Example: plt.savefig('plot.png'); plt.close()

IMPORTANT for uploaded files:
- Uploaded files are automatically available in the /mnt/data/ directory
- Access them directly: pd.read_csv('/mnt/data/filename.csv') or pd.read_excel('/mnt/data/filename.xlsx')
- Use os.listdir('/mnt/data/') to see all available files

Be educational and thorough. You're helping students learn data analysis and computational thinking.
"""

# -----------------------------------------------------------------------------
# Manage page tracking and session state
# -----------------------------------------------------------------------------
THIS_PAGE = "ai_sandbox"
if "cur_page" not in st.session_state:
    st.session_state.cur_page = THIS_PAGE

# Clear any existing messages when switching to this page
if (st.session_state.cur_page != THIS_PAGE) and ("sandbox_messages" in st.session_state):
    del st.session_state.sandbox_messages
    if "sandbox_file_ids" in st.session_state:
        del st.session_state.sandbox_file_ids

# Initialize sandbox-specific session state
if "sandbox_messages" not in st.session_state:
    st.session_state.sandbox_messages = []

if "sandbox_file_ids" not in st.session_state:
    st.session_state.sandbox_file_ids = []

if "sandbox_output_files" not in st.session_state:
    st.session_state.sandbox_output_files = []

if "sandbox_container_id" not in st.session_state:
    st.session_state.sandbox_container_id = None

# Initialize model_choice with default if not set
if "model_choice" not in st.session_state:
    st.session_state.model_choice = DEFAULT_MODEL

st.session_state.cur_page = THIS_PAGE
# -----------------------------------------------------------------------------

# Page Configuration:
# -------------------
st.set_page_config(
    page_title="AI Sandbox - ChatISA",
    page_icon="assets/favicon.png",
    layout="wide"
)

apply_theme_css(include_code_styles=True)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def upload_file_to_openai(file) -> Optional[str]:
    """
    Upload a file to OpenAI for use with code interpreter.
    Returns the file ID if successful, None otherwise.
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        # Upload to OpenAI
        with open(tmp_path, 'rb') as f:
            uploaded_file = client.files.create(
                file=f,
                purpose="user_data"
            )

        # Clean up temp file
        os.unlink(tmp_path)

        return uploaded_file.id
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

def render_code_execution(code: str, outputs: List[Dict]):
    """Render code execution in notebook style with outputs."""
    st.markdown('<div class="tool-badge">Python Code Execution</div>', unsafe_allow_html=True)

    # Show the code
    st.code(code, language="python")

    # Show outputs
    if outputs:
        st.markdown('<div class="output-section">', unsafe_allow_html=True)
        st.markdown("**Output:**")

        for output in outputs:
            if output["type"] == "logs":
                st.text(output["logs"])
            elif output["type"] == "image":
                # Display image inline
                try:
                    image_data = base64.b64decode(output["image"]["b64_json"])
                    image = Image.open(io.BytesIO(image_data))

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.image(image, width="stretch")
                    with col2:
                        st.download_button(
                            label="Download",
                            data=image_data,
                            file_name=f"plot_{int(time.time())}.png",
                            mime="image/png",
                            key=f"download_{output['image'].get('file_id', time.time())}"
                        )
                except Exception as e:
                    st.warning(f"Could not display image: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)

def process_sandbox_request(user_message: str, file_ids: List[str] = None) -> Dict[str, Any]:
    """
    Process a user request using OpenAI Responses API with code interpreter.
    Returns a dictionary with response content and code executions.
    """
    try:
        # Build conversation history into instructions
        conversation_context = ""
        if st.session_state.sandbox_messages:
            conversation_context = "\n\nConversation history:\n"
            for msg in st.session_state.sandbox_messages[-6:]:  # Last 6 messages for context
                role_label = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role_label}: {msg['content']}\n"

        # Combine system instructions with conversation context
        full_instructions = SYSTEM_INSTRUCTIONS + conversation_context

        # Build tools config with container reuse and file attachment
        # If we have an existing container, reuse it; otherwise create new with files attached
        if st.session_state.sandbox_container_id:
            # Reuse existing container
            tools_config = [
                {
                    "type": "code_interpreter",
                    "container": st.session_state.sandbox_container_id
                }
            ]
        else:
            # Create new container with files attached
            container_config = {
                "type": "auto",
                "memory_limit": "1g"
            }
            if file_ids:
                container_config["file_ids"] = file_ids

            tools_config = [
                {
                    "type": "code_interpreter",
                    "container": container_config
                }
            ]

        # Models that don't support temperature parameter
        no_temperature_models = ["gpt-5-mini-2025-08-07", "o1", "o1-mini", "o1-preview"]
        current_model = get_current_model()

        # Build request parameters (use simple string input, not array)
        request_params = {
            "model": current_model,
            "tools": tools_config,
            "instructions": full_instructions,
            "input": user_message,
            "max_output_tokens": MAX_TOKENS,
            "include": ["code_interpreter_call.outputs"]  # Get stdout/stderr outputs
        }

        # Only add temperature for models that support it
        if not any(m in current_model for m in no_temperature_models):
            request_params["temperature"] = TEMPERATURE

        # Call Responses API with code interpreter tool
        start_time = time.time()
        response = client.responses.create(**request_params)
        response_time_ms = (time.time() - start_time) * 1000

        # Extract text content and code executions from response.output
        # response.output is a LIST with items of type "code_interpreter_call" and "message"
        content = ""
        code_executions = []
        processed_image_ids = set()  # Track processed images to avoid duplicates by file_id
        processed_image_hashes = set()  # Track processed images to avoid duplicates by content hash

        if hasattr(response, 'output') and response.output:
            for output_item in response.output:
                # Extract code execution
                if output_item.type == "code_interpreter_call":
                    # Store container_id for reuse in future requests
                    if hasattr(output_item, 'container_id') and output_item.container_id:
                        st.session_state.sandbox_container_id = output_item.container_id

                    code_executions.append({
                        "code": output_item.code,
                        "outputs": [],  # Will be populated from annotations
                        "container_id": output_item.container_id if hasattr(output_item, 'container_id') else None
                    })

                # Extract text response and check for image annotations
                elif output_item.type == "message":
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if content_item.type == "output_text":
                                content += content_item.text + "\n"

                                # Check for image annotations (container_file_citation)
                                if hasattr(content_item, 'annotations') and content_item.annotations:
                                    for annotation in content_item.annotations:
                                        if annotation.type == "container_file_citation":
                                            file_id = annotation.file_id

                                            # Skip if we've already processed this image
                                            if file_id in processed_image_ids:
                                                continue

                                            # Mark as processed
                                            processed_image_ids.add(file_id)

                                            # Download the container file from OpenAI
                                            try:
                                                container_id = annotation.container_id

                                                # Use HTTP request directly (more reliable than SDK for binary content)
                                                url = f"https://api.openai.com/v1/containers/{container_id}/files/{file_id}/content"
                                                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
                                                img_response = requests.get(url, headers=headers, timeout=60)
                                                img_response.raise_for_status()
                                                image_bytes = img_response.content

                                                # Deduplicate by content hash (in case different file_ids point to same image)
                                                img_hash = hashlib.sha256(image_bytes).hexdigest()
                                                if img_hash in processed_image_hashes:
                                                    continue
                                                processed_image_hashes.add(img_hash)

                                                # Convert to base64 for storage
                                                image_b64 = base64.b64encode(image_bytes).decode('utf-8')

                                                # Add to last code execution's outputs
                                                if code_executions:
                                                    code_executions[-1]["outputs"].append({
                                                        "type": "image",
                                                        "image": {
                                                            "file_id": file_id,
                                                            "container_id": container_id,
                                                            "filename": annotation.filename if hasattr(annotation, 'filename') else None,
                                                            "b64_json": image_b64
                                                        }
                                                    })
                                            except Exception as img_error:
                                                print(f"Warning: Could not download image {file_id}: {img_error}")

        # Log usage
        # Responses API uses 'input_tokens' and 'output_tokens' (not prompt_tokens/completion_tokens)
        input_tokens = response.usage.input_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'input_tokens') else 0
        output_tokens = response.usage.output_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'output_tokens') else 0

        from config import calculate_cost
        model_name = get_current_model()
        cost_info = calculate_cost(model_name, input_tokens, output_tokens) if input_tokens and output_tokens else {"total_cost": 0}
        cost = cost_info["total_cost"]

        # Store cost info in session state for tracking
        if 'total_costs' not in st.session_state:
            st.session_state.total_costs = {}
        if model_name not in st.session_state.total_costs:
            st.session_state.total_costs[model_name] = 0.0
        st.session_state.total_costs[model_name] += cost

        log_enhanced_usage(
            page=THIS_PAGE,
            model_used=model_name,
            prompt=user_message[:1000],
            response=content[:1000],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            response_time_ms=response_time_ms,
            additional_metadata={
                "has_code_execution": len(code_executions) > 0,
                "num_files_attached": len(file_ids) if file_ids else 0,
                "api_type": "responses"
            }
        )

        return {
            "content": content,
            "code_executions": code_executions,
            "success": True
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in process_sandbox_request: {error_details}")
        return {
            "content": f"Error: {str(e)}",
            "code_executions": [],
            "success": False,
            "error": str(e)
        }

# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------

st.title("AI Sandbox")
st.markdown("""
**Secure Python code execution environment for data analysis and computational problem-solving.**

This tool helps you:
-  **Analyze Data**: Upload CSV/Excel files for instant statistical analysis and visualizations
-  **Process Images**: Upload graphs/charts to recreate them, extract data, or analyze visual information
-  **Solve Math Problems**: Get step-by-step solutions with executable Python code
-  **Create Visualizations**: Generate charts and graphs with matplotlib/seaborn
-  **Perform Computations**: Run complex calculations with numpy/scipy

All code runs in a secure 1GB sandbox. You can download any graphs or visualizations created.

*Powered by {model}*
""".format(model=MODELS[get_current_model()]["display_name"]))

# File Upload Section
st.markdown("---")
st.subheader("Upload Files (Optional)")
st.markdown("Upload files for analysis. Supports: CSV, Excel, images, text files, PDFs, and more.")

uploaded_files = st.file_uploader(
    "Choose files",
    accept_multiple_files=True,
    type=['csv', 'xlsx', 'xls', 'txt', 'json', 'tsv', 'dat', 'png', 'jpg', 'jpeg', 'pdf'],
    help="Upload files for analysis. The AI can read data files, analyze images, recreate graphs from screenshots, and more."
)

# Process uploaded files
current_file_ids = []
if uploaded_files:
    with st.spinner("Uploading files to sandbox..."):
        for uploaded_file in uploaded_files:
            file_id = upload_file_to_openai(uploaded_file)
            if file_id:
                current_file_ids.append(file_id)
                st.success(f"{uploaded_file.name} uploaded successfully!")
            else:
                st.error(f"Failed to upload {uploaded_file.name}")

# Store file IDs for this session
if current_file_ids:
    st.session_state.sandbox_file_ids = current_file_ids

# Chat Interface
st.markdown("---")
st.subheader(" Conversation")

# Display conversation history
for message in st.session_state.sandbox_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Render code executions if present
        if "code_executions" in message:
            for execution in message["code_executions"]:
                render_code_execution(execution["code"], execution.get("outputs", []))

# Chat input
if prompt := st.chat_input("Ask me to analyze data, solve problems, or create visualizations..."):
    # Add user message
    st.session_state.sandbox_messages.append({
        "role": "user",
        "content": prompt
    })

    # Process request (don't display yet - let history loop handle it)
    with st.spinner("Processing..."):
        result = process_sandbox_request(
            prompt,
            st.session_state.sandbox_file_ids if st.session_state.sandbox_file_ids else None
        )

        if result["success"]:
            # Save to history
            st.session_state.sandbox_messages.append({
                "role": "assistant",
                "content": result["content"],
                "code_executions": result["code_executions"]
            })
        else:
            # Save error to history
            st.session_state.sandbox_messages.append({
                "role": "assistant",
                "content": f"Error: {result.get('error', 'Unknown error')}"
            })

    # Rerun to display the new messages from history
    st.rerun()

# PDF Export Section
# ------------------
if len(st.session_state.sandbox_messages) > 0:
    st.markdown("---")
    with st.expander("Export Conversation to PDF"):
        st.markdown("Generate a polished PDF document of your AI Sandbox session with all code executions and visualizations embedded.")

        row = st.columns([2, 2])
        user_name = row[0].text_input("Enter your name:", key="sandbox_pdf_name")
        user_course = row[1].text_input("Enter your course name:", key="sandbox_pdf_course")

        if user_name != "" and user_course != "":
            # Calculate total cost from session state
            total_cost = st.session_state.get('total_costs', {}).get(get_current_model(), 0.0)

            try:
                from lib import chatpdf
                pdf_output_path = chatpdf.create_sandbox_pdf(
                    sandbox_messages=st.session_state.sandbox_messages,
                    user_name=user_name,
                    user_course=user_course,
                    total_cost=total_cost
                )

                with open(pdf_output_path, "rb") as file:
                    st.download_button(
                        label="Download PDF",
                        data=file,
                        file_name=f"{user_course}_{user_name}_sandbox.pdf",
                        mime="application/pdf",
                        width="stretch"
                    )
            except Exception as pdf_error:
                st.error(f"Error generating PDF: {str(pdf_error)}")
                import traceback
                st.code(traceback.format_exc())

# Import theme colors
from config import THEME_COLORS

# Render full sidebar first (navigation + model selector)
sidebar.render_sidebar()

# Now render Sandbox-specific settings (after model selector so it reflects current selection)
current_model = get_current_model()
current_model_name = MODELS[current_model]['display_name']

# Check if model supports temperature
no_temp_models = ["gpt-5-mini-2025-08-07", "o1", "o1-mini", "o1-preview"]
supports_temperature = not any(m in current_model for m in no_temp_models)

# Build settings HTML
temp_line = f"<strong>Temperature:</strong> {TEMPERATURE}<br>" if supports_temperature else ""

st.sidebar.markdown(
    f'<h3 style="color: {THEME_COLORS["primary"]};">Sandbox Settings</h3>',
    unsafe_allow_html=True
)
st.sidebar.markdown(f"""
<div style="background-color: {THEME_COLORS['info']}10; padding: 10px; border-radius: 5px; font-size: 0.9em;">
    <strong>Model:</strong> {current_model_name}<br>
    {temp_line}<strong>Max Tokens:</strong> {MAX_TOKENS}<br>
    <strong>Memory Limit:</strong> 1GB
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("")

# Clear conversation button
if st.sidebar.button("Clear Conversation", width="stretch"):
    st.session_state.sandbox_messages = []
    st.session_state.sandbox_file_ids = []
    st.session_state.sandbox_output_files = []
    st.session_state.sandbox_container_id = None  # Clear container to start fresh
    st.rerun()
