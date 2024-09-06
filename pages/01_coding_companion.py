

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
from lib import chatpdf, chatgeneration, sidebar

# Third-Party Libraries
from dotenv import load_dotenv
import streamlit as st


# -----------------------------------------------------------------------------


# Models:
# -------
models = [
  'gpt-4o', 'gpt-4o-mini', 
  'claude-3-5-sonnet-20240620',
  'command-r-plus',
  'gemma2-9b-it',
  'llama-3.1-8b-instant',
  'llama-3.1-70b-versatile' 
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


# Constant Values:
# ----------------
TEMPERATURE = 0


SYSTEM_PROMPT = f"""
You are an upbeat, encouraging tutor who helps undergraduate students majoring in business analytics understand concepts by explaining ideas and asking students questions. Start by introducing yourself to the student as their ChatISA Assistant who is happy to help them with any questions.

Only ask one question at a time. Ask them about the subject title and topic they want to learn about. Wait for their response.  Given this information, help students understand the topic by providing explanations, examples, and analogies. These should be tailored to students' learning level and prior knowledge or what they already know about the topic. When appropriate also provide them with code in both R (use tidyverse styling) and Python (use pandas whenever possible), showing them how to implement whatever concept they are asking about.

When you show R code, you must use:
  (a) library_name::function_name() syntax as this avoids conflicts in function names and makes it clear to the student where the function is imported from when there are multiple packages loaded. Based on this, do NOT use library() in the beginning of your code chunk and use if(require(library)==FALSE) install.packages(library), and
  (b) use the native pipe |> as your pipe operator.

On the other hand for Python, break chained methods into multiple lines using parentheses; for example, do NOT write df.groupby('Region')['Sales'].agg('sum') on one line.
"""
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Manage page tracking and associated session state
# -----------------------------------------------------------------------------
THIS_PAGE = "coding_companion"
if "cur_page" not in st.session_state:
    st.session_state.cur_page = THIS_PAGE

if ("token_counts" not in st.session_state) or (st.session_state.cur_page != THIS_PAGE):
    st.session_state.token_counts = {model: {"input_tokens": 0, "output_tokens": 0} for model in models}

if ("model_choice" not in st.session_state) or (st.session_state.cur_page != THIS_PAGE):
    st.session_state.model_choice = models[0]

if ("messages" not in st.session_state) or (st.session_state.cur_page != THIS_PAGE):
  st.session_state.messages = [{
    "role": "system",
    "content": SYSTEM_PROMPT
  }, {
    "role": "user", 
    "content": "Hi, I am an undergraduate student studying business analytics."
    }]

st.session_state.cur_page = THIS_PAGE
# -----------------------------------------------------------------------------

# The Streamlit Interface:
# ------------------------

# Streamlit Title:
# ----------------
st.set_page_config(page_title = "ChatISA Coding Companion", layout = "centered",page_icon='🤖')
st.markdown("## 🤖 ChatISA: Coding Companion 🤖")

# Dropdown to Select Model:
# -------------------------
st.sidebar.markdown("### Choose Your LLM")
model_choice = st.sidebar.selectbox(
    "Choose your LLM",
    models,
    index=models.index(st.session_state.model_choice),
    key='model_choice',
    label_visibility='collapsed'
)

# -----------------------------------------------------------------------------
# Render the sidebar
# -----------------------------------------------------------------------------
sidebar.render_sidebar()
# -----------------------------------------------------------------------------
    

# Main Window: Where the Chat is Invoked, Displayed, and Stored:
# --------------------------------------------------------------
for message in st.session_state.messages[2:]:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# Display chatbox input & process user input
if prompt := st.chat_input("Ask me to help you with your code and/or to explain analytical concepts."):
  # Store the user's prompt to memory
  st.session_state.messages.append({"role": "user", "content": prompt})
  # Display the user's prompt to the chat window
  st.chat_message("user").markdown(prompt)
  # Stream response from the LLM
  with st.chat_message("assistant"):
    
    # initializing the response objects
    message_placeholder = st.empty()
    full_response = ""
    input_token_count = 0
    output_token_count = 0
    
    # generating the response
    outputs = chatgeneration.generate_chat_completion(
      model = st.session_state.model_choice,
      messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
      temp = TEMPERATURE,
      max_num_tokens = 1000
    )
    
    # extracting the response, input tokens, and output tokens
    response, input_tokens, output_tokens = outputs
    full_response += response
    message_placeholder.markdown(full_response + "▌")
    
    # Update the token counts for the specific model in session state
    st.session_state.token_counts[st.session_state.model_choice]['input_tokens'] += input_tokens
    st.session_state.token_counts[st.session_state.model_choice]['output_tokens'] += output_tokens
  
  # Store the full response from the LLM in memory
  st.session_state.messages.append({"role": "assistant", "content": full_response})


# Generating the PDF from the Chat:
# ---------------------------------
with st.expander("Export Chat to PDF"):
  row = st.columns([2, 2])
  user_name = row[0].text_input("Enter your name:")
  user_course = row[1].text_input("Enter your course name:")
  if user_name != "" and user_course != "":
    pdf_output_path = chatpdf.create_pdf(chat_messages=st.session_state.messages, models = models, token_counts = st.session_state.token_counts, user_name=user_name, user_course=user_course)

    with open(pdf_output_path, "rb") as file:
      st.download_button(label="Download PDF", data=file, file_name=f"{user_course}_{user_name}_chatisa.pdf", mime="application/pdf", use_container_width=True)


# -----------------------------------------------------------------------------
