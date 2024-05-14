

# Importing the necessary libraries:
# ----------------------------------
# see https://python.langchain.com/docs/modules/model_io/chat/quick_start/
from langchain.prompts.chat import ChatPromptTemplate

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

import streamlit as st

# Function to generate chat completions based on different models
def generate_chat_completion(model, messages, temp=0, max_num_tokens=1000):
    """
    Function to generate chat completions, with reasonable defaults for traditional chat completions parameters.
    """
    
    if model == 'gpt-4o': 
        chat_model = ChatOpenAI(model="gpt-4o", temperature=temp, max_tokens=max_num_tokens)  
    elif model == 'gpt-3.5-turbo':
        chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=temp, max_tokens=max_num_tokens)
    elif model == "claude-3-opus-20240229": 
        chat_model = ChatAnthropic(model="claude-3-opus-20240229", temperature=temp, max_tokens=max_num_tokens)
    elif model == "claude-3-sonnet-20240229":
        chat_model = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=temp, max_tokens=max_num_tokens)
    elif model == "claude-3-haiku-20240307":
        chat_model = ChatAnthropic(model="claude-3-haiku-20240307", temperature=temp, max_tokens=max_num_tokens)
    elif model == "command-r-plus":
        chat_model = ChatCohere(model="command-r-plus", temperature=temp, max_tokens=max_num_tokens)
    elif model == "gemini-pro": 
        chat_model = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest", convert_system_message_to_human=True, temperature=temp, max_tokens=max_num_tokens)
    elif model == "mistral-small-latest": 
        chat_model = ChatMistralAI(model="mistral-small-latest", temperature=temp, max_tokens=max_num_tokens)
    elif model == "mistral-medium-latest":
        chat_model = ChatMistralAI(model="mistral-medium-latest", temperature=temp, max_tokens=max_num_tokens)
    elif model == "mistral-large-latest":
        chat_model = ChatMistralAI(model="mistral-large-latest", temperature=temp, max_tokens=max_num_tokens)
    elif model == "llama3-8b-8192":
        chat_model = ChatGroq(model="llama3-8b-8192", temperature=temp, max_tokens=max_num_tokens)
    elif model == "llama3-70b-8192":
        chat_model = ChatGroq(model="llama3-70b-8192", temperature=temp, max_tokens=max_num_tokens)
    elif model == "gemma-7b-it":
        chat_model = ChatGroq(model="gemma-7b-it", temperature=temp, max_tokens=max_num_tokens)
    else:
        raise ValueError(f"Model {model} is not supported.")  

    # generating the response
    chat_response = chat_model.invoke(messages)
    
    # extracting the token usage [work in progress]
    # for openai models
    if model in ['gpt-4o', 'gpt-3.5-turbo']:
        input_tokens = chat_response.response_metadata['token_usage']['prompt_tokens']
        output_tokens = chat_response.response_metadata['token_usage']['completion_tokens']
    # for anthropic models
    elif model in ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]:
        input_tokens = chat_response.response_metadata['usage']['input_tokens']
        output_tokens = chat_response.response_metadata['usage']['output_tokens']
    # for cohere models
    elif model == "command-r-plus":
        input_tokens = chat_response.response_metadata['token_count']['input_tokens']
        output_tokens = chat_response.response_metadata['token_count']['output_tokens']
    # for google models (there is no token usage metadata)
    elif model == "gemini-pro":
        input_tokens = None
        output_tokens = None
    # for mistral models
    elif model in ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"]:
        input_tokens = chat_response.response_metadata['token_usage']['prompt_tokens']
        output_tokens = chat_response.response_metadata['token_usage']['completion_tokens']
    # for groq models
    elif model in ["llama3-8b-8192", "llama3-70b-8192", "gemma-7b-it"]:
        input_tokens = chat_response.response_metadata['token_usage']['prompt_tokens']
        output_tokens = chat_response.response_metadata['token_usage']['completion_tokens']
    else:
        input_tokens = None
        output_tokens = None

    chat_response = chat_response.content

    return chat_response, input_tokens, output_tokens
# -----------------------------------------------------------------------------


# Function to process the generated chat
def process_messages(messages, model, temp = 0, max_num_tokens = 1000):
    response, input_tokens, output_tokens = generate_chat_completion(
        model=model,
        messages=messages,
        temp=temp,
        max_num_tokens=max_num_tokens
    )

    # Update the token counts for the specific model in session state
    if input_tokens is not None and output_tokens is not None:
        st.session_state.token_counts[model]['input_tokens'] += input_tokens
        st.session_state.token_counts[model]['output_tokens'] += output_tokens
    
    return response
