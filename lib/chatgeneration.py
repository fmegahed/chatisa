
"""
Enhanced chat generation module using centralized configuration.
Supports multiple LLM providers with automatic token tracking and cost calculation.
"""

# Importing the necessary libraries
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import streamlit as st
import time
from config import MODELS, calculate_cost, get_model_display_name
from lib.enhanced_usage_logger import log_enhanced_usage as log_usage

def generate_chat_completion(model, messages, temp=None, max_num_tokens=None):
    """
    Generate chat completions using centralized model configuration.
    
    Args:
        model (str): Model name from config.MODELS
        messages (list): Chat messages
        temp (float, optional): Temperature override
        max_num_tokens (int, optional): Max tokens override
        
    Returns:
        tuple: (response_text, input_tokens, output_tokens)
    """
    
    # Validate model exists in config
    if model not in MODELS:
        supported_models = list(MODELS.keys())
        raise ValueError(
            f"Model '{model}' is not supported. "
            f"Supported models: {', '.join(supported_models)}"
        )
    
    model_config = MODELS[model]
    provider = model_config["provider"]
    
    # Use config defaults if not specified
    if temp is None:
        temp = model_config["default_temperature"]

    # Always use the model's configured max_tokens to ensure proper operation
    # (some models like Gemini 3 Pro use reasoning tokens that count against the limit)
    max_num_tokens = model_config["max_tokens"]
    
    # Validate temperature range
    temp_min, temp_max = model_config["temperature_range"]
    if not (temp_min <= temp <= temp_max):
        st.warning(
            f"Temperature {temp} is outside recommended range "
            f"[{temp_min}, {temp_max}] for {model}"
        )
    
    # Models that don't support temperature parameter
    no_temperature_models = ["gpt-5-mini-2025-08-07", "o1", "o1-mini", "o1-preview"]

    # Initialize the appropriate chat model based on provider
    try:
        if provider == "openai":
            # Some OpenAI models don't support temperature
            if any(m in model for m in no_temperature_models):
                chat_model = ChatOpenAI(
                    model=model,
                    max_tokens=max_num_tokens
                )
            else:
                chat_model = ChatOpenAI(
                    model=model,
                    temperature=temp,
                    max_tokens=max_num_tokens
                )
        elif provider == "anthropic":
            chat_model = ChatAnthropic(
                model=model,
                temperature=temp,
                max_tokens=max_num_tokens
            )
        elif provider == "cohere":
            chat_model = ChatCohere(
                model=model,
                temperature=temp,
                max_tokens=max_num_tokens
            )
        elif provider == "google":
            chat_model = ChatGoogleGenerativeAI(
                model=model,
                temperature=temp,
                max_output_tokens=max_num_tokens
            )
        elif provider == "groq" or provider == "meta (via Groq)":
            chat_model = ChatGroq(
                model=model,
                temperature=temp,
                max_tokens=max_num_tokens
            )
        elif provider == "huggingface_inference":
            # Use HuggingFace Inference API
            from config import HUGGINGFACEHUB_API_TOKEN

            # Determine the correct task type for the model
            # Newer chat models use "conversational", older ones use "text-generation"
            task_type = "text-generation"  # Default

            # Models that require conversational task type
            conversational_models = [
                "gemma-3",
                "Phi-4",
                "Qwen3-VL",
                "Llama-4"
            ]

            if any(model_id in model for model_id in conversational_models):
                task_type = "conversational"

            # Configure endpoint parameters
            endpoint_kwargs = {
                "repo_id": model,
                "temperature": temp,
                "max_new_tokens": max_num_tokens,
                "huggingfacehub_api_token": HUGGINGFACEHUB_API_TOKEN,
                "task": task_type,
            }

            # Don't use provider="auto" for some models that have issues with it
            if not any(x in model for x in ["gpt2"]):
                endpoint_kwargs["provider"] = "auto"

            llm = HuggingFaceEndpoint(**endpoint_kwargs)
            chat_model = ChatHuggingFace(llm=llm)
        else:
            raise ValueError(f"Provider '{provider}' not implemented")

        # Generate the response with timing
        start_time = time.time()
        chat_response = chat_model.invoke(messages)
    except Exception as e:
        st.error(f"Failed to generate response with {model}: {str(e)}")
        raise
    response_time_ms = (time.time() - start_time) * 1000

    # Extract token usage based on provider
    input_tokens, output_tokens = extract_token_usage(chat_response, provider, model)
    
    # Calculate cost if tokens are available
    cost = 0.0
    if input_tokens is not None and output_tokens is not None:
        cost_info = calculate_cost(model, input_tokens, output_tokens)
        cost = cost_info["total_cost"]
        
        # Store cost info in page-specific session state for tracking
        current_page = getattr(st.session_state, 'cur_page', 'unknown')
        page_costs_key = f'total_costs_{current_page}'
        
        if page_costs_key not in st.session_state:
            st.session_state[page_costs_key] = {}
        if model not in st.session_state[page_costs_key]:
            st.session_state[page_costs_key][model] = 0.0
        st.session_state[page_costs_key][model] += cost
        
        # Also maintain global total_costs for backward compatibility with sidebar
        if 'total_costs' not in st.session_state:
            st.session_state.total_costs = {}
        if model not in st.session_state.total_costs:
            st.session_state.total_costs[model] = 0.0
        st.session_state.total_costs[model] += cost

    # Log the usage
    try:
        # Get current page from session state
        current_page = getattr(st.session_state, 'cur_page', 'unknown')
        
        # Extract user prompt (last user message)
        user_prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_prompt = str(msg.get("content", ""))[:1000]  # Limit length
                break
        
        log_usage(
            page=current_page,
            model_used=model,
            prompt=user_prompt,
            response=chat_response.content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            response_time_ms=response_time_ms,
            additional_metadata={
                "provider": provider,
                "temperature": temp,
                "max_tokens": max_num_tokens
            }
        )
    except Exception as log_error:
        # Don't let logging errors break the main functionality
        print(f"Warning: Failed to log usage: {log_error}")

    # Ensure content is a string (Gemini can return a list of content parts instead of
    # a string, causing "TypeError: can only concatenate str (not 'list') to str")
    content = chat_response.content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and 'text' in part:
                parts.append(part['text'])
            else:
                parts.append(str(part))
        content = "".join(parts)

    return content, input_tokens, output_tokens

def extract_token_usage(chat_response, provider, model):
    """
    Extract token usage from chat response based on provider.
    
    Args:
        chat_response: LangChain chat response object
        provider (str): Provider name (openai, anthropic, etc.)
        model (str): Model name
        
    Returns:
        tuple: (input_tokens, output_tokens)
    """
    try:
        if provider == "openai":
            usage = chat_response.response_metadata['token_usage']
            return usage['prompt_tokens'], usage['completion_tokens']
            
        elif provider == "anthropic":
            usage = chat_response.response_metadata['usage']
            return usage['input_tokens'], usage['output_tokens']
            
        elif provider == "cohere":
            usage = chat_response.response_metadata['token_count']
            return usage['input_tokens'], usage['output_tokens']
            
        elif provider == "groq" or provider == "meta (via Groq)":
            usage = chat_response.response_metadata['token_usage']
            return usage['prompt_tokens'], usage['completion_tokens']

        elif provider == "google":
            usage = chat_response.response_metadata.get('usage_metadata', {})
            return usage.get('prompt_token_count', None), usage.get('candidates_token_count', None)

        elif provider == "huggingface_inference":
            usage = chat_response.response_metadata['token_usage']
            return usage['prompt_tokens'], usage['completion_tokens']

        else:
            # For providers without token usage metadata
            return None, None
            
    except (KeyError, AttributeError) as e:
        st.warning(f"Could not extract token usage for {model}: {str(e)}")
        return None, None
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
