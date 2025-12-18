"""
Central configuration file for ChatISA application.
All shared parameters, settings, LLM configurations, and costs are defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# ==================== Versioning and Date ====================
DATE = "Dec 18, 2025"
VERSION = "5.0.3"
APP_NAME = "ChatISA"
APP_DESCRIPTION = "Educational AI Assistant with Multiple LLM Support"
PAGE_ICON = "assets/favicon.png"

# ==================== API Configuration ====================
# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ==================== LLM Model Configuration ====================
# Available models with their configurations
MODELS = {
    # OpenAI Models
    "gpt-5.2-2025-12-11": {
        "provider": "openai",
        "display_name": "GPT-5.2",
        "description": "Latest OpenAI GPT-5 model for chat completions",
        "cost_per_1k_input": 0.00175,  # USD per 1K input tokens
        "cost_per_1k_output": 0.014,  # USD per 1K output tokens
        "max_tokens": 128000,
        "context_window": 400000,
        "supports_vision": True,
        "supports_function_calling": True,
        "temperature_range": (0.0, 2.0),
        "default_temperature": 0.7,
        "open_weight": False,
        "tags": ["premium", "coding", "reasoning", "large_context", "transcription", "sandbox"]
    },
    "gpt-5-mini-2025-08-07": {
        "provider": "openai",
        "display_name": "GPT-5 Mini",
        "description": "Faster, more cost-effective OpenAI model",
        "cost_per_1k_input": 0.00025,
        "cost_per_1k_output": 0.002,
        "max_tokens": 128000,
        "context_window": 400000,
        "supports_vision": True,
        "supports_function_calling": True,
        "temperature_range": (0.0, 2.0),
        "default_temperature": 0.7,
        "open_weight": False,
        "tags": ["fast", "cost_effective", "coding", "transcription", "sandbox"]
    },
    
    # Anthropic Models
    "claude-sonnet-4-5-20250929": {
        "provider": "anthropic",
        "display_name": "Claude Sonnet 4.5",
        "description": "Anthropic's latest Claude model with excellent reasoning",
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "max_tokens": 64000,
        "context_window": 200000,
        "supports_vision": True,
        "supports_function_calling": True,
        "temperature_range": (0.0, 1.0),
        "default_temperature": 0.7,
        "open_weight": False,
        "tags": ["premium", "coding", "reasoning", "large_context"]
    },
    
    # Cohere Models
    "command-a-03-2025": {
        "provider": "cohere",
        "display_name": "Command A",
        "description": "Cohere's advanced language model",
        "cost_per_1k_input": 0.0025,
        "cost_per_1k_output": 0.01,
        "max_tokens": 4096,
        "context_window": 128000,
        "supports_vision": False,
        "supports_function_calling": True,
        "temperature_range": (0.0, 1.0),
        "default_temperature": 0.3,
        "open_weight": True,
        "tags": ["coding", "reasoning"]
    },

    # Google Models
    "gemini-3-pro-preview": {
        "provider": "google",
        "display_name": "Gemini 3 Pro Preview",
        "description": "Google's latest Gemini 3 Pro model with advanced reasoning and multimodal capabilities",
        "cost_per_1k_input": 0.002,  # $2.00 per 1M input tokens
        "cost_per_1k_output": 0.012,  # $12.00 per 1M output tokens
        "max_tokens": 65536,
        "context_window": 1048576,  # 1M tokens
        "supports_vision": True,
        "supports_function_calling": True,
        "temperature_range": (0.0, 2.0),
        "default_temperature": 0.7,
        "open_weight": False,
        "tags": ["premium", "multimodal", "reasoning", "large_context", "coding"]
    },
    "gemini-3-flash-preview": {
        "provider": "google",
        "display_name": "Gemini 3 Flash Preview",
        "description": "Google's fast and efficient Gemini 3 Flash model with multimodal support",
        "cost_per_1k_input": 0.0005,  # $0.50 per 1M input tokens
        "cost_per_1k_output": 0.003,  # $3.00 per 1M output tokens
        "max_tokens": 65536,
        "context_window": 1048576,  # 1M tokens
        "supports_vision": True,
        "supports_function_calling": True,
        "temperature_range": (0.0, 2.0),
        "default_temperature": 0.7,
        "open_weight": False,
        "tags": ["fast", "multimodal", "cost_effective", "large_context", "coding"]
    },
    
    # Groq Models (open-weight models via API)
    "llama-3.3-70b-versatile": {
        "provider": "meta (via Groq)",
        "display_name": "Llama 3.3 70B",
        "description": "Meta's Llama model optimized for versatile tasks",
        "cost_per_1k_input": 0.00059,
        "cost_per_1k_output": 0.00079,
        "max_tokens": 32768,
        "context_window": 131072,
        "supports_vision": False,
        "supports_function_calling": True,
        "temperature_range": (0.0, 2.0),
        "default_temperature": 0.7,
        "open_weight": True,
        "tags": ["fast", "coding", "cost_effective"]
    },
    "llama-3.1-8b-instant": {
        "provider": "meta (via Groq)",
        "display_name": "Llama 3.1 8B Instant",
        "description": "Fast and efficient Llama model",
        "cost_per_1k_input": 0.00005,
        "cost_per_1k_output": 0.00008,
        "max_tokens": 131072,
        "context_window": 131072,
        "supports_vision": False,
        "supports_function_calling": True,
        "temperature_range": (0.0, 2.0),
        "default_temperature": 0.7,
        "open_weight": True,
        "tags": ["fast", "cost_effective"]
    },
    
    # OpenAI Realtime Model (for speech-to-speech interviews)
    "gpt-4o-realtime-preview-2025-06-03": {
        "provider": "openai",
        "display_name": "GPT-4o Realtime",
        "description": "OpenAI's real-time speech model for interview conversations",
        "cost_per_1k_input": 0.04,
        "cost_per_1k_output": 0.08,
        "max_tokens": 4096,
        "context_window": 32000,
        "supports_vision": False,
        "supports_function_calling": False,
        "temperature_range": (0.6, 1.2),
        "default_temperature": 0.8,
        "realtime_only": True,
        "open_weight": False,
        "tags": ["realtime", "speech"]
    },
    
    # ==================== HuggingFace Inference API Models ====================

    # DeepSeek Models
    "deepseek-ai/DeepSeek-R1-0528": {
        "provider": "huggingface_inference",
        "display_name": "DeepSeek R1",
        "description": "DeepSeek's advanced reasoning model",
        "cost_per_1k_input": 0.0,  # Free tier
        "cost_per_1k_output": 0.0,
        "max_tokens": 8192,
        "context_window": 32768,
        "supports_vision": False,
        "supports_function_calling": True,
        "temperature_range": (0.0, 1.0),
        "default_temperature": 0.7,
        "inference_provider": "auto",
        "open_weight": True,
        "tags": ["reasoning", "coding", "cost_effective"]
    },
    "deepseek-ai/DeepSeek-V3": {
        "provider": "huggingface_inference",
        "display_name": "DeepSeek V3",
        "description": "DeepSeek's latest generation model",
        "cost_per_1k_input": 0.0,  # Free tier
        "cost_per_1k_output": 0.0,
        "max_tokens": 8192,
        "context_window": 64000,
        "supports_vision": False,
        "supports_function_calling": True,
        "temperature_range": (0.0, 1.0),
        "default_temperature": 0.7,
        "inference_provider": "auto",
        "open_weight": True,
        "tags": ["reasoning", "large_context", "cost_effective"]
    },
    
    # Qwen Models - Text Only
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {
        "provider": "huggingface_inference",
        "display_name": "Qwen 3 235B Instruct",
        "description": "Large-scale Qwen text model with advanced reasoning",
        "cost_per_1k_input": 0.0,  # Free tier
        "cost_per_1k_output": 0.0,
        "max_tokens": 8192,
        "context_window": 128000,
        "supports_vision": False,
        "supports_function_calling": True,
        "temperature_range": (0.0, 1.0),
        "default_temperature": 0.7,
        "inference_provider": "auto",
        "open_weight": True,
        "tags": ["reasoning", "large_context", "cost_effective"]
    },
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": {
        "provider": "huggingface_inference",
        "display_name": "Qwen 3 Coder 480B Instruct",
        "description": "Specialized coding model with advanced programming capabilities",
        "cost_per_1k_input": 0.0,  # Free tier
        "cost_per_1k_output": 0.0,
        "max_tokens": 8192,
        "context_window": 128000,
        "supports_vision": False,
        "supports_function_calling": True,
        "temperature_range": (0.0, 1.0),
        "default_temperature": 0.7,
        "inference_provider": "auto",
        "open_weight": True,
        "tags": ["coding", "large_context", "cost_effective"]
    },
    
    # OpenAI OSS Models
    "openai/gpt-oss-120b": {
        "provider": "huggingface_inference",
        "display_name": "GPT OSS 120B",
        "description": "OpenAI's open-source model with MoE architecture",
        "cost_per_1k_input": 0.0,  # Free tier
        "cost_per_1k_output": 0.0,
        "max_tokens": 4096,
        "context_window": 128000,
        "supports_vision": False,
        "supports_function_calling": True,
        "temperature_range": (0.0, 2.0),
        "default_temperature": 0.7,
        "inference_provider": "auto",
        "open_weight": True,
        "tags": ["reasoning", "large_context", "cost_effective"]
    },
    "openai/gpt-oss-20b": {
        "provider": "huggingface_inference",
        "display_name": "GPT OSS 20B",
        "description": "Compact OpenAI open-source model",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "max_tokens": 4096,
        "context_window": 128000,
        "supports_vision": False,
        "supports_function_calling": True,
        "temperature_range": (0.0, 2.0),
        "default_temperature": 0.7,
        "inference_provider": "auto",
        "open_weight": True,
        "tags": ["fast", "cost_effective", "coding"]
    },

    # Meta Llama-4 Models
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "provider": "huggingface_inference",
        "display_name": "Llama 4 Scout 17B",
        "description": "Meta's efficient multimodal model",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "max_tokens": 4096,
        "context_window": 128000,
        "supports_vision": True,
        "supports_function_calling": True,
        "temperature_range": (0.0, 2.0),
        "default_temperature": 0.7,
        "inference_provider": "auto",
        "open_weight": True,
        "tags": ["multimodal", "cost_effective"]
    },
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": {
        "provider": "huggingface_inference",
        "display_name": "Llama 4 Maverick 17B",
        "description": "Meta's high-capacity multimodal model",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "max_tokens": 4096,
        "context_window": 128000,
        "supports_vision": True,
        "supports_function_calling": True,
        "temperature_range": (0.0, 2.0),
        "default_temperature": 0.7,
        "inference_provider": "auto",
        "open_weight": True,
        "tags": ["multimodal", "reasoning", "large_context", "cost_effective"]
    }
}

# Default model selections by page
DEFAULT_MODELS = {
    "coding_companion": "claude-sonnet-4-5-20250929",
    "project_coach": "gpt-5.2-2025-12-11", 
    "exam_ally": "gpt-5.2-2025-12-11",
    "interview_mentor": "gpt-4o-realtime-preview-2025-06-03",
    "interview_mentor_transcription": "gpt-5.2-2025-12-11"
}

# Model groups for easy selection - core models across providers
MODEL_GROUPS = {
    "recommended": ["gpt-5.2-2025-12-11", "claude-sonnet-4-5-20250929", "gemini-3-flash-preview", "llama-3.3-70b-versatile", "command-a-03-2025"],
    "cost_effective": ["gpt-5-mini-2025-08-07", "gemini-3-flash-preview", "llama-3.1-8b-instant"],
    "openai": ["gpt-5.2-2025-12-11", "gpt-5-mini-2025-08-07"],
    "anthropic": ["claude-sonnet-4-5-20250929"],
    "google": ["gemini-3-pro-preview", "gemini-3-flash-preview"],
    "cohere": ["command-a-03-2025"],
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "vision_capable": [
        "gpt-5.2-2025-12-11",
        "claude-sonnet-4-5-20250929",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    ],
    "open_weight": [m for m, config in MODELS.items() if config.get("open_weight", False)]
}

# Model categories for organized UI selection
MODEL_CATEGORIES = {
    "commercial_api": {
        "display_name": "Commercial APIs",
        "description": "Premium hosted models with guaranteed SLAs",
        "models": ["gpt-5.2-2025-12-11", "gpt-5-mini-2025-08-07", "claude-sonnet-4-5-20250929", "gemini-3-pro-preview", "gemini-3-flash-preview", "command-a-03-2025"]
    },
    "hf_multimodal": {
        "display_name": "Multimodal (HF)",
        "description": "Vision and image processing capabilities",
        "models": [
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
        ]
    },
    "hf_text": {
        "display_name": "Text-Only (HF)",
        "description": "High-quality text generation via serverless API",
        "models": [
            "deepseek-ai/DeepSeek-R1-0528",
            "deepseek-ai/DeepSeek-V3",
            "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b"
        ]
    },
    "hosted_fast": {
        "display_name": "Fast Processing (Groq)",
        "description": "Ultra-fast processing via Groq hardware",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    }
}

# ==================== Page-Specific Model Configuration ====================
# Centralized configuration for which models each page should display
# This is the SINGLE SOURCE OF TRUTH for page model availability
PAGE_MODELS = {
    "coding_companion": {
        "include_all": True,  # Show all non-realtime models
        "exclude_tags": ["realtime", "speech"],
        "description": "All available models for educational comparison"
    },
    "project_coach": {
        "tags": ["reasoning", "large_context"],  # Models good at complex reasoning
        "min_context_window": 64000,  # Larger context for project discussions
        "description": "Models optimized for complex project guidance"
    },
    "exam_ally": {
        "include_all": True,  # All models can generate exam questions
        "exclude_tags": ["realtime", "speech"],
        "description": "Models for exam preparation and question generation"
    },
    "interview_mentor": {
        "tags": ["realtime", "speech"],  # Only realtime speech models
        "include_realtime": True,  # Explicitly include realtime models
        "description": "Real-time speech models for interview practice"
    },
    "interview_mentor_transcription": {
        "tags": ["transcription", "premium"],  # Models good at transcription
        "description": "Models for transcribing and summarizing interviews"
    },
    "ai_sandbox": {
        "specific_models": ["gpt-5.2-2025-12-11"],  # Only top-tier OpenAI model
        "description": "GPT-5.2 with code interpreter and image generation capabilities"
    },
    "ai_comparisons": {
        "include_all": True,  # Show all models for comparison
        "exclude_tags": ["realtime", "speech"],
        "description": "All available models for side-by-side comparison"
    }
}

# Helper functions for model filtering and retrieval
def get_page_models(page_name: str) -> list:
    """
    Get the list of available models for a specific page.

    This is the PRIMARY function pages should use to get their model list.
    When you update a model name in MODELS dict, this automatically propagates.

    Args:
        page_name (str): Page identifier (e.g., "coding_companion")

    Returns:
        list: List of model IDs available for this page

    Example:
        >>> from config import get_page_models
        >>> models = get_page_models("coding_companion")
        >>> # Returns all non-realtime models
    """
    if page_name not in PAGE_MODELS:
        # Default: return all non-realtime models
        return [m for m, cfg in MODELS.items() if not cfg.get("realtime_only", False)]

    page_config = PAGE_MODELS[page_name]
    available_models = []

    # Check for specific models list first (highest priority)
    if "specific_models" in page_config:
        return page_config["specific_models"]

    # Start with all models or filter by criteria
    if page_config.get("include_all", False):
        available_models = list(MODELS.keys())
    else:
        # Filter by tags if specified
        if "tags" in page_config:
            required_tags = page_config["tags"]
            for model_id, model_config in MODELS.items():
                model_tags = model_config.get("tags", [])
                # Model must have at least one of the required tags
                if any(tag in model_tags for tag in required_tags):
                    available_models.append(model_id)
        else:
            # No specific criteria, include all
            available_models = list(MODELS.keys())

    # Apply exclusions
    if "exclude_tags" in page_config:
        exclude_tags = page_config["exclude_tags"]
        available_models = [
            m for m in available_models
            if not any(tag in MODELS[m].get("tags", []) for tag in exclude_tags)
        ]

    # Apply realtime_only filter (exclude realtime unless explicitly included)
    if not page_config.get("include_realtime", False):
        available_models = [
            m for m in available_models
            if not MODELS[m].get("realtime_only", False)
        ]

    # Apply minimum context window filter
    if "min_context_window" in page_config:
        min_context = page_config["min_context_window"]
        available_models = [
            m for m in available_models
            if MODELS[m].get("context_window", 0) >= min_context
        ]

    # Sort by provider and display name for consistent ordering
    available_models.sort(key=lambda m: (
        MODELS[m].get("provider", ""),
        MODELS[m].get("display_name", m)
    ))

    return available_models

def get_models_by_tag(*tags: str) -> list:
    """
    Get all models that have at least one of the specified tags.

    Args:
        *tags: Variable number of tag strings

    Returns:
        list: List of model IDs matching the tags

    Example:
        >>> get_models_by_tag("coding", "reasoning")
        ['gpt-5.2-2025-12-11', 'claude-sonnet-4-5-20250929', ...]
    """
    matching_models = []
    for model_id, model_config in MODELS.items():
        model_tags = model_config.get("tags", [])
        if any(tag in model_tags for tag in tags):
            matching_models.append(model_id)
    return matching_models

def get_default_model_for_page(page_name: str) -> str:
    """
    Get the default model for a specific page.
    Falls back to the first available model if no default is configured.

    Args:
        page_name (str): Page identifier

    Returns:
        str: Default model ID for this page
    """
    # Check if there's a configured default
    if page_name in DEFAULT_MODELS:
        default = DEFAULT_MODELS[page_name]
        # Verify it's available for this page
        available = get_page_models(page_name)
        if default in available:
            return default

    # Fallback to first available model
    available = get_page_models(page_name)
    return available[0] if available else list(MODELS.keys())[0]

# ==================== Speech Configuration ====================
# OpenAI Realtime API settings for speech-to-speech functionality
OPENAI_REALTIME_MODEL = "gpt-4o-realtime-preview-2025-06-03"
REALTIME_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
DEFAULT_REALTIME_VOICE = "alloy"

# Speech synthesis settings
TTS_VOICES = REALTIME_VOICES  # Same as realtime for consistency
DEFAULT_TTS_VOICE = DEFAULT_REALTIME_VOICE
TTS_SPEED = 1.0  # Speech speed multiplier
TTS_MODEL = "tts-1"  # OpenAI TTS model

# ==================== Page Configuration ====================
# Page settings and prompts
PAGES = {
    "coding_companion": {
        "title": "Coding Companion",
        "icon": "Code",
        "description": "Programming help with educational context",
        "default_temperature": 0.25,
        "max_tokens": 6000,
        "system_prompt_template": """You are an expert programming tutor and coding companion for students. Your goal is to help students learn programming concepts and solve coding problems in an educational manner.

Guidelines:
1. Always explain your reasoning and approach
2. Provide clear, well-commented code examples
3. Encourage best practices and clean code
4. Ask clarifying questions when needed
5. Adapt your explanations to the student's level

Student Context: {context}
Current Task: {task}
"""
    },
    "project_coach": {
        "title": "Project Coach",
        "icon": "Project",
        "description": "Team project guidance with different coaching roles",
        "default_temperature": 0.4,
        "max_tokens": 8000,
        "coaching_roles": {
            "project_manager": "Focus on planning, timelines, and resource management",
            "technical_lead": "Provide technical guidance and architecture decisions",
            "scrum_master": "Facilitate agile processes and remove blockers",
            "mentor": "Offer general advice and learning opportunities"
        }
    },
    "exam_ally": {
        "title": "Exam Ally",
        "icon": "Exam",
        "description": "Exam preparation with PDF-based question generation",
        "default_temperature": 0.3,
        "max_tokens": 8000,
        "max_pdf_pages": 10,
        "question_types": ["multiple_choice", "short_answer", "essay", "true_false"]
    },
    "interview_mentor": {
        "title": "Interview Mentor",
        "icon": "Interview",
        "description": "Speech-to-speech technical interview preparation",
        "default_temperature": 0.25,
        "max_tokens": 8000,
        "max_pdf_pages": 2,
        "speech_enabled": True,
        "auto_summary": True,
        "interview_structure": {
            "total_questions": 6,
            "question_types": [
                "background",
                "business_performance",
                "technical_skills",
                "software_knowledge",
                "situational_teamwork",
                "behavioral_soft_skills"
            ]
        }
    },
    "ai_sandbox": {
        "title": "AI Sandbox",
        "icon": "Sandbox",
        "description": "Secure Python code execution environment for data analysis and computational problem-solving",
        "default_temperature": 0.7,
        "max_tokens": 4000,
        "memory_limit": "4g",
        "supports_code_interpreter": True,
        "supports_file_upload": True,
        "allowed_file_types": ["csv", "xlsx", "xls", "txt", "json", "tsv", "dat", "png", "jpg", "jpeg", "pdf"],
        "model": "gpt-5.2-2025-12-11",
        "system_instructions": """You are an advanced AI assistant with Python code execution capabilities in a secure sandbox environment.

When users ask you to:
- Analyze data, create visualizations, or perform calculations -> Write and execute Python code
- Upload files (CSV, Excel, images, PDFs, etc.) -> Read and analyze them with pandas, matplotlib, PIL, etc.
- Process images (graphs, charts, diagrams) -> Analyze, recreate, or extract data from them
- Solve math problems -> Show step-by-step solutions with code

Always:
1. Write clear, well-commented Python code
2. Use appropriate libraries (pandas, matplotlib, numpy, scipy, PIL, etc.)
3. Explain your approach before coding
4. Provide insights after showing results
5. For visualizations, save figures to files so users can download them
6. For image analysis, explain what you observe and how you're processing it

Be educational and thorough. You're helping students learn data analysis and computational thinking."""
    },
    "ai_comparisons": {
        "title": "AI Comparisons",
        "icon": "Compare",
        "description": "Compare AI model responses side-by-side with support for text, images, and PDFs",
        "default_temperature": 0.3,
        "max_tokens": 2000,
        "default_models": ["gpt-5.2-2025-12-11", "claude-sonnet-4-5-20250929", "Qwen/Qwen2.5-VL-32B-Instruct"],
        "supports_file_upload": True,
        "supports_images": True,
        "experimental": True,
        "logging_enabled": False
    }
}

# Academic levels and majors
ACADEMIC_LEVELS = [
    "Freshman",
    "Sophomore", 
    "Junior",
    "Senior",
    "Graduate Student"
]

MAJORS = [
    "Business Analytics",
    "Cybersecurity Management",
    "Information Systems",
    "Accounting",
    "Computer Science",
    "Data Science",
    "Economics",
    "Entrepreneurship",
    "Finance",
    "Management",
    "Marketing",
    "Real Estate",
    "Software Engineering",
    "Statistics"
]

# ==================== UI Configuration ====================
# Streamlit theme and styling
PAGE_CONFIG = {
    "page_title": f"{APP_NAME} - Educational AI Assistant",
    "page_icon": PAGE_ICON,
    "layout": "centered",
    "initial_sidebar_state": "expanded"
}

# Sidebar configuration
SIDEBAR_CONFIG = {
    "show_model_selector": True,
    "show_temperature_slider": True,
    "show_token_counter": True,
    "show_cost_calculator": True,
    "show_export_options": True
}

# Theme colors (Miami University colors)
THEME_COLORS = {
    "primary": "#c3142d",  # Miami Red
    "secondary": "#2c2c2c",  # Dark Grey
    "background": "#ffffff",  # White
    "success": "#45B7D1",
    "warning": "#FFA07A",
    "error": "#c3142d",  # Miami Red for errors
    "info": "#2c2c2c"  # Dark Grey for info
}

# ==================== Cost Calculation ====================
def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> dict:
    """Calculate the cost for a given model and token usage."""
    if model_name not in MODELS:
        return {"error": f"Unknown model: {model_name}"}
    
    model_config = MODELS[model_name]
    input_cost = (input_tokens / 1000) * model_config["cost_per_1k_input"]
    output_cost = (output_tokens / 1000) * model_config["cost_per_1k_output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
        "model": model_name,
        "currency": "USD"
    }

def get_model_display_name(model_name: str) -> str:
    """Get the display name for a model."""
    if model_name in MODELS:
        return MODELS[model_name]["display_name"]
    return model_name

def get_models_by_provider(provider: str) -> list:
    """Get all models from a specific provider."""
    return [model for model, config in MODELS.items() 
            if config["provider"] == provider]

def get_cheapest_models(limit: int = 3) -> list:
    """Get the cheapest models based on average cost per 1K tokens."""
    models_with_avg_cost = []
    for model, config in MODELS.items():
        avg_cost = (config["cost_per_1k_input"] + config["cost_per_1k_output"]) / 2
        models_with_avg_cost.append((model, avg_cost))
    
    models_with_avg_cost.sort(key=lambda x: x[1])
    return [model for model, _ in models_with_avg_cost[:limit]]

# ==================== Directory Structure ====================
# Project directories
PROJECT_ROOT = Path(__file__).parent
ASSETS_DIR = PROJECT_ROOT / "assets"
PAGES_DIR = PROJECT_ROOT / "pages"
LIB_DIR = PROJECT_ROOT / "lib"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Export and temporary files
TEMP_DIR = PROJECT_ROOT / "temp"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Ensure directories exist
for directory in [TEMP_DIR, EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== Logging Configuration ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = PROJECT_ROOT / "logs" / "chatisa.log"

# Create logs directory
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ==================== Performance Settings ====================
# Token limits and chunking
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TEMPERATURE = 0.3
MAX_CONTEXT_TOKENS = 100000  # Maximum context for long conversations

# File processing limits
MAX_FILE_SIZE_MB = 10
SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".md", ".docx"]

# Rate limiting and error handling
RATE_LIMIT_ENABLED = False
MAX_REQUESTS_PER_MINUTE = 60
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# Performance optimizations
ENABLE_CACHING = True
CACHE_DURATION_MINUTES = 60
COMPRESS_RESPONSES = True

# ==================== Security Settings ====================
# API key validation
REQUIRED_API_KEYS = {
    "OPENAI_API_KEY": ["gpt-5.2-2025-12-11", "gpt-5-mini-2025-08-07", "gpt-4o-realtime-preview-2025-06-03"],
    "ANTHROPIC_API_KEY": ["claude-sonnet-4-5-20250929"],
    "COHERE_API_KEY": ["command-a-03-2025"],
    "GOOGLE_API_KEY": ["gemini-3-pro-preview", "gemini-3-flash-preview"],
    "GROQ_API_KEY": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "HUGGINGFACEHUB_API_TOKEN": [
        "deepseek-ai/DeepSeek-R1-0528",
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    ]
}

def validate_api_keys() -> dict:
    """Validate that required API keys are present."""
    missing_keys = []
    available_models = []
    
    for key_name, models in REQUIRED_API_KEYS.items():
        if os.getenv(key_name):
            available_models.extend(models)
        else:
            missing_keys.append(key_name)
    
    return {
        "missing_keys": missing_keys,
        "available_models": available_models,
        "all_keys_present": len(missing_keys) == 0
    }

# ==================== Feature Flags ====================
FEATURES = {
    "speech_to_speech": True,
    "pdf_processing": True,
    "cost_tracking": True,
    "export_functionality": True,
    "multi_model_comparison": False,  # Future feature
    "voice_cloning": False,  # Future feature
    "real_time_collaboration": False  # Future feature
}

# ==================== System Info ====================
def get_system_info() -> dict:
    """Get current system configuration info."""
    key_validation = validate_api_keys()
    
    return {
        "app_name": APP_NAME,
        "version": VERSION,
        "date": DATE,
        "total_models": len(MODELS),
        "available_models": len(key_validation["available_models"]),
        "missing_api_keys": key_validation["missing_keys"],
        "features_enabled": sum(1 for enabled in FEATURES.values() if enabled),
        "total_features": len(FEATURES),
        "project_root": str(PROJECT_ROOT)
    }

def validate_environment():
    """Validate the complete system environment for production readiness."""
    validation_results = {
        "api_keys": validate_api_keys(),
        "directories": True,
        "models": True,
        "features": True,
        "security": True,
        "warnings": [],
        "errors": []
    }
    
    # Check directory structure
    required_dirs = [PROJECT_ROOT, ASSETS_DIR, PAGES_DIR, LIB_DIR, TEMP_DIR, EXPORTS_DIR]
    for directory in required_dirs:
        if not directory.exists():
            validation_results["directories"] = False
            validation_results["errors"].append(f"Missing directory: {directory}")
    
    # Check model availability
    if len(validation_results["api_keys"]["available_models"]) < 2:
        validation_results["warnings"].append("Limited models available - consider adding more API keys")
    
    # Check features
    disabled_features = [f for f, enabled in FEATURES.items() if not enabled]
    if disabled_features:
        validation_results["warnings"].append(f"Disabled features: {', '.join(disabled_features)}")
    
    return validation_results

def print_config_summary():
    """Print a comprehensive summary of the current configuration."""
    info = get_system_info()
    validation = validate_environment()
    
    print("="*60)
    print(f"{APP_NAME} v{info['version']} - Configuration Summary")
    print("="*60)
    print(f"Release Date: {info['date']}")
    print(f"Available Models: {info['available_models']}/{info['total_models']}")
    print(f"Active Features: {info['features_enabled']}/{info['total_features']}")
    print(f"Project Root: {info['project_root']}")
    
    # API Status
    if info['missing_api_keys']:
        print(f"\n  Missing API Keys: {', '.join(info['missing_api_keys'])}")
    else:
        print("\n All API keys configured")
    
    # Model recommendations
    print(f"\n Recommended Models: {', '.join(MODEL_GROUPS['recommended'][:3])}")
    print(f" Most Cost-Effective: {', '.join(get_cheapest_models()[:2])}")
    
    # Environment validation
    if validation["errors"]:
        print(f"\n Errors: {len(validation['errors'])}")
        for error in validation["errors"]:
            print(f"    {error}")
    
    if validation["warnings"]:
        print(f"\n  Warnings: {len(validation['warnings'])}")
        for warning in validation["warnings"]:
            print(f"    {warning}")
    
    if not validation["errors"]:
        print("\n System is production-ready!")
    
    print("="*60)

# ==================== Export this configuration ====================
__all__ = [
    'APP_NAME', 'VERSION', 'DATE', 'APP_DESCRIPTION',
    'MODELS', 'DEFAULT_MODELS', 'MODEL_GROUPS', 'MODEL_CATEGORIES', 'PAGE_MODELS', 'PAGES', 'ACADEMIC_LEVELS', 'MAJORS',
    'PAGE_CONFIG', 'SIDEBAR_CONFIG', 'THEME_COLORS', 'FEATURES',
    'calculate_cost', 'get_model_display_name', 'validate_api_keys', 'get_system_info', 'validate_environment', 'print_config_summary',
    'get_page_models', 'get_models_by_tag', 'get_default_model_for_page',  # Model filtering functions
    'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'COHERE_API_KEY', 'GOOGLE_API_KEY', 'GROQ_API_KEY', 'HUGGINGFACEHUB_API_TOKEN',
    'OPENAI_REALTIME_MODEL', 'REALTIME_VOICES', 'DEFAULT_REALTIME_VOICE', 'TTS_VOICES', 'DEFAULT_TTS_VOICE',
    'MAX_RETRIES', 'TIMEOUT_SECONDS', 'ENABLE_CACHING'
]
