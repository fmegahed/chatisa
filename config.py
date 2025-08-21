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
DATE = "August 20, 2025"
VERSION = "4.3.0"
APP_NAME = "ChatISA"
APP_DESCRIPTION = "Educational AI Assistant with Multiple LLM Support"

# ==================== API Configuration ====================
# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ==================== LLM Model Configuration ====================
# Available models with their configurations
MODELS = {
    # OpenAI Models
    "gpt-5-chat-latest": {
        "provider": "openai",
        "display_name": "GPT-5 Chat",
        "description": "Latest OpenAI GPT-5 model for chat completions",
        "cost_per_1k_input": 0.00125,  # USD per 1K input tokens
        "cost_per_1k_output": 0.01,  # USD per 1K output tokens
        "max_tokens": 16384,
        "context_window": 128000,
        "supports_vision": True,
        "supports_function_calling": True,
        "temperature_range": (0.0, 2.0),
        "default_temperature": 0.7,
        "open_weight": False
    },
    "gpt-5-mini-2025-08-07": {
        "provider": "openai",
        "display_name": "GPT-5 Mini",
        "description": "Faster, more cost-effective OpenAI model",
        "cost_per_1k_input": 0.00025,
        "cost_per_1k_output": 0.002,
        "max_tokens": 16384,
        "context_window": 400000,
        "supports_vision": True,
        "supports_function_calling": True,
        "temperature_range": (0.0, 2.0),
        "default_temperature": 0.7,
        "open_weight": False
    },
    
    # Anthropic Models
    "claude-sonnet-4-20250514": {
        "provider": "anthropic",
        "display_name": "Claude Sonnet 4",
        "description": "Anthropic's latest Claude model with excellent reasoning",
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "max_tokens": 64000,
        "context_window": 200000,
        "supports_vision": True,
        "supports_function_calling": True,
        "temperature_range": (0.0, 1.0),
        "default_temperature": 0.7,
        "open_weight": False
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
        "open_weight": True
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
        "open_weight": True
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
        "open_weight": True
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
        "open_weight": False
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
        "open_weight": True
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
        "open_weight": True
    },
    
    # Qwen Models
    "Qwen/Qwen2.5-VL-32B-Instruct": {
        "provider": "huggingface_inference", 
        "display_name": "Qwen 2.5 VL 32B Instruct",
        "description": "Advanced multimodal model with image and text capabilities",
        "cost_per_1k_input": 0.0,  # Free tier
        "cost_per_1k_output": 0.0,
        "max_tokens": 8192,
        "context_window": 32000,
        "supports_vision": True,
        "supports_function_calling": True,
        "temperature_range": (0.0, 1.0),
        "default_temperature": 0.7,
        "inference_provider": "auto",
        "open_weight": True
    },
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
        "open_weight": True
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
        "open_weight": True
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
        "open_weight": True
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
        "open_weight": True
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
        "open_weight": True
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
        "open_weight": True
    }
}

# Default model selections by page
DEFAULT_MODELS = {
    "coding_companion": "claude-sonnet-4-20250514",
    "project_coach": "gpt-5-chat-latest", 
    "exam_ally": "gpt-5-chat-latest",
    "interview_mentor": "gpt-4o-realtime-preview-2025-06-03",
    "interview_mentor_transcription": "gpt-5-chat-latest"
}

# Model groups for easy selection - core 4 models across providers
MODEL_GROUPS = {
    "recommended": ["gpt-5-chat-latest", "claude-sonnet-4-20250514", "llama-3.3-70b-versatile", "command-a-03-2025"],
    "cost_effective": ["gpt-5-mini-2025-08-07", "llama-3.1-8b-instant"],
    "openai": ["gpt-5-chat-latest", "gpt-5-mini-2025-08-07"],
    "anthropic": ["claude-sonnet-4-20250514"],
    "cohere": ["command-a-03-2025"],
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "vision_capable": [
        "gpt-5-chat-latest", 
        "claude-sonnet-4-20250514",
        "Qwen/Qwen2.5-VL-32B-Instruct",
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
        "models": ["gpt-5-chat-latest", "gpt-5-mini-2025-08-07", "claude-sonnet-4-20250514", "command-a-03-2025"]
    },
    "hf_multimodal": {
        "display_name": "Multimodal (HF)",
        "description": "Vision and image processing capabilities",
        "models": [
            "Qwen/Qwen2.5-VL-32B-Instruct",
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
        "icon": "💻",
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
        "icon": "🎯",
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
        "icon": "📝",
        "description": "Exam preparation with PDF-based question generation",
        "default_temperature": 0.3,
        "max_tokens": 8000,
        "max_pdf_pages": 10,
        "question_types": ["multiple_choice", "short_answer", "essay", "true_false"]
    },
    "interview_mentor": {
        "title": "Interview Mentor",
        "icon": "👔",
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
    "ai_comparisons": {
        "title": "AI Comparisons",
        "icon": "⚖️",
        "description": "Compare AI model responses side-by-side with support for text, images, and PDFs",
        "default_temperature": 0.3,
        "max_tokens": 2000,
        "default_models": ["gpt-5-chat-latest", "claude-sonnet-4-20250514", "Qwen/Qwen2.5-VL-32B-Instruct"],
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
    "Computer Science", 
    "Cybersecurity Management",
    "Data Science",
    "Information Systems",
    "Statistics",
    "Software Engineering"
]

# ==================== UI Configuration ====================
# Streamlit theme and styling
PAGE_CONFIG = {
    "page_title": f"{APP_NAME} - Educational AI Assistant",
    "page_icon": "assets/favicon.png",
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
    "OPENAI_API_KEY": ["gpt-5-chat-latest", "gpt-5-mini-2025-08-07", "gpt-4o-realtime-preview-2025-06-03"],
    "ANTHROPIC_API_KEY": ["claude-sonnet-4-20250514"],
    "COHERE_API_KEY": ["command-a-03-2025"],
    "GROQ_API_KEY": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "HUGGINGFACEHUB_API_TOKEN": [
        "deepseek-ai/DeepSeek-R1-0528",
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen2.5-VL-32B-Instruct",
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
        print(f"\n⚠️  Missing API Keys: {', '.join(info['missing_api_keys'])}")
    else:
        print("\n✅ All API keys configured")
    
    # Model recommendations
    print(f"\n📊 Recommended Models: {', '.join(MODEL_GROUPS['recommended'][:3])}")
    print(f"💰 Most Cost-Effective: {', '.join(get_cheapest_models()[:2])}")
    
    # Environment validation
    if validation["errors"]:
        print(f"\n❌ Errors: {len(validation['errors'])}")
        for error in validation["errors"]:
            print(f"   • {error}")
    
    if validation["warnings"]:
        print(f"\n⚠️  Warnings: {len(validation['warnings'])}")
        for warning in validation["warnings"]:
            print(f"   • {warning}")
    
    if not validation["errors"]:
        print("\n🎉 System is production-ready!")
    
    print("="*60)

# ==================== Export this configuration ====================
__all__ = [
    'APP_NAME', 'VERSION', 'DATE', 'APP_DESCRIPTION',
    'MODELS', 'DEFAULT_MODELS', 'MODEL_GROUPS', 'MODEL_CATEGORIES', 'PAGES', 'ACADEMIC_LEVELS', 'MAJORS',
    'PAGE_CONFIG', 'SIDEBAR_CONFIG', 'THEME_COLORS', 'FEATURES',
    'calculate_cost', 'get_model_display_name', 'validate_api_keys', 'get_system_info', 'validate_environment', 'print_config_summary',
    'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'COHERE_API_KEY', 'GROQ_API_KEY', 'HUGGINGFACEHUB_API_TOKEN',
    'OPENAI_REALTIME_MODEL', 'REALTIME_VOICES', 'DEFAULT_REALTIME_VOICE', 'TTS_VOICES', 'DEFAULT_TTS_VOICE',
    'MAX_RETRIES', 'TIMEOUT_SECONDS', 'ENABLE_CACHING'
]