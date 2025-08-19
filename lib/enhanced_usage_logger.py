"""
Enhanced usage logging system for ChatISA application.
Uses JSON for better data structure and supports optional fields for privacy.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
import streamlit as st

from config import PROJECT_ROOT

# Create logs directory
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Consolidated privacy-compliant log file
ACTIVITY_LOG_FILE = LOGS_DIR / "activity_log.json"
ANALYTICS_CACHE_FILE = LOGS_DIR / "analytics_cache.json"

# Legacy file paths (for migration)
USAGE_LOG_FILE = LOGS_DIR / "usage_log.json"
SESSION_LOG_FILE = LOGS_DIR / "session_log.json"

def get_session_id() -> str:
    """Get or create a page-specific session ID for the current session."""
    import streamlit as st
    
    # Get current page
    current_page = getattr(st.session_state, 'cur_page', 'unknown')
    
    # Create page-specific session ID key
    page_session_key = f"session_id_{current_page}"
    
    # Generate page-specific session ID if it doesn't exist
    if page_session_key not in st.session_state:
        st.session_state[page_session_key] = str(uuid.uuid4())
    
    return st.session_state[page_session_key]

def load_json_log(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON log file, return empty list if file doesn't exist."""
    if not file_path.exists():
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def save_json_log(file_path: Path, data: List[Dict[str, Any]]):
    """Save data to JSON log file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    except IOError as e:
        print(f"Error saving log file {file_path}: {e}")

def log_activity(
    activity_type: str,
    page: str,
    data: Optional[Dict[str, Any]] = None,
    model_used: Optional[str] = None,
    tokens: Optional[Dict[str, int]] = None,
    cost: Optional[float] = None,
    performance: Optional[Dict[str, Any]] = None
):
    """
    Log any user activity in a privacy-compliant way.
    
    Args:
        activity_type: Type of activity ('ai_generation', 'page_visit', 'model_change', 'pdf_export', 'feature_use')
        page: Page where activity occurred
        data: Additional non-personal metadata
        model_used: AI model used (for ai_generation activities)
        tokens: Token usage dict with input/output/total
        cost: Cost of the activity
        performance: Performance metrics
    """
    try:
        activity_data = load_json_log(ACTIVITY_LOG_FILE)
        
        # Create privacy-compliant entry
        activity_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": get_session_id(),
            "activity_type": activity_type,
            "page": page,
            "data": data or {}
        }
        
        # Add AI-specific data if applicable
        if model_used:
            activity_entry["model_used"] = model_used
        if tokens:
            activity_entry["tokens"] = tokens
        if cost:
            activity_entry["cost"] = cost
        if performance:
            activity_entry["performance"] = performance
            
        activity_data.append(activity_entry)
        save_json_log(ACTIVITY_LOG_FILE, activity_data)
        
    except Exception as e:
        print(f"Error logging activity: {e}")

def log_enhanced_usage(
    page: str,
    model_used: str,
    prompt: str,
    response: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cost: Optional[float] = None,
    response_time_ms: Optional[float] = None,
    exported_to_pdf: bool = False,
    course_name: Optional[str] = None,  # Optional field for academic context
    academic_level: Optional[str] = None,
    major: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
):
    """
    Log a usage event with enhanced JSON structure.
    
    Args:
        page: Current page/interface name
        model_used: LLM model used for the interaction
        prompt: User's input/prompt (first 200 chars for privacy)
        response: AI's response (first 500 chars for privacy)
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        cost: Cost of the interaction in USD
        response_time_ms: Response time in milliseconds
        exported_to_pdf: Whether the session was exported to PDF
        course_name: Optional course name (for academic usage tracking)
        academic_level: Student's academic level
        major: Student's major
        additional_metadata: Any additional structured data
    """
    try:
        # Use privacy-compliant activity logging
        tokens = {
            "input": input_tokens or 0,
            "output": output_tokens or 0,
            "total": (input_tokens or 0) + (output_tokens or 0)
        }
        
        performance = {"response_time_ms": response_time_ms} if response_time_ms else None
        
        # Privacy-compliant: Only store metadata, no personal info
        data = {
            "prompt_length": len(prompt) if prompt else 0,
            "response_length": len(response) if response else 0,
            "exported_to_pdf": exported_to_pdf,
            "course_provided": bool(course_name),  # Just whether course was provided, not the name
            "academic_level": academic_level,  # This is not personal
            "major": major  # This is not personal
        }
        
        if additional_metadata:
            # Filter out any potential personal info
            safe_metadata = {k: v for k, v in additional_metadata.items() 
                           if not any(personal in k.lower() for personal in ['name', 'email', 'id', 'user'])}
            data.update(safe_metadata)
        
        log_activity(
            activity_type="ai_generation",
            page=page,
            data=data,
            model_used=model_used,
            tokens=tokens,
            cost=cost,
            performance=performance
        )
        
    except Exception as e:
        print(f"Error logging usage: {e}")

def log_session_action(action: str, page: str, data: Optional[Dict[str, Any]] = None):
    """
    Legacy function - now uses privacy-compliant activity logging.
    """
    # Filter out any personal information from data
    safe_data = {}
    if data:
        safe_data = {k: v for k, v in data.items() 
                    if not any(personal in k.lower() for personal in ['name', 'email', 'id', 'user'])}
        
        # Special handling for filename - just store extension and type info
        if 'filename' in data:
            filename = data['filename']
            safe_data['file_type'] = filename.split('.')[-1] if '.' in filename else 'unknown'
            safe_data['has_filename'] = True
    
    log_activity(
        activity_type=action,
        page=page,
        data=safe_data
    )

def log_page_visit(page: str, metadata: Optional[Dict[str, Any]] = None):
    """Log a page visit."""
    log_session_action("page_visit", page, metadata)

def log_model_selection(page: str, model: str, previous_model: Optional[str] = None):
    """Log model selection change."""
    log_session_action("model_change", page, {
        "new_model": model,
        "previous_model": previous_model
    })

def get_enhanced_usage_stats(days: int = 7) -> Dict[str, Any]:
    """
    Get comprehensive usage statistics for the specified number of days.
    
    Args:
        days: Number of days to look back
        
    Returns:
        Dictionary with comprehensive stats
    """
    try:
        from datetime import timedelta
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Load data
        usage_data = load_json_log(USAGE_LOG_FILE)
        session_data = load_json_log(SESSION_LOG_FILE)
        
        # Filter by date
        recent_usage = [
            entry for entry in usage_data 
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_date
        ]
        
        recent_sessions = [
            entry for entry in session_data
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_date
        ]
        
        # Calculate statistics
        stats = {
            "period_days": days,
            "total_interactions": len(recent_usage),
            "unique_sessions": len(set(entry["session_id"] for entry in recent_usage)),
            "total_cost": sum(entry["cost"] or 0 for entry in recent_usage),
            "total_tokens": {
                "input": sum(entry["tokens"]["input"] or 0 for entry in recent_usage),
                "output": sum(entry["tokens"]["output"] or 0 for entry in recent_usage)
            },
            "popular_models": {},
            "popular_pages": {},
            "academic_usage": {
                "courses_mentioned": set(),
                "levels": {},
                "majors": {}
            },
            "export_rate": 0,
            "average_response_time": 0
        }
        
        # Calculate popular models and pages
        for entry in recent_usage:
            model = entry["model_used"]
            page = entry["page"]
            
            stats["popular_models"][model] = stats["popular_models"].get(model, 0) + 1
            stats["popular_pages"][page] = stats["popular_pages"].get(page, 0) + 1
            
            # Academic context (only if provided)
            academic = entry.get("academic_context", {})
            if academic.get("course_name"):
                stats["academic_usage"]["courses_mentioned"].add(academic["course_name"])
            if academic.get("level"):
                level = academic["level"]
                stats["academic_usage"]["levels"][level] = stats["academic_usage"]["levels"].get(level, 0) + 1
            if academic.get("major"):
                major = academic["major"]
                stats["academic_usage"]["majors"][major] = stats["academic_usage"]["majors"].get(major, 0) + 1
        
        # Calculate export rate
        exports = sum(1 for entry in recent_usage if entry["actions"].get("exported_to_pdf"))
        stats["export_rate"] = (exports / len(recent_usage)) * 100 if recent_usage else 0
        
        # Calculate average response time
        response_times = [
            entry["performance"]["response_time_ms"] 
            for entry in recent_usage 
            if entry["performance"].get("response_time_ms")
        ]
        stats["average_response_time"] = sum(response_times) / len(response_times) if response_times else 0
        
        # Convert courses set to list for JSON serialization
        stats["academic_usage"]["courses_mentioned"] = list(stats["academic_usage"]["courses_mentioned"])
        
        # Sort popular items
        stats["popular_models"] = dict(sorted(stats["popular_models"].items(), key=lambda x: x[1], reverse=True))
        stats["popular_pages"] = dict(sorted(stats["popular_pages"].items(), key=lambda x: x[1], reverse=True))
        
        return stats
        
    except Exception as e:
        return {"error": f"Error calculating stats: {e}"}

def cleanup_old_logs(days_to_keep: int = 90):
    """Remove log entries older than specified days."""
    try:
        from datetime import timedelta
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        # Clean usage logs
        usage_data = load_json_log(USAGE_LOG_FILE)
        usage_data = [
            entry for entry in usage_data
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_date
        ]
        save_json_log(USAGE_LOG_FILE, usage_data)
        
        # Clean session logs
        session_data = load_json_log(SESSION_LOG_FILE)
        session_data = [
            entry for entry in session_data
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_date
        ]
        save_json_log(SESSION_LOG_FILE, session_data)
        
    except Exception as e:
        print(f"Error cleaning logs: {e}")

# Migration function from old CSV format (if needed)
def migrate_from_csv():
    """Migrate existing CSV logs to JSON format."""
    try:
        import csv
        
        old_csv_file = LOGS_DIR / "usage_log.csv"
        if not old_csv_file.exists():
            return
        
        usage_data = []
        with open(old_csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert CSV row to new JSON format
                entry = {
                    "timestamp": row.get("timestamp", ""),
                    "session_id": row.get("session_id", ""),
                    "page": row.get("page", ""),
                    "model_used": row.get("model_used", ""),
                    "prompt_preview": (row.get("prompt", "") or "")[:200],
                    "response_preview": (row.get("response", "") or "")[:500],
                    "tokens": {
                        "input": int(row.get("input_tokens", 0) or 0),
                        "output": int(row.get("output_tokens", 0) or 0)
                    },
                    "cost": float(row.get("cost", 0) or 0),
                    "performance": {
                        "response_time_ms": float(row.get("response_time_ms", 0) or 0)
                    },
                    "actions": {
                        "exported_to_pdf": row.get("exported_to_pdf", "").lower() == "true"
                    },
                    "academic_context": {},
                    "metadata": {}
                }
                usage_data.append(entry)
        
        save_json_log(USAGE_LOG_FILE, usage_data)
        print(f"Migrated {len(usage_data)} entries from CSV to JSON")
        
    except Exception as e:
        print(f"Error migrating from CSV: {e}")

# Backward compatibility aliases
def log_usage(*args, **kwargs):
    """Backward compatibility wrapper."""
    log_enhanced_usage(*args, **kwargs)

def log_pdf_export(page: str, data: Optional[Dict[str, Any]] = None):
    """Log PDF export action."""
    log_session_action("pdf_export", page, data)

def clean_personal_data_from_logs():
    """
    Clean any personal data from existing logs and migrate to privacy-compliant format.
    """
    try:
        # Clean and migrate existing logs
        if USAGE_LOG_FILE.exists():
            with open(USAGE_LOG_FILE, 'r') as f:
                old_usage = json.load(f)
            
            # Migrate to activity log format
            for entry in old_usage:
                if 'model_used' in entry:
                    log_activity(
                        activity_type="ai_generation",
                        page=entry.get('page', 'unknown'),
                        data={
                            'prompt_length': entry.get('prompt_length', 0),
                            'response_length': entry.get('response_length', 0)
                        },
                        model_used=entry.get('model_used'),
                        tokens=entry.get('tokens', {}),
                        cost=entry.get('cost'),
                        performance=entry.get('performance')
                    )
        
        if SESSION_LOG_FILE.exists():
            with open(SESSION_LOG_FILE, 'r') as f:
                old_session = json.load(f)
            
            # Clean and migrate session data
            for entry in old_session:
                action = entry.get('action', 'unknown')
                page = entry.get('page', 'unknown')
                data = entry.get('data', {})
                
                # Remove personal info
                safe_data = {k: v for k, v in data.items() 
                           if not any(personal in k.lower() for personal in ['name', 'email', 'id', 'user'])}
                
                # Add non-personal metadata
                if 'filename' in data:
                    filename = data['filename']
                    safe_data['file_type'] = filename.split('.')[-1] if '.' in filename else 'unknown'
                    safe_data['has_filename'] = True
                
                log_activity(
                    activity_type=action,
                    page=page,
                    data=safe_data
                )
        
        # Archive old files
        import shutil
        if USAGE_LOG_FILE.exists():
            shutil.move(str(USAGE_LOG_FILE), str(USAGE_LOG_FILE.with_suffix('.json.backup')))
        if SESSION_LOG_FILE.exists():
            shutil.move(str(SESSION_LOG_FILE), str(SESSION_LOG_FILE.with_suffix('.json.backup')))
            
        print("✓ Personal data cleaned and logs migrated to privacy-compliant format")
        
    except Exception as e:
        print(f"Error cleaning logs: {e}")

def get_usage_stats(days: int = 7) -> Dict[str, Any]:
    """Get usage statistics from privacy-compliant activity log."""
    try:
        from datetime import timedelta
        activity_data = load_json_log(ACTIVITY_LOG_FILE)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        recent_activities = [
            entry for entry in activity_data
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_date
        ]
        
        # Calculate statistics
        ai_generations = [a for a in recent_activities if a.get('activity_type') == 'ai_generation']
        page_visits = [a for a in recent_activities if a.get('activity_type') == 'page_visit']
        
        stats = {
            "total_activities": len(recent_activities),
            "ai_generations": len(ai_generations),
            "page_visits": len(page_visits),
            "unique_sessions": len(set(a["session_id"] for a in recent_activities)),
            "total_cost": sum(a.get("cost", 0) for a in ai_generations),
            "total_tokens": sum(a.get("tokens", {}).get("total", 0) for a in ai_generations),
            "popular_pages": {},
            "model_usage": {}
        }
        
        # Page popularity
        for activity in recent_activities:
            page = activity.get("page", "unknown")
            stats["popular_pages"][page] = stats["popular_pages"].get(page, 0) + 1
        
        # Model usage
        for activity in ai_generations:
            model = activity.get("model_used", "unknown")
            stats["model_usage"][model] = stats["model_usage"].get(model, 0) + 1
        
        return stats
        
    except Exception as e:
        print(f"Error getting usage stats: {e}")
        return {}