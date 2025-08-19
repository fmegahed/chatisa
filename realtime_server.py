"""
FastAPI server for OpenAI Realtime API integration with ChatISA.
Provides endpoints for session management and interview functionality.
"""

import json
import logging
import os
import time
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import uvicorn
import requests

# Import configuration
from config import (
    OPENAI_API_KEY, OPENAI_REALTIME_MODEL, DEFAULT_REALTIME_VOICE, 
    REALTIME_VOICES, APP_NAME, VERSION
)

logger = logging.getLogger(__name__)

# ==================== FastAPI App Setup ====================
app = FastAPI(
    title=f"{APP_NAME} Realtime API Server", 
    version=VERSION,
    description="Speech-to-speech functionality for ChatISA interview mentor"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging."""
    logger.info(f"Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        return JSONResponse(
            content={"error": "Internal server error"}, 
            status_code=500
        )

# ==================== Exception Handlers ====================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error for {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Invalid request format", "details": str(exc)}
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(f"HTTP error for {request.url}: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error for {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"}
    )

# ==================== Pydantic Models ====================
class SessionRequest(BaseModel):
    """Request model for creating ephemeral sessions."""
    model: Optional[str] = OPENAI_REALTIME_MODEL
    instructions: Optional[str] = None
    voice: Optional[str] = DEFAULT_REALTIME_VOICE

class InterviewRequest(BaseModel):
    """Request model for interview-specific sessions."""
    job_title: str
    candidate_grade: str
    candidate_major: str
    resume_text: str
    job_description: str
    voice: Optional[str] = DEFAULT_REALTIME_VOICE
    model: Optional[str] = OPENAI_REALTIME_MODEL

# ==================== API Endpoints ====================
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": f"{APP_NAME} Realtime API Server",
        "version": VERSION,
        "status": "running",
        "supported_voices": REALTIME_VOICES,
        "default_voice": DEFAULT_REALTIME_VOICE,
        "endpoints": {
            "session": "POST /session - Create realtime session",
            "interview": "POST /interview - Create interview session",
            "health": "GET /health - Health check",
            "voices": "GET /voices - List available voices"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": f"{APP_NAME} Realtime API Server",
        "version": VERSION,
        "openai_configured": bool(OPENAI_API_KEY)
    }

@app.get("/voices")
async def list_voices():
    """List available voices for the realtime API."""
    return {
        "voices": REALTIME_VOICES,
        "default": DEFAULT_REALTIME_VOICE,
        "descriptions": {
            "alloy": "Neutral, balanced voice",
            "echo": "Clear, professional tone",
            "fable": "Warm, expressive voice", 
            "onyx": "Deep, authoritative voice",
            "nova": "Bright, energetic voice",
            "shimmer": "Soft, gentle voice"
        }
    }

@app.post("/session")
async def create_ephemeral_session(request: SessionRequest) -> JSONResponse:
    """
    Create an ephemeral session token for OpenAI Realtime API.
    This token will be used by the frontend WebRTC client.
    """
    try:
        logger.info(f"Creating ephemeral session with model: {request.model or OPENAI_REALTIME_MODEL}")
        
        # Validate voice selection
        voice = request.voice or DEFAULT_REALTIME_VOICE
        if voice not in REALTIME_VOICES:
            logger.warning(f"Invalid voice '{voice}', using default '{DEFAULT_REALTIME_VOICE}'")
            voice = DEFAULT_REALTIME_VOICE
        
        # Default instructions for general use
        default_instructions = (
            "You are a helpful AI assistant for ChatISA, an educational platform. "
            "Speak naturally and professionally. Keep responses clear and concise. "
            "You are designed to help students with their educational needs."
        )
        
        session_data = {
            "model": request.model or OPENAI_REALTIME_MODEL,
            "voice": voice,
            "modalities": ["audio", "text"],
            "instructions": request.instructions or default_instructions,
        }
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Make direct HTTP request to OpenAI's realtime sessions endpoint
        response = requests.post(
            "https://api.openai.com/v1/realtime/sessions",
            json=session_data,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            session_result = response.json()
            
            response_data = {
                "client_secret": (
                    session_result.get("client_secret", {}).get("value") 
                    or session_result.get("client_secret")
                ),
                "model": request.model or OPENAI_REALTIME_MODEL,
                "session_id": session_result.get("id"),
                "voice": voice,
                "expires_at": session_result.get("expires_at")
            }
            
            logger.info("Ephemeral session created successfully")
            return JSONResponse(content=response_data, status_code=200)
        else:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return JSONResponse(
                content={"error": f"OpenAI API error: {response.status_code} - {response.text}"}, 
                status_code=response.status_code
            )
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error creating ephemeral session: {e}")
        return JSONResponse(
            content={"error": f"Network error: {str(e)}"}, 
            status_code=500
        )
    except Exception as e:
        logger.error(f"Error creating ephemeral session: {e}")
        return JSONResponse(
            content={"error": f"Session creation failed: {str(e)}"}, 
            status_code=500
        )

@app.post("/interview")
async def create_interview_session(request: InterviewRequest) -> JSONResponse:
    """
    Create a specialized interview session with custom instructions.
    This creates a session specifically configured for interview practice.
    """
    try:
        logger.info(f"Creating interview session for {request.job_title} position")
        
        # Validate voice selection
        voice = request.voice or DEFAULT_REALTIME_VOICE
        if voice not in REALTIME_VOICES:
            voice = DEFAULT_REALTIME_VOICE
        
        # Create specialized interview instructions
        interview_instructions = (
            f"You are an expert technical interviewer conducting a speech-to-speech interview for a {request.job_title} position. "
            f"Your interviewee is a {request.candidate_grade} student majoring in {request.candidate_major}.\n\n"
            f"Resume information:\n{request.resume_text[:1000]}...\n\n"  # Truncate to avoid token limits
            f"Job description:\n{request.job_description[:1000]}...\n\n"
            "VOICE BEHAVIOR:\n"
            "- Speak naturally like a professional interviewer\n"
            "- Use appropriate pauses and emphasis\n"
            "- Sound confident and engaging but not intimidating\n"
            "- Keep responses concise but thorough\n"
            "- Acknowledge the candidate's responses appropriately\n\n"
            "INTERVIEW STRUCTURE:\n"
            "1. Start with a warm greeting and brief introduction\n"
            "2. Conduct exactly 6 structured questions:\n"
            "   - Background question about interest in the position\n"
            "   - Business performance measurement question\n"
            "   - Technical skills assessment question\n"
            "   - Software knowledge question\n"
            "   - Situational teamwork question\n"
            "   - Behavioral soft skills question\n"
            "3. After all questions, provide comprehensive feedback\n\n"
            "IMPORTANT GUIDELINES:\n"
            "- Ask ONE question at a time and wait for the complete response\n"
            "- Be encouraging and professional throughout\n"
            "- At the end, provide specific feedback and a score out of 100\n"
            "- Keep the interview focused and structured\n"
            "- Adapt question difficulty to the candidate's academic level"
        )
        
        session_data = {
            "model": request.model or OPENAI_REALTIME_MODEL,
            "voice": voice,
            "modalities": ["audio", "text"],
            "instructions": interview_instructions,
        }
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Make direct HTTP request to OpenAI's realtime sessions endpoint
        response = requests.post(
            "https://api.openai.com/v1/realtime/sessions",
            json=session_data,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            session_result = response.json()
            
            # Generate a unique interview session ID for tracking
            interview_session_id = f"interview_{uuid.uuid4().hex[:8]}"
            
            response_data = {
                "client_secret": (
                    session_result.get("client_secret", {}).get("value") 
                    or session_result.get("client_secret")
                ),
                "model": request.model or OPENAI_REALTIME_MODEL,
                "session_id": session_result.get("id"),
                "interview_session_id": interview_session_id,
                "voice": voice,
                "job_title": request.job_title,
                "candidate_info": {
                    "grade": request.candidate_grade,
                    "major": request.candidate_major
                },
                "expires_at": session_result.get("expires_at")
            }
            
            logger.info(f"Interview session created successfully: {interview_session_id}")
            return JSONResponse(content=response_data, status_code=200)
        else:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return JSONResponse(
                content={"error": f"OpenAI API error: {response.status_code} - {response.text}"}, 
                status_code=response.status_code
            )
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error creating interview session: {e}")
        return JSONResponse(
            content={"error": f"Network error: {str(e)}"}, 
            status_code=500
        )
    except Exception as e:
        logger.error(f"Error creating interview session: {e}")
        return JSONResponse(
            content={"error": f"Interview session creation failed: {str(e)}"}, 
            status_code=500
        )


@app.options("/{full_path:path}")
async def options_handler(request: Request, response: Response):
    """Handle CORS preflight requests."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# ==================== Server Startup ====================
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=f"{APP_NAME} Realtime API Server")
    parser.add_argument("--https", action="store_true", help="Enable HTTPS with self-signed certificate")
    parser.add_argument("--port", type=int, default=5050, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Suppress uvicorn access logs for cleaner output
    uvicorn_logger = logging.getLogger("uvicorn.access")
    uvicorn_logger.setLevel(logging.WARNING)
    
    # Check API key configuration
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found in environment variables")
        logger.error("Please set your OpenAI API key in the .env file")
        exit(1)
    
    # Prepare uvicorn configuration
    uvicorn_config = {
        "app": "realtime_server:app",
        "host": args.host,
        "port": args.port,
        "reload": True,
        "log_level": "warning",
        "access_log": False
    }
    
    # Add SSL configuration if HTTPS is requested
    if args.https:
        logger.info("Starting server with HTTPS (self-signed certificate)")
        logger.warning("⚠️  Self-signed certificate will show security warnings in browser")
        logger.info("For production, use a proper SSL certificate from a CA")
        
        # Note: You would need to generate SSL certificates
        # For development, you can create self-signed certificates:
        # openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
        uvicorn_config.update({
            "ssl_keyfile": "key.pem",
            "ssl_certfile": "cert.pem"
        })
        
        print(f"🔒 Starting HTTPS server on https://{args.host}:{args.port}")
        print("📝 To generate self-signed certificates, run:")
        print("   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes")
    else:
        print(f"🌐 Starting HTTP server on http://{args.host}:{args.port}")
        print("⚠️  HTTP only works for localhost. Use --https for production deployment.")
    
    print(f"🚀 {APP_NAME} Realtime API Server v{VERSION}")
    print(f"📱 Supported voices: {', '.join(REALTIME_VOICES)}")
    print(f"🔧 Default voice: {DEFAULT_REALTIME_VOICE}")
    
    # Run the server
    uvicorn.run(**uvicorn_config)
