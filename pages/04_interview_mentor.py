"""
Speech-to-Speech Interview Mentor using OpenAI Realtime API
Based on the example folder implementation
"""

import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
import requests
from streamlit_mic_recorder import mic_recorder
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
from pdf4llm import to_markdown

# Our Own Modules
from lib import chatpdf, sidebar, chatgeneration
from lib.speech import transcribe_audio, generate_speech
from config import OPENAI_API_KEY, OPENAI_REALTIME_MODEL, REALTIME_VOICES, DEFAULT_REALTIME_VOICE, VERSION, DATE

# Load environment variables
load_dotenv()

# -------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------
THIS_PAGE = "interview_mentor"
MAX_PDF_PAGES = 2
TEMPERATURE = 0.25

# Server configuration
SERVER_URL = "http://localhost:5050"
REALTIME_MODEL = OPENAI_REALTIME_MODEL
DEFAULT_VOICE = DEFAULT_REALTIME_VOICE
AVAILABLE_VOICES = REALTIME_VOICES

# -------------------------------------------------------------------------------
# Session State Management
# -------------------------------------------------------------------------------
if "cur_page" not in st.session_state:
    st.session_state.cur_page = THIS_PAGE

if ("submitted_speech" not in st.session_state) or (st.session_state.cur_page != THIS_PAGE):
    st.session_state.submitted_speech = False

if ("interview_active" not in st.session_state) or (st.session_state.cur_page != THIS_PAGE):
    st.session_state.interview_active = False

if ("interview_transcript" not in st.session_state) or (st.session_state.cur_page != THIS_PAGE):
    st.session_state.interview_transcript = []

if ("interview_completed" not in st.session_state) or (st.session_state.cur_page != THIS_PAGE):
    st.session_state.interview_completed = False

if ("interview_token_usage" not in st.session_state) or (st.session_state.cur_page != THIS_PAGE):
    st.session_state.interview_token_usage = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "audio_input_tokens": 0,
        "text_input_tokens": 0,
        "audio_output_tokens": 0,
        "text_output_tokens": 0
    }


# Purge interview state if coming from a different page
if (st.session_state.cur_page != THIS_PAGE) and ("interview_submission" in st.session_state):
    del st.session_state.interview_submission

st.session_state.cur_page = THIS_PAGE

# -------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------
@st.cache_data(ttl=30)
def check_realtime_server_health():
    """Check if the realtime server is running."""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# -------------------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------------------
st.set_page_config(
    page_title="ChatISA: Interview Mentor", 
    layout="centered", 
    page_icon="assets/favicon.png"
)
st.markdown("## 👔 ChatISA: Interview Mentor")

# -------------------------------------------------------------------------------
# Custom Sidebar for Interview Mentor (both speech-to-speech and transcription modes)
# -------------------------------------------------------------------------------
st.sidebar.markdown(f"""
### 👔 ChatISA v{VERSION}
**Interview Mentor**  
*{DATE}*

### Status
✅ **Dual Mode Support**

### Maintained By 
  - [Fadel Megahed](https://miamioh.edu/fsb/directory/?up=/directory/megahefm)   
  - [Joshua Ferris](https://miamioh.edu/fsb/directory/?up=/directory/ferrisj2)

### Key Features
  - Speech-to-speech conversation
  - Voice-to-text transcription mode
  - Multiple AI model support
  - Professional interview practice
  - Resume-based questioning
  - PDF export (transcription mode)

### Support & Funding
  - Farmer School of Business
  - US Bank  
  - Raymond E. Glos Professorship
""")

# -------------------------------------------------------------------------------
# First Screen: Submission Form
# -------------------------------------------------------------------------------
if not st.session_state.submitted_speech:
    st.title("🎤 ChatISA: Interview Mentor")
    
    # Toggle for speech-to-speech vs transcription mode
    col_toggle1, col_toggle2 = st.columns([1, 3])
    with col_toggle1:
        use_transcription = st.toggle("Use Transcription Mode", value=False, key="transcription_mode")
    
    if use_transcription:
        st.markdown(
            "**Interactive interview preparation with voice transcription!** This tool allows you to speak your responses, "
            "which will be transcribed to text for the AI interviewer. You can choose from multiple AI models "
            "and optionally enable AI voice responses. Upload your resume and job description to get "
            "personalized interview questions."
        )
    else:
        st.markdown(
            "**Experience a truly interactive interview preparation!** This advanced tool uses OpenAI's Realtime API "
            "for natural speech-to-speech conversation. Speak naturally with an AI interviewer who will ask you "
            "personalized questions based on your resume and the job description. "
            "The AI will respond with natural speech, creating a realistic interview experience."
        )
    
    # Check server status only for speech-to-speech mode
    if not use_transcription:
        server_healthy = check_realtime_server_health()
        if not server_healthy:
            st.error("❌ **Voice Server Offline** - Please start the realtime server to use speech-to-speech functionality")
            st.code("python realtime_server.py", language="bash")
            st.info("💡 **Alternative:** Toggle to 'Use Transcription Mode' above for voice-to-text interaction")
            st.stop()
        
        st.success("✅ **Voice Server Online** - Ready for speech-to-speech interview")
    else:
        st.success("✅ **Transcription Mode Active** - Ready for voice-to-text interview")
        
        # Model selection for transcription mode
        from config import DEFAULT_MODELS
        st.markdown("### AI Model Selection")
        transcription_models = ['gpt-5-chat-latest', 'gpt-5-mini-2025-08-07', 'claude-sonnet-4-20250514', 'command-a-03-2025', 'llama-3.3-70b-versatile']
        
        # Get default for transcription mode
        transcription_default = DEFAULT_MODELS.get('interview_mentor_transcription', 'gpt-5-chat-latest')
        default_index = transcription_models.index(transcription_default) if transcription_default in transcription_models else 0
        
        selected_model = st.selectbox(
            "Choose AI Model for Interview:",
            transcription_models,
            index=default_index,  # Use config default
            key='transcription_model_choice',
            help="Select the AI model for conducting the interview in transcription mode"
        )
    
    col1, col2 = st.columns(2, gap='large')
    
    with col1:
        st.markdown("### Interviewee Information")
        grade = st.selectbox(
            "What is your Current Grade Level?", 
            ["Freshman", "Sophomore", "Junior", "Senior", "Graduate Student"],
            index=3,
            key='grade_speech',
            help="What is your current grade level?"
        )
        major = st.selectbox(
            "What is your Major?",
            ["Business Analytics", "Computer Science", "Cybersecurity Management",
             "Data Science", "Information Systems", "Statistics", "Software Engineering"],
            index=0,
            key='major_speech',
            help="What is your current major?"
        )
        raw_resume = st.file_uploader(
            "Upload your Resume",
            type=['pdf'],
            key='resume_speech',
            help="Upload your resume in PDF format. This will be used to generate interview questions."
        )
    
    with col2:
        st.markdown("### Job Information")
        job_title = st.text_input(
            "Input the Job Title",
            value="",
            key='job_title_speech',
            help="What is the job title you are interviewing for?",
            placeholder="Business Analyst"
        )
        job_description = st.text_area(
            "Paste the Job Description",
            value="",
            key='job_description_speech',
            help="Paste the job description for the position you are interviewing for.",
            placeholder="As a Business Analyst, you will be responsible for analyzing data...",
            height=300
        )
    
    # Voice settings - different for each mode
    if not use_transcription:
        st.markdown("### Voice Settings")
        col3, col4 = st.columns(2)
        with col3:
            chosen_voice = st.selectbox(
                "AI Interviewer Voice:",
                AVAILABLE_VOICES,
                index=0,
                key='chosen_voice_speech',
                help="Select the AI voice for the interviewer"
            )
        with col4:
            response_speed = st.slider(
                "Response Speed (seconds):",
                min_value=1, max_value=5, value=2,
                key='response_speed_speech',
                help="How quickly the AI should respond after you stop speaking"
            )
    else:
        st.markdown("### Audio Settings")
        ai_audio_enabled = st.checkbox("Enable AI voice responses", value=False, key='ai_audio_transcription')
        if ai_audio_enabled:
            chosen_voice = st.selectbox(
                "AI Voice:",
                ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                index=0,
                key='chosen_voice_transcription',
                help="Select the AI voice for responses (optional)"
            )
        else:
            chosen_voice = "alloy"  # Default fallback
    
    button_text = 'Start Transcription Interview' if use_transcription else 'Start Speech-to-Speech Interview'
    if st.button(button_text, type="primary", use_container_width=True):
        if all([grade, major, raw_resume, job_title, job_description]):
            # Process the resume
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(raw_resume.getvalue())
                tmp_path = tmp.name

            infile = PdfReader(tmp_path, "rb")
            if len(infile.pages) > MAX_PDF_PAGES:
                output = PdfWriter()
                for i in range(MAX_PDF_PAGES):
                    output.add_page(infile.pages[i])
                with open(tmp_path, "wb") as f:
                    output.write(f)

            resume_text = to_markdown(tmp_path)
            os.unlink(tmp_path)

            # Create interview instructions for the realtime API
            interview_instructions = (
                f"You are an expert technical interviewer conducting a speech-to-speech interview for a {job_title} position. "
                f"Your interviewee is a {grade} student majoring in {major}.\n\n"
                f"Resume information:\n{resume_text}\n\n"
                f"Job description:\n{job_description}\n\n"
                "VOICE BEHAVIOR:\n"
                "- Speak naturally like a professional interviewer\n"
                "- Use appropriate pauses and emphasis\n"
                "- Sound confident and engaging\n"
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
                "IMPORTANT:\n"
                "- Ask ONE question at a time and wait for the response\n"
                "- Be encouraging and professional throughout\n"
                "- At the end, provide specific feedback and a score out of 100\n"
                "- Keep the interview focused and structured"
            )

            # Store submission details in session state
            submission_data = {
                'grade': grade,
                'major': major,
                'resume_text': resume_text,
                'job_title': job_title,
                'job_description': job_description,
                'chosen_voice': chosen_voice,
                'use_transcription': use_transcription
            }
            
            if use_transcription:
                submission_data.update({
                    'model_choice': selected_model,
                    'ai_audio_enabled': ai_audio_enabled,
                    'interview_mode': 'transcription'
                })
            else:
                submission_data.update({
                    'response_speed': response_speed,
                    'interview_instructions': interview_instructions,
                    'interview_mode': 'speech_to_speech'
                })
            
            st.session_state['interview_submission'] = submission_data
            
            mode_text = 'transcription interview' if use_transcription else 'speech-to-speech interview'
            st.success(f'Interview setup complete! Starting {mode_text}...')
            st.session_state.submitted_speech = True
            st.rerun()
        else:
            st.error('Please fill in all fields before starting the interview.')

# -------------------------------------------------------------------------------
# Second Screen: Speech-to-Speech Interview
# -------------------------------------------------------------------------------
if st.session_state.submitted_speech:
    # Retrieve submission details
    submission = st.session_state.interview_submission
    
    if submission.get('interview_mode') == 'transcription':
        st.title("🎤 Live Transcription Interview")
    else:
        st.title("🎤 Live Speech-to-Speech Interview")
    
    # Display interview details
    with st.expander("📋 Interview Details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Position:** {submission['job_title']}")
            st.write(f"**Grade:** {submission['grade']}")
            st.write(f"**Major:** {submission['major']}")
        with col2:
            if submission.get('interview_mode') == 'transcription':
                st.write(f"**Model:** {submission.get('model_choice', 'gpt-5-chat-latest')}")
                st.write(f"**AI Audio:** {'Enabled' if submission.get('ai_audio_enabled', False) else 'Disabled'}")
                if submission.get('ai_audio_enabled'):
                    st.write(f"**Voice:** {submission['chosen_voice']}")
                st.write(f"**Mode:** Voice-to-Text Interview")
            else:
                st.write(f"**Voice:** {submission['chosen_voice']}")
                st.write(f"**Response Speed:** {submission.get('response_speed', 2)}s")
                st.write(f"**Mode:** Speech-to-Speech Interview")
    
    # Interview Tips
    with st.expander("💡 Interview Tips", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if submission.get('interview_mode') == 'transcription':
                st.markdown("""
                **During the Interview:**
                - Speak clearly for accurate transcription
                - Wait for the AI response before continuing
                - Answer questions thoroughly but concisely
                - Review transcribed text for accuracy
                """)
            else:
                st.markdown("""
                **During the Interview:**
                - Speak clearly and at a normal pace
                - Wait for the AI to finish speaking before responding
                - Answer questions thoroughly but concisely
                - Ask for clarification if needed
                """)
        
        with col2:
            if submission.get('interview_mode') == 'transcription':
                st.markdown("""
                **Technical Notes:**
                - Keep your browser tab active
                - Ensure stable internet connection
                - Use a good quality microphone
                - Speak in a quiet environment
                """)
            else:
                st.markdown("""
                **Technical Notes:**
                - Keep your browser tab active for best performance
                - Ensure stable internet connection
                - Use headphones to prevent audio feedback
                - Speak in a quiet environment
                """)
    
    # Mode-specific interface
    if submission.get('interview_mode') == 'transcription':
        # Transcription Mode Interface
        st.success("✅ **Transcription Mode Active** - Ready for voice-to-text interview")
        
        # Purge messages if coming from a different page (for transcription mode)
        if (st.session_state.cur_page != THIS_PAGE) and ("messages" in st.session_state):
            del st.session_state.messages
        
        # Initialize system prompt and messages for transcription mode
        if "messages" not in st.session_state:
            SYSTEM_PROMPT = (
                f"You are an expert technical interviewer conducting an interview for a {submission['job_title']} position. "
                f"Your interviewee is a {submission['grade']} student majoring in {submission['major']}.\n\n"
                f"Resume information:\n{submission['resume_text']}\n\n"
                f"Job description:\n{submission['job_description']}\n\n"
                "Instructions:\n"
                "1. Analyze the resume and job description thoroughly to understand how the candidate's qualifications match the position requirements.\n"
                "2. Conduct a structured interview with six questions, asking one question at a time.\n"
                "3. Wait for the candidate to answer each question before proceeding to the next question.\n"
                "4. Be concise, professional, and constructive throughout the interview.\n\n"
                "Question structure:\n"
                "1. Background question about the candidate's interest in the position.\n"
                "2. Question about how the candidate would measure business performance at the company.\n"
                "3. Technical question assessing skills related to job requirements.\n"
                "4. Technical question assessing software knowledge related to job requirements.\n"
                "5. Situational question assessing teamwork abilities and handling of difficult situations.\n"
                "6. Behavioral question screening for soft skills.\n\n"
                "At the end of the interview:\n"
                "- Thank the candidate for their time.\n"
                "- Provide specific, actionable feedback on their performance.\n"
                "- Include a performance score out of 100.\n"
                "- Format feedback as: summary of interview (question/answer), positive feedback, constructive criticism with improvement advice, and overall score."
            )
            
            st.session_state.messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Hello, I am excited about this opportunity."}
            ]
        
        # Display chat messages (excluding system prompt)
        with st.container():
            for message in st.session_state.messages[2:]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Voice input interface for transcription mode
        num_assistant_messages = sum(1 for m in st.session_state.messages if m["role"] == "assistant")
        if num_assistant_messages < 15: 
            st.markdown("#### Resume the interview by clicking 'Press to Speak' and then 'Stop Recording' when done.")
            audio_data = mic_recorder(
                start_prompt="Press to Speak", 
                stop_prompt="Press to Stop Recording", 
                format="wav", 
                key="mic_recorder"
            )

            if audio_data:
                user_text = transcribe_audio(audio_data["bytes"])
                st.session_state.messages.append({"role": "user", "content": user_text})
                with st.chat_message("user"):
                    st.markdown(user_text)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    outputs = chatgeneration.generate_chat_completion(
                        model=submission['model_choice'],
                        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                        temp=0.25,
                        max_num_tokens=6000
                    )
                    response, input_tokens, output_tokens = outputs
                    message_placeholder.markdown(response)

                    # Generate audio for AI response if enabled
                    if submission.get('ai_audio_enabled', False):
                        with st.spinner("Generating audio response..."):
                            # Set the chosen voice in session state for the generate_speech function
                            st.session_state.chosen_voice = submission['chosen_voice']
                            ai_audio = generate_speech(response)
                            st.audio(ai_audio, format="audio/mp3")
                
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Add export functionality for transcription mode
        st.markdown("---")
        with st.expander("📄 Export Interview to PDF", expanded=False):
            col1, col2 = st.columns(2)
            user_name = col1.text_input("Enter your name:", key="trans_name")
            user_name = user_name.replace(" ", "_")
            company_name = col2.text_input("Enter company name:", key="trans_company")
            company_name = company_name.replace(" ", "_")

            if user_name and company_name:
                # Create models list and token counts for PDF
                models = ['gpt-5-chat-latest', 'gpt-5-mini-2025-08-07', 'claude-sonnet-4-20250514', 'command-a-03-2025', 'llama-3.3-70b-versatile']
                token_counts = {model: {"input_tokens": 0, "output_tokens": 0} for model in models}
                
                # Add token counts for the selected model (approximate)
                if submission['model_choice'] in token_counts:
                    token_counts[submission['model_choice']] = {"input_tokens": 2000, "output_tokens": 3000}
                
                pdf_output_path = chatpdf.create_pdf(
                    chat_messages=st.session_state.messages, 
                    models=models, 
                    token_counts=token_counts, 
                    user_name=user_name, 
                    user_course=company_name
                )
                with open(pdf_output_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Interview PDF", 
                        data=file, 
                        file_name=f"{company_name}_{user_name}_interview.pdf", 
                        mime="application/pdf", 
                        use_container_width=True
                    )
    
    else:
        # Speech-to-Speech Mode Interface
        # Server status check
        server_healthy = check_realtime_server_health()
        if not server_healthy:
            st.error("❌ **Voice Server Offline** - Please start the realtime server")
            st.code("python realtime_server.py", language="bash")
            if st.button("🔄 Retry Connection"):
                st.rerun()
            st.stop()
        
        # Interview status
        if st.session_state.interview_active:
            st.success("🔴 **LIVE INTERVIEW** - Speak naturally with the AI interviewer")
        else:
            st.info("🎤 **Ready to Start** - Click 'Start Interview' to begin your speech-to-speech interview")
    
        # Voice Chat Interface with Interview-specific instructions (Speech-to-Speech Mode Only)
        # Add JavaScript to listen for messages from the iframe
        st.markdown("""
        <script>
        window.addEventListener('message', function(event) {
            if (event.data.type === 'transcript_update') {
                // Store transcript data
                console.log('Transcript update:', event.data.data);
            } else if (event.data.type === 'token_usage_update') {
                // Store token usage data
                console.log('Token usage update:', event.data.data);
            }
        });
        </script>
        """, unsafe_allow_html=True)
    
        components.html(f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body {{ 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
    padding: 20px;
    background: linear-gradient(135deg, #4CAF50 0%, #2196F3 100%);
    color: white;
    border-radius: 10px;
  }}
  .container {{
    max-width: 800px;
    margin: 0 auto;
    background: rgba(255,255,255,0.1);
    padding: 30px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
  }}
  .controls {{ 
    display: flex; 
    gap: 20px; 
    align-items: center; 
    justify-content: center;
    margin-bottom: 30px;
  }}
  .status-display {{
    text-align: center;
    margin: 20px 0;
    padding: 15px;
    border-radius: 10px;
    background: rgba(255,255,255,0.2);
  }}
  .status-idle {{ background: rgba(108, 117, 125, 0.3); }}
  .status-connecting {{ background: rgba(255, 193, 7, 0.3); }}
  .status-active {{ background: rgba(40, 167, 69, 0.3); }}
  .status-error {{ background: rgba(220, 53, 69, 0.3); }}
  
  button {{ 
    padding: 12px 24px; 
    font-size: 16px;
    border: none;
    border-radius: 25px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: bold;
  }}
  .start-btn {{ 
    background: linear-gradient(45deg, #4CAF50, #45a049);
    color: white;
  }}
  .start-btn:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(76,175,80,0.4); }}
  .start-btn:disabled {{ 
    background: #6c757d; 
    cursor: not-allowed; 
    transform: none;
    box-shadow: none;
  }}
  .stop-btn {{ 
    background: linear-gradient(45deg, #f44336, #da190b);
    color: white;
  }}
  .stop-btn:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(244,67,54,0.4); }}
  .stop-btn:disabled {{ 
    background: #6c757d; 
    cursor: not-allowed;
    transform: none; 
    box-shadow: none;
  }}
  
  .log {{ 
    height: 200px; 
    overflow-y: auto; 
    border: 1px solid rgba(255,255,255,0.3); 
    padding: 15px; 
    background: rgba(0,0,0,0.2);
    border-radius: 10px;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 13px;
    line-height: 1.4;
  }}
  .audio-controls {{
    text-align: center;
    margin: 20px 0;
  }}
  .pulse {{
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{
    0% {{ transform: scale(1); }}
    50% {{ transform: scale(1.05); }}
    100% {{ transform: scale(1); }}
  }}
  .visualizer {{
    width: 100%;
    height: 60px;
    background: rgba(0,0,0,0.2);
    border-radius: 10px;
    margin: 10px 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
  }}
</style>
</head>
<body>
  <div class="container">
    <div class="status-display status-idle" id="statusDisplay">
      <h3 id="statusTitle">🎤 Interview Ready</h3>
      <p id="statusText">Click "Start Interview" to begin your speech-to-speech interview</p>
    </div>
    
    <div class="controls">
      <button id="startBtn" class="start-btn">🎤 Start Interview</button>
      <button id="stopBtn" class="stop-btn" disabled>⏹️ End Interview</button>
    </div>
    
    <div class="audio-controls">
      <audio id="remoteAudio" autoplay style="width: 100%; max-width: 400px;"></audio>
    </div>
    
    <div class="visualizer" id="visualizer">
      🎙️ Ready for interview - Audio will show here when active
    </div>
    
    <div class="log" id="log"></div>
  </div>

<script>
(async () => {{
  const serverBase = {SERVER_URL!r};
  const interviewInstructions = {submission['interview_instructions']!r};
  const chosenVoice = {submission['chosen_voice']!r};
  const responseSpeed = {submission['response_speed']!r};
  
  const logEl = document.getElementById('log');
  const statusDisplay = document.getElementById('statusDisplay');
  const statusTitle = document.getElementById('statusTitle');
  const statusText = document.getElementById('statusText');
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const visualizer = document.getElementById('visualizer');
  
  let pc, dc, micStream;
  let isConnected = false;
  let questionStartTime = null;
  let interviewStarted = false;
  
  function updateStatus(status, title, text, className) {{
    statusDisplay.className = `status-display ${{className}}`;
    statusTitle.textContent = title;
    statusText.textContent = text;
  }}
  
  function log(msg, type = 'info') {{
    const timestamp = new Date().toLocaleTimeString();
    const icon = type === 'error' ? '❌' : type === 'success' ? '✅' : type === 'warning' ? '⚠️' : 'ℹ️';
    logEl.innerHTML += `<div>${{timestamp}} ${{icon}} ${{msg}}</div>`;
    logEl.scrollTop = logEl.scrollHeight;
  }}

  async function start() {{
    startBtn.disabled = true;
    stopBtn.disabled = false;
    updateStatus('connecting', '🔄 Starting Interview...', 'Connecting to AI interviewer', 'status-connecting');
    
    try {{
      log('Starting interview session...', 'info');
      
      // 1) Fetch ephemeral session token
      const sessResp = await fetch(serverBase + "/session", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ 
          voice: chosenVoice,
          instructions: interviewInstructions
        }})
      }});
      
      if (!sessResp.ok) {{
        throw new Error(`Server error: ${{sessResp.status}} ${{sessResp.statusText}}`);
      }}
      
      const sess = await sessResp.json();
      if (sess.error) throw new Error(sess.error);
      
      const EPHEMERAL_KEY = sess.client_secret;
      if (!EPHEMERAL_KEY) throw new Error("No ephemeral token from server");
      
      log('✅ Interview session created', 'success');
      
      // 2) Setup WebRTC
      pc = new RTCPeerConnection();
      const remoteAudio = document.getElementById('remoteAudio');
      pc.ontrack = (event) => {{
        log('🔊 AI interviewer connected', 'success');
        const stream = event.streams[0];
        if (stream && stream.getAudioTracks().length > 0) {{
          remoteAudio.srcObject = stream;
          visualizer.textContent = '🔊 AI interviewer ready - speak naturally';
          log('🎵 Audio connection established', 'success');
        }} else {{
          log('⚠️ No audio stream received', 'warning');
          visualizer.textContent = '⚠️ No audio connection';
        }}
      }};

      // 3) Create data channel
      dc = pc.createDataChannel("oai-data");
      dc.onopen = () => {{
        log('🔗 Communication channel ready', 'success');
      }};
      dc.onerror = (error) => {{
        log('❌ Communication error: ' + error, 'error');
      }};
      dc.onmessage = (e) => handleDataMessage(e);

      // 4) Get microphone
      log('🎤 Setting up microphone...', 'info');
      micStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
      log('✅ Microphone ready', 'success');
      visualizer.textContent = '🎤 Microphone active - ready for interview';
      
      for (const track of micStream.getTracks()) {{
        pc.addTrack(track, micStream);
      }}

      // 5) Setup audio receiving
      pc.addTransceiver("audio", {{ direction: "recvonly" }});
      log('🔊 Audio system configured', 'success');

      // 6) Create and set local description
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      log('📡 Connection offer created', 'success');

      // 7) Exchange SDP with OpenAI Realtime
      const baseUrl = "https://api.openai.com/v1/realtime";
      const model = sess.model || "gpt-4o-realtime-preview-2025-06-03";
      const sdpResp = await fetch(`${{baseUrl}}?model=${{encodeURIComponent(model)}}`, {{
        method: "POST",
        body: offer.sdp,
        headers: {{
          Authorization: `Bearer ${{EPHEMERAL_KEY}}`,
          "Content-Type": "application/sdp"
        }}
      }});
      
      if (!sdpResp.ok) throw new Error(`Connection failed: ${{sdpResp.status}}`);
      
      const answer = {{ type: "answer", sdp: await sdpResp.text() }};
      await pc.setRemoteDescription(answer);
      
      // 8) Configure the interview session
      setTimeout(() => {{
        if (dc.readyState === 'open') {{
          const sessionConfig = {{
            type: "session.update",
            session: {{
              turn_detection: {{
                type: "server_vad",
                threshold: 0.5,
                prefix_padding_ms: 300,
                silence_duration_ms: responseSpeed * 1000
              }},
              input_audio_transcription: {{
                model: "whisper-1"
              }},
              voice: chosenVoice,
              temperature: 0.7,
              max_response_output_tokens: 1000,
              modalities: ["audio", "text"],
              response_format: "audio",
              instructions: interviewInstructions
            }}
          }};
          dc.send(JSON.stringify(sessionConfig));
          log('⚙️ Interview configuration applied', 'success');
          
          // Start the interview with a greeting
          const greeting = {{
            type: "conversation.item.create",
            item: {{
              type: "message",
              role: "user",
              content: [{{
                type: "input_text",
                text: "Hello! I'm ready to begin the interview."
              }}]
            }}
          }};
          dc.send(JSON.stringify(greeting));
          
          const startRequest = {{
            type: "response.create",
            response: {{
              modalities: ["audio"],
              instructions: "Start the interview with a warm greeting and then ask the first question."
            }}
          }};
          dc.send(JSON.stringify(startRequest));
          
          interviewStarted = true;
          
        }}
      }}, 500);
      
      isConnected = true;
      updateStatus('active', '🎤 Interview In Progress', 'Speak naturally with the AI interviewer', 'status-active');
      startBtn.classList.add('pulse');
      
      // Update Streamlit session state
      window.parent.postMessage({{type: 'interview_started'}}, '*');
      
    }} catch (error) {{
      log(`❌ Interview setup failed: ${{error.message}}`, 'error');
      updateStatus('error', '❌ Connection Failed', error.message, 'status-error');
      startBtn.disabled = false;
      stopBtn.disabled = true;
      cleanup();
    }}
  }}

  function cleanup() {{
    try {{
      if (dc && dc.readyState === 'open') dc.close();
      if (pc) pc.close();
      if (micStream) micStream.getTracks().forEach(t => t.stop());
    }} catch (e) {{ /* ignore cleanup errors */ }}
    startBtn.classList.remove('pulse');
    visualizer.textContent = '🔇 Interview ended';
  }}

  async function stop() {{
    startBtn.disabled = false;
    stopBtn.disabled = true;
    isConnected = false;
    interviewStarted = false;
    updateStatus('idle', '⚪ Interview Completed', 'Interview session has ended. You can start a new interview if needed.', 'status-idle');
    log('🏁 Interview completed', 'info');
    cleanup();
    
    // Update Streamlit session state
    window.parent.postMessage({{type: 'interview_ended'}}, '*');
  }}

  // Handle interview conversation events
  async function handleDataMessage(e) {{
    if (!isConnected) return;
    
    try {{
      const msg = JSON.parse(e.data);
      
      if (msg.type === "input_audio_buffer.speech_started") {{
        questionStartTime = Date.now();
        visualizer.textContent = '🎤 You are speaking...';
        log('🎤 Voice detected', 'info');
      }}
      
      if (msg.type === "input_audio_buffer.speech_stopped") {{
        visualizer.textContent = '🤔 AI interviewer is thinking...';
        log('⏸️ Processing your response...', 'info');
      }}
      
      if (msg.type === "response.audio.delta") {{
        visualizer.textContent = '🔊 AI interviewer speaking...';
      }}
      
      // Capture conversation transcript
      if (msg.type === "conversation.item.created") {{
        const item = msg.item;
        if (item.content && item.content.length > 0) {{
          const content = item.content[0];
          if (content.type === "input_text" || content.type === "text") {{
            const transcriptEntry = {{
              role: item.role,
              content: content.text || content.transcript,
              timestamp: new Date().toISOString()
            }};
            // Send transcript to Streamlit
            window.parent.postMessage({{
              type: 'transcript_update',
              data: transcriptEntry
            }}, '*');
          }}
        }}
      }}
      
      // Capture token usage from response.done events
      if (msg.type === "response.done" && msg.response && msg.response.usage) {{
        const usage = msg.response.usage;
        const tokenData = {{
          total_tokens: usage.total_tokens || 0,
          input_tokens: usage.input_tokens || 0,
          output_tokens: usage.output_tokens || 0,
          audio_input_tokens: usage.input_token_details?.audio_tokens || 0,
          text_input_tokens: usage.input_token_details?.text_tokens || 0,
          audio_output_tokens: usage.output_token_details?.audio_tokens || 0,
          text_output_tokens: usage.output_token_details?.text_tokens || 0
        }};
        
        // Send token usage to Streamlit
        window.parent.postMessage({{
          type: 'token_usage_update',
          data: tokenData
        }}, '*');
        
        if (questionStartTime) {{
          const totalTime = ((Date.now() - questionStartTime) / 1000).toFixed(1);
          visualizer.textContent = '🎤 Your turn - respond naturally';
          log(`✅ Exchange completed in ${{totalTime}}s (Tokens: ${{usage.total_tokens}})`, 'success');
          questionStartTime = null;
        }} else {{
          visualizer.textContent = '🎤 Your turn - speak your response';
          log(`✅ AI response complete (Tokens: ${{usage.total_tokens}})`, 'success');
        }}
      }}
      
      if (msg.type === "input_audio_buffer.speech_started" && !interviewStarted) {{
        log('👋 Interview starting...', 'success');
      }}
      
    }} catch (err) {{
      // Ignore non-JSON messages
    }}
  }}

  startBtn.onclick = start;
  stopBtn.onclick = stop;
  
  // Initialize
  log('🚀 Speech-to-Speech Interview ready', 'success');
}})();
</script>
</body>
</html>
""", height=600, scrolling=True)
    
    # Interview controls and information
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Restart Interview", help="Start a fresh interview session"):
            st.session_state.interview_active = False
            st.rerun()
    
    with col2:
        if st.button("🏠 Back to Setup", help="Return to interview setup form"):
            st.session_state.submitted_speech = False
            if "interview_submission" in st.session_state:
                del st.session_state.interview_submission
            st.rerun()
    

