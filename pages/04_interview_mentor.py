import json
import streamlit as st
import streamlit.components.v1 as components

from config import (
    APP_NAME,
    DATE,
    OPENAI_REALTIME_MODEL,
    DEFAULT_REALTIME_VOICE,
    REALTIME_VOICES,
    DEFAULT_MODELS,
    MAJORS,
    THEME_COLORS,
    PAGE_ICON,
    get_page_models,
)

from lib.speech import (
    ensure_interview_state_defaults,
    check_realtime_ready,
    mint_realtime_client_secret,
    pdf_to_markdown,
    transcribe_audio,
    generate_speech,
)
from lib import chatpdf, chatgeneration, sidebar
from lib.ui import apply_theme_css

THIS_PAGE = "interview_mentor"
st.session_state.cur_page = THIS_PAGE

st.set_page_config(page_title="ChatISA: Interview Mentor", page_icon=PAGE_ICON, layout="wide")
apply_theme_css()

# ---------- Minimal WebRTC frontend ----------
RTC_HTML = """
<style>
.btn { padding: 8px 12px; border: 0; border-radius: 6px; cursor:pointer; }
.btn-primary { background: #c3142d; color: #fff; }
.btn-ghost { background: #eee; }
.row { display:flex; gap:8px; align-items:center; }
.small { font-size: 12px; color:#666; }
</style>
<script>
async function startRealtime(clientSecret, model) {
  const pc = new RTCPeerConnection();
  const audioEl = document.getElementById("realtime-audio");
  pc.ontrack = (event) => { audioEl.srcObject = event.streams[0]; };

  const ms = await navigator.mediaDevices.getUserMedia({ audio: true });
  for (const track of ms.getTracks()) pc.addTrack(track, ms);

  const dc = pc.createDataChannel("oai-events");
  dc.onopen = () => console.log("oai-events open");
  dc.onmessage = (e) => console.log("OAI:", e.data);

  const offer = await pc.createOffer({ offerToReceiveAudio: true });
  await pc.setLocalDescription(offer);

  const res = await fetch("https://api.openai.com/v1/realtime?model=" + encodeURIComponent(model), {
    method: "POST",
    headers: {
      "Authorization": "Bearer " + clientSecret,
      "Content-Type": "application/sdp"
    },
    body: offer.sdp
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error("Realtime SDP exchange failed: " + res.status + " " + errText);
  }
  const answerSdp = await res.text();
  await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });

  window._oai_pc = pc;
  parent.postMessage({ type: "interview_started" }, "*");
}

async function stopRealtime() {
  const pc = window._oai_pc;
  if (pc) {
    try { pc.getSenders().forEach(s => s.track && s.track.stop()); } catch(e){}
    try { pc.getReceivers().forEach(r => r.track && r.track.stop()); } catch(e){}
    pc.close();
    window._oai_pc = null;
  }
  parent.postMessage({ type: "interview_stopped" }, "*");
}
</script>

<div class="row">
  <button id="startBtn" class="btn btn-primary">Start Interview</button>
  <button id="stopBtn" class="btn btn-ghost">Stop</button>
  <audio id="realtime-audio" autoplay></audio>
</div>
<div class="small">Grant mic permission when prompted. Keep this tab in the foreground.</div>

<script>
(() => {
  const startBtn = document.getElementById("startBtn");
  const stopBtn  = document.getElementById("stopBtn");
  startBtn.onclick = async () => {
    try {
      const payload = JSON.parse(decodeURIComponent(window.location.hash.slice(1)));
      await startRealtime(payload.client_secret, payload.model);
    } catch (err) {
      alert(err.message);
    }
  };
  stopBtn.onclick = async () => { await stopRealtime(); };
})();
</script>
"""

# ---------- Sidebar ----------
# Render navigation first
sidebar.render_navigation()

# Interview-specific info with Miami colors
st.sidebar.markdown(
    f'<h3 style="color: {THEME_COLORS["primary"]};">Interview Guide</h3>',
    unsafe_allow_html=True
)
st.sidebar.markdown(f"""
<div style="background-color: {THEME_COLORS['info']}10; padding: 10px; border-radius: 5px; font-size: 0.9em;">
    <strong>Modes</strong><br>
    &bull; Speech-to-speech (default)<br>
    &bull; Transcription (voice-text + optional AI audio)
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="background-color: {THEME_COLORS['success']}15; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 0.9em;">
    <strong>Steps</strong><br>
    1. Fill the setup form<br>
    2. Start the interview<br>
    3. (Optional) Export transcript to PDF
</div>
""", unsafe_allow_html=True)

# Render rest of sidebar (skip navigation since we already rendered it)
sidebar.render_sidebar(skip_navigation=True)

# ---------- Page logic ----------
ensure_interview_state_defaults()

def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start:end + 1]

def build_interview_profile(resume_text: str, job_description: str, job_title: str, grade: str, major: str, model: str) -> dict:
    prompt = (
        "Extract a concise interview profile as JSON with keys:\n"
        "- role_focus (string)\n"
        "- priority_topics (list of 6-10 strings)\n"
        "- strengths (list of 3-6 strings)\n"
        "- gaps (list of 3-6 strings)\n"
        "- starter_questions (list of 3 strings)\n"
        "Return ONLY valid JSON.\n\n"
        f"Role: {job_title}\n"
        f"Candidate: {grade}, {major}\n\n"
        f"Resume:\n{resume_text[:4000]}\n\n"
        f"Job description:\n{job_description[:4000]}"
    )
    try:
        resp, _, _ = chatgeneration.generate_chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a JSON-only extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            temp=0.2,
            max_num_tokens=1200
        )
        json_text = _extract_json(resp)
        return json.loads(json_text) if json_text else {}
    except Exception:
        return {}

# ============ FIRST SCREEN: Setup ============
if not st.session_state.submitted_speech:
    st.markdown(f'<h1 style="color: {THEME_COLORS["primary"]};">ChatISA: Interview Mentor</h1>', unsafe_allow_html=True)

    # Mode toggle
    col_toggle1, _ = st.columns([1, 3])
    with col_toggle1:
        use_transcription = st.toggle("Use Transcription Mode", value=False, key="transcription_mode")

    if use_transcription:
        st.markdown(
            "Speak your responses; they're transcribed to text for the AI interviewer. "
            "Optionally enable AI voice responses. Upload your resume and a job description for tailored questions."
        )
    else:
        st.markdown(
            "Natural two-way audio using OpenAI Realtime. The AI tailors questions to your resume and job description."
        )

    # Realtime readiness
    if not use_transcription:
        if not check_realtime_ready():
            st.error("Realtime not ready (missing OPENAI_API_KEY).")
            st.stop()
        st.success("Realtime ready.")
    else:
        st.success("Transcription mode.")

    # Model selection (transcription)
    if use_transcription:
        # Get transcription models dynamically from config
        transcription_models = get_page_models("interview_mentor_transcription")
        default_model = DEFAULT_MODELS.get('interview_mentor_transcription', 'gpt-5.2-2025-12-11')
        default_index = transcription_models.index(default_model) if default_model in transcription_models else 0
        st.markdown("### AI Model Selection")
        selected_model = st.selectbox(
            "Choose AI model",
            transcription_models,
            index=default_index,
            key='transcription_model_choice',
        )

    # Interview type
    st.markdown("### Interview Type")
    st.caption("Interview type is tailored to the major and job (e.g., technical in marketing differs from business analytics).")
    interview_type = st.selectbox(
        "Select interview format",
        ["Behavioral", "Technical", "Coding/LeetCode", "Case", "Mixed"],
        index=4,
        key="interview_type",
        help=(
            "Behavioral: stories and decision-making. "
            "Technical: role-specific skills and domain knowledge. "
            "Coding/LeetCode: algorithms, data structures, and complexity. "
            "Case: structured problem solving and business judgment. "
            "Mixed: blend of the above."
        )
    )

    # Candidate + job info
    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.markdown("### Interviewee Information")
        grade = st.selectbox(
            "Current Grade Level",
            ["Freshman", "Sophomore", "Junior", "Senior", "Graduate Student"],
            index=3,
            key='grade_speech'
        )
        major_options = MAJORS + ["Other (enter below)"]
        major_choice = st.selectbox(
            "Major",
            major_options,
            index=0,
            key='major_speech'
        )
        custom_major = ""
        if major_choice == "Other (enter below)":
            custom_major = st.text_input("Enter your major", key="major_speech_custom")
        major = custom_major.strip() if major_choice == "Other (enter below)" else major_choice
        raw_resume = st.file_uploader(
            "Upload Resume (PDF)",
            type=['pdf'],
            key='resume_speech'
        )
    with col2:
        st.markdown("### Job Information")
        job_title = st.text_input("Job Title", value="", key='job_title_speech', placeholder="Business Analyst")
        job_description = st.text_area(
            "Job Description",
            value="",
            key='job_description_speech',
            placeholder="Paste the description here...",
            height=300
        )

    # Voice settings
    if not use_transcription:
        st.markdown("### Voice Settings")
        c3, c4 = st.columns(2)
        with c3:
            v_idx = REALTIME_VOICES.index(DEFAULT_REALTIME_VOICE)
            chosen_voice = st.selectbox("AI Interviewer Voice", REALTIME_VOICES, index=v_idx, key='chosen_voice_speech')
        with c4:
            response_speed = st.slider("Response Speed (seconds)", min_value=1, max_value=5, value=2, key='response_speed_speech')
    else:
        st.markdown("### Audio Settings")
        ai_audio_enabled = st.checkbox("Enable AI voice responses", value=False, key='ai_audio_transcription')
        if ai_audio_enabled:
            v_idx = REALTIME_VOICES.index(DEFAULT_REALTIME_VOICE)
            chosen_voice = st.selectbox("AI Voice", REALTIME_VOICES, index=v_idx, key='chosen_voice_transcription')
        else:
            chosen_voice = DEFAULT_REALTIME_VOICE

    # Submit
    button_text = 'Start Transcription Interview' if use_transcription else 'Start Speech-to-Speech Interview'
    if st.button(button_text, type="primary", use_container_width=True):
        if all([grade, major, raw_resume, job_title, job_description]):
            resume_text = pdf_to_markdown(raw_resume.getvalue())
            profile_model = st.session_state.get('transcription_model_choice', 'gpt-5.2-2025-12-11')
            interview_profile = build_interview_profile(
                resume_text=resume_text,
                job_description=job_description,
                job_title=job_title,
                grade=grade,
                major=major,
                model=profile_model,
            )

            interview_instructions = (
                f"You are an expert technical interviewer for a {job_title} role. "
                f"The candidate is a {grade} majoring in {major}.\n\n"
                f"Interview type: {interview_type}\n\n"
                f"Resume:\n{resume_text[:4000]}\n\n"
                f"Job description:\n{job_description[:4000]}\n\n"
                f"Interview profile (JSON):\n{json.dumps(interview_profile, ensure_ascii=True)}\n\n"
                "VOICE BEHAVIOR:\n"
                "- Speak naturally and professionally\n"
                "- Use appropriate pauses and emphasis\n"
                "- Keep responses concise but thorough\n\n"
                "INTERVIEW STRUCTURE:\n"
                "- Follow the selected interview type (behavioral, technical, coding/leetcode, case, or mixed)\n"
                "- Aim for 6-8 questions total; add follow-ups when answers are shallow\n"
                "- Choose the next topic based on the profile, what is not yet covered, and the candidate's answers\n"
                "- Ask ONE question at a time and wait for the response\n"
                "- Use a simple rubric per answer: 0=incorrect/irrelevant, 1=partial, 2=solid, 3=excellent\n"
                "- Provide feedback only after all questions are answered\n"
                "- End with: 3 strengths, 3 gaps/risks, 3 next-step actions, and an overall score out of 100\n"
                "- Compute the score as (sum of points / (3 * number of questions)) * 100, rounded"
            )

            submission_data = {
                'grade': grade,
                'major': major,
                'interview_type': interview_type,
                'resume_text': resume_text,
                'job_title': job_title,
                'job_description': job_description,
                'chosen_voice': chosen_voice,
                'interview_profile': interview_profile,
                'interview_mode': 'transcription' if use_transcription else 'speech_to_speech'
            }
            if use_transcription:
                submission_data.update({
                    'model_choice': st.session_state.get('transcription_model_choice', 'gpt-5.2-2025-12-11'),
                    'ai_audio_enabled': st.session_state.get('ai_audio_transcription', False),
                })
            else:
                submission_data.update({
                    'response_speed': response_speed,
                    'interview_instructions': interview_instructions
                })

            st.session_state['interview_submission'] = submission_data
            st.session_state.submitted_speech = True
            st.session_state.interview_active = False
            st.rerun()
        else:
            st.error('Please complete all fields and upload your resume (PDF).')

# ============ SECOND SCREEN: Live ============
if st.session_state.submitted_speech:
    submission = st.session_state.interview_submission or {}

    if submission.get('interview_mode') == 'transcription':
        st.markdown(f'<h1 style="color: {THEME_COLORS["primary"]};">Live Transcription Interview</h1>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h1 style="color: {THEME_COLORS["primary"]};">Live Speech-to-Speech Interview</h1>', unsafe_allow_html=True)

    with st.expander(" Interview Details", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Position:** {submission.get('job_title','')}")
            st.write(f"**Grade:** {submission.get('grade','')}")
            st.write(f"**Major:** {submission.get('major','')}")
            st.write(f"**Type:** {submission.get('interview_type','Mixed')}")
        with c2:
            if submission.get('interview_mode') == 'transcription':
                st.write(f"**Model:** {submission.get('model_choice', 'gpt-5.2-2025-12-11')}")
                st.write(f"**AI Audio:** {'Enabled' if submission.get('ai_audio_enabled', False) else 'Disabled'}")
                if submission.get('ai_audio_enabled', False):
                    st.write(f"**Voice:** {submission.get('chosen_voice', DEFAULT_REALTIME_VOICE)}")
                st.write("**Mode:** Voice-to-Text")
            else:
                st.write(f"**Voice:** {submission.get('chosen_voice', DEFAULT_REALTIME_VOICE)}")
                st.write(f"**Response Speed:** {submission.get('response_speed', 2)} s")
                st.write("**Mode:** Speech-to-Speech")

    with st.expander(" Interview Tips", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            if submission.get('interview_mode') == 'transcription':
                st.markdown(
                    "- Speak clearly for accurate transcription\n"
                    "- Wait for the AI before continuing\n"
                    "- Answer concisely and stay on topic\n"
                    "- Skim the transcript for accuracy"
                )
            else:
                st.markdown(
                    "- Speak clearly at a normal pace\n"
                    "- Wait for the AI to finish before responding\n"
                    "- Answer concisely and stay on topic\n"
                    "- Ask for clarification if needed"
                )
        with c2:
            st.markdown(
                "- Keep this tab active\n"
                "- Use stable internet\n"
                "- Prefer a quiet environment"
            )

    # ----- Mode-specific -----
    if submission.get('interview_mode') == 'transcription':
        st.success("Transcription mode active.")

        if (st.session_state.get('cur_page') != THIS_PAGE) and ("messages" in st.session_state):
            del st.session_state["messages"]

        if "messages" not in st.session_state:
            interview_profile = submission.get('interview_profile', {})
            SYSTEM_PROMPT = (
                f"You are an expert interviewer for a {submission.get('job_title','')} role. "
                f"The candidate is a {submission.get('grade','')} majoring in {submission.get('major','')}.\n\n"
                f"Interview type: {submission.get('interview_type','Mixed')}\n\n"
                f"Resume:\n{submission.get('resume_text','')}\n\n"
                f"Job description:\n{submission.get('job_description','')}\n\n"
                f"Interview profile (JSON):\n{json.dumps(interview_profile, ensure_ascii=True)}\n\n"
                "Ask one question at a time. Follow the selected interview type (behavioral, technical, coding/leetcode, case, or mixed). "
                "Aim for 6-8 total questions and add follow-ups when needed. "
                "Choose topics dynamically based on the profile, what has not yet been covered, and the candidate's answers. "
                "Be concise and professional. Use a simple rubric per answer: 0=incorrect/irrelevant, 1=partial, 2=solid, 3=excellent. "
                "Provide feedback only after all questions are answered. "
                "At the end, include 3 strengths, 3 gaps/risks, 3 next-step actions, and an overall score out of 100. "
                "Compute the score as (sum of points / (3 * number of questions)) * 100, rounded."
            )
            st.session_state.messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Hello, I am excited about this opportunity."}
            ]

        with st.container():
            for m in st.session_state.messages[2:]:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        try:
            from streamlit_mic_recorder import mic_recorder
            st.markdown("#### Press to speak, then stop to submit your answer.")
            audio_data = mic_recorder(
                start_prompt="Press to Speak",
                stop_prompt="Press to Stop Recording",
                format="wav",
                key="mic_recorder_trans"
            )
        except Exception:
            audio_data = None
            st.warning("Microphone widget unavailable.")

        if audio_data:
            user_text = transcribe_audio(audio_data["bytes"])
            st.session_state.messages.append({"role": "user", "content": user_text})
            with st.chat_message("user"):
                st.markdown(user_text)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                resp, in_tok, out_tok = chatgeneration.generate_chat_completion(
                    model=submission.get('model_choice', 'gpt-5.2-2025-12-11'),
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    temp=0.25,
                    max_num_tokens=6000
                )
                placeholder.markdown(resp)

                if submission.get('ai_audio_enabled', False):
                    st.session_state.chosen_voice = submission.get('chosen_voice', DEFAULT_REALTIME_VOICE)
                    audio_bytes = generate_speech(resp)
                    st.audio(audio_bytes, format="audio/mp3")

            st.session_state.messages.append({"role": "assistant", "content": resp})

        st.markdown("---")
        with st.expander(" Export Interview to PDF", expanded=False):
            c1, c2 = st.columns(2)
            user_name = c1.text_input("Your Name", key="trans_name").replace(" ", "_")
            company_name = c2.text_input("Company", key="trans_company").replace(" ", "_")
            if user_name and company_name:
                path = chatpdf.create_pdf(
                    chat_messages=st.session_state.messages,
                    models=[],
                    token_counts={},
                    user_name=user_name,
                    user_course=company_name
                )
                with open(path, "rb") as f:
                    st.download_button(
                        "Download Interview PDF",
                        f,
                        file_name=f"{company_name}_{user_name}_interview.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

    else:
        # Speech-to-speech
        # Auto-mint the token when entering the screen if not present
        if not st.session_state.get("_rt_token"):
            try:
                tok = mint_realtime_client_secret(
                    model=OPENAI_REALTIME_MODEL,
                    voice=submission.get('chosen_voice', DEFAULT_REALTIME_VOICE),
                    instructions=submission.get('interview_instructions')
                )
                st.session_state["_rt_token"] = tok
                st.success(f"Ephemeral token minted (expires at {tok['expires_at']}).")
            except Exception as e:
                st.error(str(e))

        tok = st.session_state.get("_rt_token")
        if tok:
            st.info(
                f"Model **{tok['model']}**, Voice **{tok['voice']}** -- session `{tok['session_id']}`.")
            payload = json.dumps({"client_secret": tok["client_secret"], "model": tok["model"]})
            components.html(
                RTC_HTML + f"<script>location.hash = encodeURIComponent('{payload}');</script>",
                height=170
            )
            st.session_state.interview_active = True
            if st.button("Refresh Token"):
                try:
                    tok = mint_realtime_client_secret(
                        model=OPENAI_REALTIME_MODEL,
                        voice=submission.get('chosen_voice', DEFAULT_REALTIME_VOICE),
                        instructions=submission.get('interview_instructions')
                    )
                    st.session_state["_rt_token"] = tok
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        else:
            st.warning("Token not available; click **Refresh Token** to try again.")

    # Controls
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Restart Interview"):
            st.session_state.interview_active = False
            st.session_state["_rt_token"] = None
            st.rerun()
    with c2:
        if st.button("Back to Setup"):
            st.session_state.submitted_speech = False
            st.session_state.interview_active = False
            st.session_state["_rt_token"] = None
            if "interview_submission" in st.session_state:
                del st.session_state["interview_submission"]
            st.rerun()
