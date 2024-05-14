

# Import required libraries:
# --------------------------
import os # Standard Library
import tempfile

# Our Own Modules
from lib import chatpdf, chatgeneration

# Third-Party Libraries
from dotenv import load_dotenv
import streamlit as st

import fitz # PyMuPDF
from pdf4llm import to_markdown

# ------------------------------------------------------------------------------


# Models:
# -------
models = [
  'gpt-4o', 
  'claude-3-opus-20240229', 
  'command-r-plus',
  'llama3-70b-8192' 
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
TEMPERATURE = 0.25



# -----------------------------------------------------------------------------

# Streamlit Application:
# ----------------------
st.set_page_config(page_title = "ChatISA Interview Mentor", layout = "centered",page_icon='🤖')
st.markdown("## 🤖 ChatISA: Interview Mentor 🤖")


# First "Screen" of the Interview Mentor:
# ---------------------------------------

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

if not st.session_state.submitted:
    st.markdown(
        "Welcome to ChatISA Interview Mentor! This tool is designed to help you prepare for technical interviews by generating interview questions based on your resume and the job description. "
        "To get started, please fill out the form below with your information and the job details. "
        "Once you have submitted the form, the tool will generate a structured interview with six questions that you can use to practice for your interview. "
        "Please note that the questions are generated by a language model and may not be perfect. "
        "Feel free to modify the questions as needed to better fit your interview preparation. "
        "Good luck with your interview preparation!"
        )
    
    st.sidebar.markdown("### Choose Your LLM")
    model_choice = st.sidebar.selectbox(
      "Choose your LLM",
      models,
      index= 0,
      key='model_choice',
      label_visibility='collapsed',
      help="Choose the LLM you want to use for generating the interview questions."
      )

    # Rest of the Sidebar Markdown:
    # ----------------------------
    st.sidebar.markdown("""
    ### Mantained By 
      - [Fadel Megahed](https://miamioh.edu/fsb/directory/?up=/directory/megahefm)   
      - [Joshua Ferris](https://miamioh.edu/fsb/directory/?up=/directory/ferrisj2)

    ### Version 
      1.3.0 (May 14, 2024)

    ### Key Features
      - Free to use
      - Chat with one of nine LLMs
      - Export chat to PDF
      
    ### Support & Funding
      - Farmer School of Business
      - US Bank
    """)

    # A Button to show/hide Disclaimers and References:
    if 'show_info' not in st.session_state:
        st.session_state.show_info = False

    # Toggle button
    if st.sidebar.button('Toggle Disclaimers & References'):
        st.session_state.show_info = not st.session_state.show_info  # Toggle the state

    # Conditionally display the Markdown
    if st.session_state.show_info:
        st.sidebar.markdown("""
        ### Disclaimers
        - ChatISA is designed for educational purposes only.
        - If you plan to use ChatISA for classwork, you must obtain approval from your instructor beforehand.
        - Neither Miami University nor the Farmer School of Business can be held responsible for any content generated by this app.
        - Always use ChatISA at your own risk and evaluate the accuracy of the generated answers.

        ### References
        - **Prompt Engineering:** Adapted from [Assigning AI by Mollick and Mollic 2023](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4475995)
        - **Streamlit App:** Adapted from [ChatGPT Apps with Streamlit](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-chatgpt-like-app)
        - **Our Code Repo:** [![Click here to access](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/fmegahed/chatisa)
        """)
        
      
    col1, col2 = st.columns(2, gap = 'large')

    with col1:
      st.markdown("### Interviewee Information")
      
      grade = st.selectbox(
        "What is your Current Grade Level?", 
        ["Freshman", "Sophomore", "Junior", "Senior", "Graduate Student"],
        index = 3,
        key = 'grade',
        help = "What is your current grade level?"
        )
        
      major = st.selectbox(
        "What is your Major?",
        [ "Business Analytics", "Computer Science", "Cybersecurity Management",
        "Data Science", "Information Systems", "Statistics", "Software Engineering"],
        index = 0,
        key = 'major',
        help = "What is your current major?"
        )
        
      raw_resume = st.file_uploader(
        "Upload your Resume",
        type = ['pdf'],
        key = 'resume',
        help = "Upload your resume in PDF format. This will be used to generate interview questions based on the information in your resume."
        )
      
    with col2:
      st.markdown('### Job Information')
      
      job_title = st.text_input(
        "Input the Job Title",
        value = "",
        key = 'job_title',
        help = "What is the job title you are interviewing for?",
        placeholder = "Business Analyst"
      )
      
      job_description = st.text_area(
        "Paste the Job Description",
        value = "",
        key = 'job_description',
        help = "Paste the job description for the position you are interviewing for. This should be copied and pasted from the job advertisement. It is expected that your text will cover the job duties and responsibilities, required qualifications, preferred qualifications, and any other relevant information.",
        placeholder = "As a Business Analyst, you will be responsible for analyzing data to help the company make informed business decisions. You will work closely with stakeholders to understand their needs and provide data-driven insights to support their decision-making process. The ideal candidate will have a strong background in data analysis, business intelligence, and data visualization. They will also have excellent communication skills and the ability to work collaboratively with cross-functional teams.",
        height = 300
      )
      
    if st.button('Submit'):
      if all([model_choice, grade, major, raw_resume, job_title, job_description]):
          # Process the resume
          with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
              tmp.write(raw_resume.getvalue())
              tmp_path = tmp.name

          resume_text = to_markdown(tmp_path)
          os.unlink(tmp_path)

          # Store information in session_state
          st.session_state['submission'] = {
              'model_choice': model_choice,
              'grade': grade,
              'major': major,
              'resume_text': resume_text,
              'job_title': job_title,
              'job_description': job_description
          }
          
          # Clear the form or redirect/show other content
          st.success('Your submission has been recorded.')
          st.session_state.submitted = True
          st.rerun()
          
      else:
          st.error('Please fill in all fields before submitting.')


# Next Screen:
# ------------
if st.session_state.submitted:
    # Sidebar Markdown:
    # -----------------
    st.sidebar.markdown("""
      ### Mantained By 
        - [Fadel Megahed](https://miamioh.edu/fsb/directory/?up=/directory/megahefm)   
        - [Joshua Ferris](https://miamioh.edu/fsb/directory/?up=/directory/ferrisj2)

      ### Version 
        1.3.0 (May 13, 2024)

      ### Key Features
        - Free to use
        - Chat with one of nine LLMs
        - Export chat to PDF
        
      ### Support & Funding
        - Farmer School of Business
        - US Bank
    """)

    # A Button to show/hide Disclaimers and References:
    if 'show_info' not in st.session_state:
        st.session_state.show_info = False

    # Toggle button
    if st.sidebar.button('Toggle Disclaimers & References'):
        st.session_state.show_info = not st.session_state.show_info  # Toggle the state

    # Conditionally display the Markdown
    if st.session_state.show_info:
        st.sidebar.markdown("""
        ### Disclaimers
        - ChatISA is designed for educational purposes only.
        - If you plan to use ChatISA for classwork, you must obtain approval from your instructor beforehand.
        - Neither Miami University nor the Farmer School of Business can be held responsible for any content generated by this app.
        - Always use ChatISA at your own risk and evaluate the accuracy of the generated answers.

        ### References
        - **Prompt Engineering:** Adapted from [Assigning AI by Mollick and Mollic 2023](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4475995)
        - **Streamlit App:** Adapted from [ChatGPT Apps with Streamlit](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-chatgpt-like-app)
        - **Our Code Repo:** [![Click here to access](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/fmegahed/chatisa)
        """)

    # add token counts to session state
    if 'token_counts' not in st.session_state:
      st.session_state.token_counts = {model: {'input_tokens': 0, 'output_tokens': 0} for model in models}

    # Retrieve the information from the session_state
    model_choice = st.session_state.submission['model_choice']
    grade = st.session_state.submission['grade']
    major = st.session_state.submission['major']
    resume_text = st.session_state.submission['resume_text']
    job_title = st.session_state.submission['job_title']
    job_description = st.session_state.submission['job_description']
    
    # The system prompt
    SYSTEM_PROMPT = (
      f"You are an expert technical interviewer. You are interviewing a {{grade}} student, "
      f"who is majoring in {{major}} for a {{job_title}} position.\n\n"
      f"The student has provided you with their resume:\n{{resume_text}}\n\n"
      f"The job description for the {{job_title}} position is as follows:\n"
      f"{{job_description}}\n\n"
      "Carefully read and analyze the student's resume to understand their background "
      "and qualifications, and how it relates to the job description. Extract relevant "
      "information from the job description pertaining to the job duties and "
      "responsibilities, required qualifications, preferred qualifications, and any "
      "other relevant information.\n\n"
      "Once you have analyzed the resume and job description, conduct a structured "
      "interview with the student to assess their qualifications for the position. The "
      "interview should consist of six questions, asked one at a time:\n\n"
      "1. Ask a background question about the student's interest in the position. "
      "Assess whether the candidate has a good understanding of the role, and has the "
      "necessary skills/drive to learn on the job.\n\n"
      "2. Ask a background question about how the student would measure business "
      "performance at the company and what information/metrics they would use. Look for "
      "answers that show the candidate did their research and has a good sense of the "
      "company's goals. Also, look for signs that the student can adopt a business "
      "mindset and is familiar with the industry's practices and norms.\n\n"
      "3. Ask a technical question that assesses the student's skills as they relate to "
      "the job requirements and/or required qualifications.\n\n"
      "4. Ask another technical question that assesses the student's software knowledge "
      "as it relates to the job requirements and/or required qualifications.\n\n"
      "5. Ask a situational question that assesses the student's ability to work in a "
      "team and/or handle difficult situations. Make the question tailored to what "
      "would be expected in this job. Possible questions can include, but are not "
      "limited to:\n"
      "   - Tell me about a time when you think you demonstrated good data sense.\n"
      "   - Describe your most complex data project from start to finish. What were the "
      "most difficult challenges, and how did you handle them?\n"
      "   - Tell me about a time when you had to work with a difficult team member. How "
      "did you handle the situation?\n\n"
      "6. Ask a behavioral question to screen for their soft skills. Example questions "
      "to ask include, but are not limited to:\n"
      "   - What do you think are the three best qualities that great data analysts "
      "share?\n"
      "   - How would you explain your findings and processes to an audience who might "
      "not know what a data analyst does?\n"
      "   - How do you stay current with the latest data analysis trends and tools?\n"
      "   - What do you do when you encounter a problem you don't know how to solve?\n\n"
      "Let the interviewee answer each question before moving on to the next question. "
      "Only ask follow-up questions if necessary. Any follow-up questions are not "
      "counted towards the six questions.\n\n"
      "At the end of the interview, thank the student for their time and then provide "
      "constructive feedback on their performance. The feedback should be based on the "
      "student's responses to the questions, as well as their overall demeanor and "
      "professionalism during the interview. The feedback should be specific, actionable,"
      " and focused on areas where the student can improve.\n\n"
      "Provide the feedback in the following format:\n\n"
      "Start by summarizing the interview, highlighting each question and answer.\n\n"
      "Provide positive feedback on the student's performance, highlighting areas where "
      "they excelled. Provide constructive criticism on the student's performance, "
      "highlighting areas where they could improve. Be specific and provide actionable "
      "advice on how the student can improve in these areas. It is important to be "
      "honest and direct in your feedback, but also supportive and encouraging. Provide "
      "examples of how the student could improve in these areas.\n\n"
      "Provide an overall score for the student's performance in the interview out of "
      "100. The score should be based on the student's responses to the questions, as "
      "well as their overall demeanor and professionalism during the interview.\n\n"
      "Remember to be professional, courteous, and respectful throughout the interview "
      "process. Do not repeat questions, ask leading questions, or provide hints to the "
      "student."
  )
    
    # Generate the interview questions
    if "messages" not in st.session_state:
      st.session_state.messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT
      },{
        "role": "user", 
        "content": "Hello, I am excited about this opportunity."
        }]
      
    for message in st.session_state.messages[2:]:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

    # Display chatbox input & process user input
    if prompt := st.chat_input("To start, type Hi. Then, answer the questions throughoutfully."):
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
          model = st.session_state.submission['model_choice'],
          messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
          temp = TEMPERATURE,
          max_num_tokens = 3000
        )
        
        # extracting the response, input tokens, and output tokens
        response, input_tokens, output_tokens = outputs
        full_response += response
        message_placeholder.markdown(full_response + "▌")
        
        # Update the token counts for the specific model in session state
        st.session_state.token_counts[st.session_state.submission['model_choice']]['input_tokens'] += input_tokens
        st.session_state.token_counts[st.session_state.submission['model_choice']]['output_tokens'] += output_tokens
      
      # Store the full response from the LLM in memory
      st.session_state.messages.append({"role": "assistant", "content": full_response})



  
    # Generating the PDF from the Chat:
    # ---------------------------------
    with st.expander("Export Chat to PDF"):
      row = st.columns([2, 2])
      user_name = row[0].text_input("Enter your name:")
      user_name = user_name.replace(" ", "_")
      user_course = row[1].text_input("Enter company name:")
      user_course = user_course.replace(" ", "_")
      if user_name != "" and user_course != "":
        pdf_output_path = chatpdf.create_pdf(chat_messages=st.session_state.messages, models = models, token_counts = st.session_state.token_counts, user_name=user_name, user_course=user_course)

        with open(pdf_output_path, "rb") as file:
          st.download_button(label="Download PDF", data=file, file_name=f"{user_course}_{user_name}_chatisa.pdf", mime="application/pdf", use_container_width=True)
