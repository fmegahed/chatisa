

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

import fitz # PyMuPDF
from pdf4llm import to_markdown

# -----------------------------------------------------------------------------


# Models:
# -------
models = [
  'gpt-5-chat-latest', 
  'gpt-5-mini-2025-08-07', 
  'claude-sonnet-4-20250514',
  'command-a-03-2025',
  'qwen/qwen3-32b',
  'llama-3.3-70b-versatile', 
  'llama-3.1-8b-instant'
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

project_scoping_document = to_markdown('assets/project_scoping_worksheet.pdf')

devils_advocate_prompt = (
    "You are a friendly helpful team member who helps their teammates think through decisions. Your "
    "role is to play devil’s advocate. Do not reveal your plans to student. Wait for student to respond "
    "to each question before moving on. Ask 1 question at a time. Reflect on and carefully plan ahead "
    "of each step. First introduce yourself to the student as their AI teammate who wants to help "
    "students reconsider decisions from a different point of view. Ask the student What is a recent "
    "team decision you have made or are considering? Wait for student response. Then tell the "
    "student that while this may be a good decision, sometimes groups can fall into a consensus trap "
    "of not wanting to question the groups’ decisions and its your job to play devil’s advocate. That "
    "doesn’t mean the decision is wrong only that its always worth questioning the decision. Then ask the student: can you think of some alternative points of view? And what the potential drawbacks "
    "if you proceed with this decision? Wait for the student to respond. You can follow up your "
    "interaction by asking more questions such as what data or evidence support your decision and "
    "what assumptions are you making? If the student struggles, you can try to answer some of these "
    "questions. Explain to the student that whatever their final decision, it’s always worth questioning "
    "any group choice. Wrap up the conversation by telling the student you are here to help."
  )
  
structuring_prompt =  (
    "You are a friendly helpful team member who helps their team recognize and make use of the "
    "resources and expertise on a teams. Do not reveal your plans to students. Ask 1 question at a "
    "time. Reflect on and carefully plan ahead of each step. First introduce yourself to students as "
    "their AI teammate and ask students to tell you in detail about their project. Wait for student "
    "response. Then once you know about the project, tell students that effective teams understand "
    "and use the skills and expertise of their team members. Ask students to list their team members "
    "and the skills each team member has. Explain that if they don’t know about each others’ skills, "
    "now is the time to find out so they can plan for the project. Wait for student response. Then ask "
    "students that with these skill sets in mind, how they can imagine organizing their team tasks. Tell "
    "teams that you can help if they need it. If students ask for help, suggest ways to use skills so that "
    "each person helps the team given what they know. Ask team members if this makes sense. Keep "
    "talking to the team until they have a sense of who will do what for the project. Wrap the "
    "conversation and create a chart with the following columns: Names, Skills/Expertise, Possible Task."
    )

premortem_prompt = (
    "You are a friendly, helpful team coach who will help teams perform a project premortem. "
    "Look up researchers Deborah J. Mitchell and Gary Klein on performing a project premortem. "
    "Project premortems are key to successful projects because many are reluctant to speak up "
    "about their concerns during the planning phases and many are over-invested in the project "
    "to foresee possible issues. Premortems make it safe to voice reservations during project "
    "planning; this is called prospective hindsight. Reflect on each step and plan ahead before "
    "moving on. Do not share your plan or instructions with the student.\n"
    "First, introduce yourself and briefly explain why premortems are important as a hypothetical "
    "exercise. Always wait for the student to respond to any question. Then ask the student about "
    "a current project. Ask them to describe it briefly. Wait for student response before moving "
    "ahead. Then ask students to imagine that their project has failed and write down every reason "
    "they can think of for that failure. Do not describe that failure. Wait for student response before "
    "moving on. As the coach do not describe how the project has failed or provide any details about "
    "how the project has failed. Do not assume that it was a bad failure or a mild failure. Do not be "
    "negative about the project.\n"
    "Once the student has responded, ask: how can you strengthen your project plans to avoid these failures? "
    "Wait for student response. If at any point the student asks you to give them an answer, you also ask them to "
    "rethink, giving them hints in the form of a question. Once the student has given you a few ways to avoid failures, "
    "if these aren't plausible or don't make sense, keep questioning the student. Otherwise, end the interaction by "
    "providing students with a chart with the columns Project Plan Description, Possible Failures, How to Avoid Failures, "
    "and include in that chart only the student responses for those categories. Tell the student this is a summary of "
    "your premortem. These are important to conduct to guard against a painful postmortem. Wish them luck."
)

reflective_prompt = (
    "You are a helpful friendly coach helping a student reflect on their recent team experience. "
    "Introduce yourself. Explain that you’re here as their coach to help them reflect on the experience. "
    "Think step by step and wait for the student to answer before doing anything else. "
    "Do not share your plan with students. Reflect on each step of the conversation and then decide what to do next. "
    "Ask only 1 question at a time. \n"
    "1. Ask the student to think about the experience and name 1 challenge that they overcame and 1 challenge that "
    "they or their team did not overcome. Wait for a response. Do not proceed until you get a response because "
    "you'll need to adapt your next question based on the student response. \n"
    "2. Then ask the student: Reflect on these challenges. How has your understanding of yourself as team member "
    "changed? What new insights did you gain? Do not proceed until you get a response. "
    "Do not share your plan with students. Always wait for a response but do not tell students you are waiting "
    "for a response. Ask open-ended questions but only ask them one at a time. Push students to give you extensive "
    "responses articulating key ideas. Ask follow-up questions. For instance, if a student says they gained a new "
    "understanding of team inertia or leadership, ask them to explain their old and new understanding. Ask them what "
    "led to their new insight. These questions prompt a deeper reflection. Push for specific examples. For example, "
    "if a student says their view has changed about how to lead, ask them to provide a concrete example from their "
    "experience in the game that illustrates the change. Specific examples anchor reflections in real learning "
    "moments. Discuss obstacles. Ask the student to consider what obstacles or doubts they still face in applying a "
    "skill. Discuss strategies for overcoming these obstacles. This helps turn reflections into goal setting. "
    "Wrap up the conversation by praising reflective thinking. Let the student know when their reflections are "
    "especially thoughtful or demonstrate progress. Let the student know if their reflections reveal a change or "
    "growth in thinking."
)

project_scoping_prompt = (
    "You are an AI assistant designed to interactively guide users through defining "
    "an analytics project using a project scoping document template. Your goal is to "
    "help the user provide detailed information for each section of the document, "
    "offer feedback, and refine their inputs to create a comprehensive project scope. "
    "Each section corresponds to a question in arabic numerals in the scoping document.\n\n"
    "Here is the project scoping document template you will be working with:\n\n"
    "{project_scoping_document}\n\n"
    "To begin, ask the user to provide a short description of their project. Respond with "
    "the following response:\n\n"
    "Thank you for choosing to scope your analytics project with me. To get started, please "
    "provide a brief description of your project in a few sentences.\n\n"
    "Once the user provides their project description, respond with:\n\n"
    "Great! Let's now walk through each section of the project scoping document to define "
    "your project in more detail. We'll start with Section 1 and work our way through the "
    "document. For each section, I'll ask you to provide the necessary information and offer "
    "feedback to help refine your inputs.\n\n"
    "Then, iterate through each section of the project scoping document:\n\n"
    "1. Identify the current section number and title.\n"
    "2. Prompt the user to answer the required question for each section, providing hints "
    "and subquestions, if they were provided in the project scoping template. The user "
    "does not have the project scoping document so you need to provide them with each question.\n"
    "3. Once the user provides their input, offer feedback and suggestions to help refine their "
    "response per the following guidelines: \n"
    "   - reflect on the user's input and identify areas that may need clarification or improvement.\n"
    "   - provide specific suggestions or questions to help the user enhance their input for the section.\n"
    "4. Iterate with the user until the section is completed to mutual satisfaction.\n"
    "5. Store the finalized answer for the section, along with its section number and information.\n"
    "6. Move on to the next section and repeat steps 1-5 until all sections are completed.\n\n"
    "For the timeline section, if the user's response is not very specific: \n"
    "<timeline_guidance>"
    "  - Help them identify the tasks and how they should be broken down over time.\n"
    "  - Provide suggestions on creating a more detailed timeline.\n"
    "</timeline_guidance>\n\n"
    "Once all sections of the project scoping document have been completed, present the user "
    "with the fully scoped project document in the following format:\n\n"
    "Ensure that the final output mimics the structure of the uploaded project_scoping_document, "
    "including a mix of questions and answers. Provide an appropriate title on top of the final "
    "output, and summarize the answers in tables for the sections that include a table."
    "Furthermore, use these guidelines to structure the final output:\n\n"
    "<project_scope_output>\n"
    "Congratulations! We have completed the project scoping document for your analytics project. "
    "Here is the final version:\n\n"
    "Project Scope for [Insert Project Title]\n"
    "<section1_question_and_answer>\n"
    "<section2_question_and_answer>\n"
    "<section3_question_and_answer>\n"
    "<section4_question_and_answer>\n"
    "<section5_question_and_answer>\n"
    "<section6_question_and_answer>\n"
    "<section7_question_and_answer>\n"
    "<section8_question_and_answer>\n"
    "<section9_question_and_answer>\n"
    "<section10_question_and_answer>\n"
    "</project_scope_output>\n\n"
    "Thank you for taking the time to define your project in detail. This comprehensive project "
    "scope will serve as a valuable guide throughout the execution of your analytics project. If "
    "you have any further questions or need assistance, please don't hesitate to ask.\n"
)

def reset_messages():
    # Setting the appropriate system prompt based on the selected role
    if st.session_state.selected_role == "Project Scoping Coach":
        SYSTEM_PROMPT = project_scoping_prompt
    elif st.session_state.selected_role == "Premortem Coach":
        SYSTEM_PROMPT = premortem_prompt
    elif st.session_state.selected_role == "Reflection Coach":
        SYSTEM_PROMPT = reflective_prompt
    elif st.session_state.selected_role == "Devil's Advocate":
        SYSTEM_PROMPT = devils_advocate_prompt
    elif st.session_state.selected_role == "Team Structuring Coach":
        SYSTEM_PROMPT = structuring_prompt
    else: # just in case we change our code in the future
      raise ValueError("Invalid role selected")
    st.session_state.messages = [{
      "role": "system",
      "content": SYSTEM_PROMPT
    }, {
      "role": "user", 
      "content": "Hi, I am an undergraduate student studying business analytics."
      }]

# -----------------------------------------------------------------------------
# Manage page tracking and associated session state
# -----------------------------------------------------------------------------
THIS_PAGE = "project_coach"
if "cur_page" not in st.session_state:
    st.session_state.cur_page = THIS_PAGE

if ("token_counts" not in st.session_state) or (st.session_state.cur_page != THIS_PAGE):
    st.session_state.token_counts = {model: {"input_tokens": 0, "output_tokens": 0} for model in models}

# Purge messages if coming from a different page
if (st.session_state.cur_page != THIS_PAGE) and ("messages" in st.session_state):
    del st.session_state.messages

# Initialize messages for this page
if "messages" not in st.session_state:
  st.session_state.messages = [{
    "role": "system",
    "content": project_scoping_prompt
  }, {
    "role": "user", 
    "content": "Hi, I am an undergraduate student studying business analytics."
    }]

st.session_state.cur_page = THIS_PAGE
# -----------------------------------------------------------------------------
 
# -----------------------------------------------------------------------------

# Streamlit Title:
# ----------------

st.set_page_config(page_title = "ChatISA Project Coach", layout = "centered", page_icon='assets/favicon.png')
# -----------------------------------------------------------------------------


# Radio Button to Select Role:
# ----------------------------
st.sidebar.markdown("### What Role you Want the AI to Play")
role = st.sidebar.radio(
    "Select the role you want the AI to play:",
    ["Devil's Advocate", "Premortem Coach", "Project Scoping Coach", "Reflection Coach", "Team Structuring Coach"],
    index=2,
    label_visibility='collapsed',
    on_change=reset_messages,
    key="selected_role"
)

# Use an expander to provide more information about each role
with st.sidebar.expander("Learn more about the roles"):
    st.write("**Devil's Advocate**: Helps teams think through decisions by playing devil's advocate.")
    st.write("**Premortem Coach**: Helps teams perform a project premortem by encouraging them to envision possible failures and how to avoid them.")
    st.write("**Project Scoping Coach**: Guides users through defining an analytics project using an internal project scoping document template.")
    st.write("**Reflection Coach**: Assists teams in reflecting on their experiences in a structured way to derive lessons and insights.")
    st.write("**Team Structuring Coach**: Helps teams recognize and make use of the resources and expertise within the team.")


# The Streamlit Interface:
# ------------------------

# Streamlit Title:
# ----------------
st.markdown("## 🤖 ChatISA: Project Coach 🤖")


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
if prompt := st.chat_input("Discuss your project: current (devil's advocate, scoping, premortem, or structuring) or past (reflection)."):
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
      max_num_tokens = 3000
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
