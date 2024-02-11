"""
ChatISA App
-----------

This Streamlit application serves as an interactive chat interface powered by OpenAI's GPT model. It is designed to assist students in understanding concepts related to business analytics by engaging in a conversational format. The app allows users to input questions or topics, and it generates responses that aim to explain ideas, provide examples, and if applicable, share code snippets in R and Python. Additionally, users have the option to export the chat conversation as a PDF, which includes custom formatting for text and code, along with a footer indicating the document's generation details.

Features:
- Free to use for Miami University students (in an effort to reduce the AI access gap among our students).
- Interactive chat interface with a conversational AI model.
- The context for the AI model is set to generate responses for students majoring in business analytics.
- Export functionality to download the chat history as a formatted PDF document.

Created by: [Fadel M. Megahed](https://miamioh.edu/fsb/directory/?up=/directory/megahefm)
Version: 1.0.0
"""

import os
from dotenv import load_dotenv
import openai
import streamlit as st
from fpdf import FPDF
import tempfile
from datetime import datetime
import re

# Load the API key for OpenAI
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

st.markdown("## 🤖 ChatISA")

# Initialize session state variables
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-1106-preview"

if "chat_started" not in st.session_state:
    st.session_state["chat_started"] = False

if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.5

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "You are an upbeat, encouraging tutor who helps undergraduate students majoring in business analytics understand concepts by explaining ideas and asking students questions. Start by introducing yourself to the student as their ChatISA Assistant who is happy to help them with any questions. Only ask one question at a time. Ask them about the subject title and topic they want to learn about. Wait for their response.  Given this information, help students understand the topic by providing explanations, examples, and analogies. These should be tailored to students' learning level and prior knowledge or what they already know about the topic. When appropriate also provide them with code in both R (use tidyverse styling) and Python (use pandas whenever possible), showing them how to implement whatever concept they are asking about. Note if you show R code, please use pkg_name::function_name() syntax, and use the |> as your pipe operator. On the other hand for Python, break chained methods into multiple lines using parentheses; for example, do NOT write df.groupby('Region')['Sales'].agg('sum') on one line."}]

if not st.session_state["chat_started"]:
    st.session_state.messages.append({"role": "user", "content": "Hi, I am an undergraduate student studying business analytics."})
    st.session_state["chat_started"] = True

class PDF(FPDF):
    def __init__(self, student_name, course_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.student_name = student_name
        self.course_name = course_name
        
    def header(self):
        # image dimensions and margin
        image_width = 33
        right_margin = 10
        x_right_aligned = self.w - image_width - right_margin
        self.image('logo-horizontal-stacked.png', x=x_right_aligned, y=8, w=image_width)
        
        self.set_y(8)
        self.set_font("Arial", size=8)
        self.cell(0, 10, f"{student_name}'s ChatISA Interaction for {course_name}", 0, 0, 'L')
        # Add a line break
        self.ln(20)
    
    def footer(self):
        self.set_y(-15)  # Position at 15 units from bottom
        self.set_font("Arial", size=8)
        date_generated = datetime.now().strftime("%Y-%m-%d")
        self.cell(0, 10, 'Generated on ' + date_generated, 0, 0, 'L')
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'R')

def create_pdf(chat_messages, student_name, course_name):
    pdf = PDF(student_name, course_name)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font('Arial', 'B', 16)
    current_date = datetime.now().strftime("%Y-%m-%d")
    pdf.cell(0, 10, f"{student_name}'s interaction with ChatISA on {current_date}", 0, 1, 'C')
    pdf.ln(10)

    # Introductory Text with Background
    pdf.set_fill_color(200, 16, 46)  # Miami background
    pdf.set_text_color(255, 255, 255)  # White text
    pdf.set_font('Arial', 'B', 14)
    pdf.multi_cell(0, 10, 'ChatISA Setup', 0, 'L', 1)
    pdf.set_font('Arial', '', 11)
    intro_text = f"The purpose behind ChatISA is to make AI access more inclusive for FSB and Miami University students, with costs covered by an industry sponsor. This tool aims to empower students to leverage AI creatively and responsibly using GPT-4-Turbo. For more information, visit the OpenAI API documentation at https://openai.com/api/. This document includes an export of my conversation with ChatISA for the coursework related to {course_name}. \nThe first message from ChatISA and the Student serve as context for more informed generated text (following best recommendations from the AI and education literatures); {student_name}'s interaction with ChatISA starts at Page 2."
    pdf.multi_cell(0, 10, intro_text, 0, 'L', 1)
    pdf.ln(10)

    # Reset text color for chat messages
    pdf.set_text_color(0, 0, 0)
    
    for message in chat_messages:
        role = message["role"]
        content = message["content"]
        content = content.replace("\n\n", "\n")  # Remove double new lines

        parts = re.split(r'(```\w+?\n.*?```)', content, flags=re.DOTALL)
        for part in parts:
            if re.match(r'```(\w+)?\n(.*?)```', part, re.DOTALL):
                code = re.findall(r'```(\w+)?\n(.*?)```', part, re.DOTALL)[0][1]
                pdf.set_font("Courier", size=10)  # Monospaced font for code
                pdf.set_fill_color(230, 230, 230)  # Light gray background for code
                pdf.multi_cell(0, 10, txt=code, fill=True)
                pdf.set_font("Arial", size=11)  # Reset font for normal text
                pdf.ln(5)  # Space after code block
            else:
                if role == 'user':
                    pdf_role = f"{student_name}"
                    pdf.set_fill_color(255, 235, 224)  # Light red
                else:
                    pdf_role = 'ChatISA'
                    pdf.set_fill_color(255, 255, 255)  # White
                pdf.multi_cell(0, 10, txt=f"{pdf_role}: {part}", fill=True)
                pdf.ln(3) if role == 'user' else pdf.ln(6)

    # Save the PDF in a temporary file and return the file path
    pdf_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name
    pdf.output(pdf_output_path)
    return pdf_output_path

# Streamlit UI components for chat display
for message in st.session_state.messages[2:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me questions about data visualization, forecasting, predictive modeling, or other busniess analytics topics."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
            temperature=st.session_state["temperature"],
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})


# PDF Generation
# Button to trigger the export process
if st.button('Prepare to Export Chat as PDF'):
    # Set flags in the session state to indicate that the export process has started
    st.session_state['prepare_export'] = True

# Check if the export process has started
if 'prepare_export' in st.session_state and st.session_state['prepare_export']:
    # Collect student's name and course name
    student_name = st.text_input("Enter your name:", key="student_name")
    course_name = st.text_input("Enter your course name:", key="course_name")

    # Button to actually create and download the PDF, only shown after entering the details
    if st.button('Export Chat as PDF'):
        pdf_file = create_pdf(st.session_state.messages, student_name, course_name)  # Adjusted to your create_pdf function
        with open(pdf_file, "rb") as file:
            st.download_button(label="Download PDF", data=file, file_name="chatisa_export.pdf", mime="application/pdf")
        # Reset the export flag to hide inputs after downloading
        st.session_state['prepare_export'] = False

# ------------------------------------------------------------------------------------------
# Sidebar information about the app, creator, version, disclaimers, references, and funding
# ------------------------------------------------------------------------------------------
st.sidebar.markdown("""
    - **Created by:** [Fadel M. Megahed](https://miamioh.edu/fsb/directory/?up=/directory/megahefm)  
    
    - **Version:** 1.0.0 (Feb 10, 2024) using the [GPT-4 Turbo with 128K context window](https://openai.com/blog/new-models-and-developer-products-announced-at-devday)  

    - **Disclaimers:**
        + ChatISA is for **educational purposes only**   
        + ChatISA's use in classwork has to be approved by instructor  
        + **MU and FSB are not responsible for the content generated by this app**  
        + **Use at your own risk** & **always** evaluate the accuracy of the generated answer  
        
    - **References:**  
        + **Prompt Engineering:** Adapted from [Assigning AI by Mollick and Mollic 2023](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4475995)  
        + **Streamlit App:** Adapted from [ChatGPT Apps with Streamlit](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-chatgpt-like-app)   
    
    - **Funding:**   
       + Thanks to **US Bank's** generous support, FSB can cover all API costs, allowing free use by our students.
""")



