"""
ChatISA App
-----------

This Streamlit application serves as an interactive chat interface powered by OpenAI's GPT model. It is designed to assist students in understanding concepts related to business analytics by engaging in a conversational format. The app allows users to input questions or topics, and it generates responses that aim to explain ideas, provide examples, and if applicable, share code snippets in R and Python. Additionally, users have the option to export the chat conversation as a PDF, which includes custom formatting for text and code, along with a footer indicating the document's generation details.

Features:
- Free to use for Miami University students (in an effort to reduce the AI access gap among our students).  
- The context for the AI model is set to generate responses for students majoring in business analytics.  
- ChatGPT is instructed to utilize tidyverse, the native pipe |> as the pipe operator for R code, and pkg::function() syntax for R code (following the pedagogy in our classes). 
- ChatGPT is instructed to break chained methods into multiple lines using parentheses for Python code.
- Export functionality to download the chat history as a formatted PDF document, which includes the student's name, course name, and a footer with the date of generation. The first page includes information about the setup of the app and the estimated cost for querying and generating the answers. In addition, we use light red boxes to indicate the student's query and light gray backgrounds for code chunks.

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
import tiktoken

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
    st.session_state.messages = [{
      "role": "system", 
      "content": ("""
      You are an upbeat, encouraging tutor who helps undergraduate students majoring
      in business analytics understand concepts by explaining ideas and asking students questions. 
      Start by introducing yourself to the student as their ChatISA Assistant who is happy to help them 
      with any questions. 
      
      Only ask one question at a time. Ask them about the subject title and topic they want to learn 
      about. Wait for their response.  Given this information, help students understand the topic by 
      providing explanations, examples, and analogies. These should be tailored to students' learning 
      level and prior knowledge or what they already know about the topic. When appropriate also 
      provide them with code in both R (use tidyverse styling) and Python (use pandas whenever possible), 
      showing them how to implement whatever concept they are asking about. 
      
      When you show R code, you must use: 
        (a) library_name::function_name() syntax as this avoids conflicts in function names and makes it 
        clear to the student where the function is imported from when there are multiple packages loaded.  
        Based on this, do NOT use library() in the beginning of your code chunk and use 
        if(require(library)==FALSE) install.packages(library), and 
        (b) use the native pipe |> as your pipe operator. 
        
      On the other hand for Python, break chained methods into multiple lines using parentheses; 
      for example, do NOT write df.groupby('Region')['Sales'].agg('sum') on one line."""
      )
      }]

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
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, f"{student_name}'s ChatISA Interaction for {course_name}", 0, 0, 'L')
        # Add a line break
        self.ln(20)
    
    def footer(self):
        self.set_y(-15)  # Position at 15 units from bottom
        self.set_font("Arial", size=8)
        self.set_text_color(0, 0, 0)
        date_generated = datetime.now().strftime("%b %d, %Y")
        self.cell(0, 10, 'Generated with love, yet without guarantees, on ' + date_generated, 0, 0, 'L')
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'R')

def calculate_cost(chat_messages, model_name="gpt-4-1106-preview"):
    encoder = tiktoken.encoding_for_model(model_name)
    total_input_tokens, total_output_tokens = 0, 0

    for message in chat_messages:
        if message["role"] == "user":
            total_input_tokens += len(encoder.encode(message["content"]))
        else:  # Assuming all other roles are responses or system messages
            total_output_tokens += len(encoder.encode(message["content"]))

    input_cost = 0.01 / 1000 * total_input_tokens
    output_cost = 0.03 / 1000 * total_output_tokens
    total_cost = input_cost + output_cost

    return total_input_tokens, total_output_tokens, total_cost

def create_pdf(chat_messages, student_name, course_name):
    # Calculate costs and token counts
    total_input_tokens, total_output_tokens, total_cost = calculate_cost(chat_messages)
    num_requests = sum(1 for msg in chat_messages if msg["role"] == "user")
    num_responses = sum(1 for msg in chat_messages if msg["role"] != "user")
    
    pdf = PDF(student_name, course_name)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font('Arial', 'B', 16)
    current_date = datetime.now().strftime("%b %d, %Y")
    pdf.cell(0, 10, f"{student_name}'s Interaction with ChatISA on {current_date}", 0, 1, 'C')
    pdf.ln(3)

    # Introductory Text
    # ------------------
    pdf.set_draw_color(200, 16, 45)
    pdf.set_line_width(1) 
    margin = 10  # Margin in mm
    page_width = 210  # letterpaper width in mm
    # line_y_position = pdf.get_y()
    # pdf.line(margin, line_y_position, page_width - margin, line_y_position)
    pdf.ln(3)  # Space after the line
    
    pdf.set_fill_color(255, 255, 255)  # White background
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(200, 16, 46)  # Miami red text
    pdf.multi_cell(0, 10, "ChatISA's Purpose", 0, 'L', 1)
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(0, 0, 0)  # Black text
    intro_text = f"The purpose behind ChatISA is to make AI access more inclusive for FSB and Miami University students, with costs covered by an industry sponsor. This tool aims to empower students to leverage AI creatively and responsibly using GPT-4-Turbo. For more information, visit the OpenAI API documentation at https://openai.com/api/. This document includes an export of my conversation with ChatISA for the coursework related to {course_name}."
    pdf.multi_cell(0, 10, intro_text, 0, 'L', 1)
  
    line_y_position = pdf.get_y() + 3  # Small gap after the text
    pdf.line(margin, line_y_position, page_width - margin, line_y_position)
    pdf.ln(6)  # Space after the line
    
    # Layout of the ChatISA Interaction
    # ---------------------------------
    pdf.set_text_color(200, 16, 46)  # Miami red text
    pdf.set_font('Arial', 'B', 14)
    pdf.multi_cell(0, 10, "ChatISA's PDF Output Style and Layout", 0, 'L', 1)
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(0, 0, 0)  # Black text
    layout_text = f"The PDF is designed to provide a clear and organized record of the interaction between the student and ChatISA. Specifically, student prompts are highlighted in light red boxes. ChatISA's responses are formatted with a light gray background for code snippets and a white background for text. This formatting is intended to improve readability and provide a clear visual distinction between the student's messages and ChatISA's responses. The PDF is intended to be used for educational and reference purposes related to the coursework for {course_name}."
    pdf.multi_cell(0, 10, layout_text, 0, 'L', 1)
  
    line_y_position = pdf.get_y() + 4  # Small gap after the text
    pdf.line(margin, line_y_position, page_width - margin, line_y_position)
    pdf.ln(6)  # Space after the line
    
    # Cost and Token Counts
    # ----------------------
    pdf.set_text_color(200, 16, 46)  # Miami red text
    pdf.set_font('Arial', 'B', 14)
    pdf.multi_cell(0, 10, 'Estimated Cost and Token Counts', 0, 'L', 1)
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(0, 0, 0)  # Black text
    cost_text = f"The estimated cost for this interaction is ${total_cost:.3f}. The estimated cost is based on the current pricing for GPT-4 Turbo: https://openai.com/pricing. Using the tiktoken python library, the estimated total number of tokens used in the input is {total_input_tokens}, and the estimated total number of tokens used in the output = {total_output_tokens}. The estimated counts include ChatISA's custom instructions and initial user message."
    pdf.multi_cell(0, 10, cost_text, 0, 'L', 1)
    line_y_position = pdf.get_y() + 3  # Small gap after the text
    pdf.line(margin, line_y_position, page_width - margin, line_y_position)
    pdf.ln(6)  # Space after the line
    
    
    # ChatISA's Interaction with the Student
    # --------------------------------------
    pdf.set_fill_color(255, 255, 255)  # White background
    pdf.set_text_color(200, 16, 46)  # Miami red text
    pdf.set_font('Arial', 'B', 14)
    pdf.multi_cell(0, 10, f"{student_name}'s Interaction with ChatISA", 0, 'L', 1)
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(0, 0, 0)  # Black text
    
    for message in chat_messages[2:]:
        role = message["role"]
        content = message["content"]
        content = content.replace("\n\n", "\n")  # Remove double new lines
        content = content.replace("\n\n", "\n")  # Remove double new lines again

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
    
    line_y_position = pdf.get_y() + 3  # Small gap after the text
    pdf.line(margin, line_y_position, page_width - margin, line_y_position)
    pdf.ln(6)  # Space after the line
    
    
    # Custom Instructions and Default Message:
    # ---------------------------------------
    pdf.set_fill_color(255, 255, 255)  # White background
    pdf.set_text_color(200, 16, 46)  # Miami red text
    pdf.set_font('Arial', 'B', 14)
    pdf.multi_cell(0, 10, "Appendix: ChatISA's Custom Instructions and Default User Message", 0, 'L', 1)
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(0, 0, 0)  # Black text
    
    for message in chat_messages[:2]:
        role = message["role"]
        content = message["content"]
        content = re.sub(r'\n\s*\n', '\n', content) # Remove double new lines

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
    
    line_y_position = pdf.get_y() + 15  # Small gap after the text
    pdf.line(50, line_y_position, page_width - 50, line_y_position)
    
    
    # Save the PDF in a temporary file and return the file path
    # ---------------------------------------------------------
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
    
    - **Version:** 1.0.0 (Feb 11, 2024) using the [GPT-4 Turbo with 128K context window](https://openai.com/blog/new-models-and-developer-products-announced-at-devday)  

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



