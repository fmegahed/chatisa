from datetime import datetime
import re
from fpdf import FPDF
import tempfile

# Import centralized configuration and logging
from config import MODELS, calculate_cost, get_model_display_name
from lib.enhanced_usage_logger import log_pdf_export

LATIN_REPLACEMENTS = {
    "\u2014": "--",     # em dash
    "\u2013": "-",      # en dash
    "\u2018": "'",      # left single quotation mark
    "\u2019": "'",      # right single quotation mark
    "\u201C": "\"",     # left double quotation mark
    "\u201D": "\"",     # right double quotation mark
    "\u2026": "...",    # ellipsis
    "\u00A0": " ",      # non-breaking space
    "\U0001f60a": ":)", # smiling face emoji
}


# Create a latin-1 encoded friendly text for the PDF
def clean_text(input_text):
  if input_text is None:
    return ""
  for original, replacement in LATIN_REPLACEMENTS.items():
    input_text = input_text.replace(original, replacement)
  return input_text.encode("latin-1", "ignore").decode("latin-1")

def create_pdf(chat_messages, models, token_counts, user_name, user_course):
  
  # Get current page for logging
  import streamlit as st
  current_page = getattr(st.session_state, 'cur_page', 'unknown')
  
  # Identifying the models used in this conversation (i.e., ones with token_counts > 0)
  subset_models = {model: token_counts[model] for model in models if token_counts[model]['input_tokens'] > 0 or token_counts[model]['output_tokens'] > 0}

  # Format model names based on single vs. multiple models
  if len(subset_models) == 1:
    # Single model - no letter labeling needed
    model_name = get_model_display_name(list(subset_models.keys())[0])
    formatted_models_with_and = model_name
  else:
    # Multiple models - use (a), (b), etc. labeling
    formatted_models = ', '.join(f"({chr(97 + i)}) {get_model_display_name(model)}" for i, model in enumerate(subset_models))
    parts = formatted_models.rsplit(', ', 1)
    formatted_models_with_and = ' and '.join(parts)
  
  # Format token counts based on single vs. multiple models
  if len(subset_models) == 1:
    # Single model - no letter labeling needed
    model = list(subset_models.keys())[0]
    model_name = get_model_display_name(model)
    formatted_token_counts_with_and = f"{model_name} ({token_counts[model]['input_tokens']}, {token_counts[model]['output_tokens']})"
  else:
    # Multiple models - use (a), (b), etc. labeling
    formatted_token_counts = ', '.join(f"({chr(97 + i)}) {get_model_display_name(model)} ({token_counts[model]['input_tokens']}, {token_counts[model]['output_tokens']})" for i, model in enumerate(subset_models))
    parts = formatted_token_counts.rsplit(', ', 1)
    formatted_token_counts_with_and = ' and '.join(parts)
  
  # Calculate total input and output tokens
  total_input_tokens = sum(counts['input_tokens'] for counts in token_counts.values() if counts['input_tokens'] is not None)
  total_output_tokens = sum(counts['output_tokens'] for counts in token_counts.values() if counts['output_tokens'] is not None)
  
  # Breaking down the tokens and costs by model using centralized config
  total_cost = 0.0
  model_details = []
  
  for model, counts in token_counts.items():
      input_tokens = counts['input_tokens']
      output_tokens = counts['output_tokens']
      
      if output_tokens > 0 and model in MODELS:
          # Use centralized cost calculation
          cost_info = calculate_cost(model, input_tokens, output_tokens)
          model_cost = cost_info["total_cost"]
          input_cost = cost_info["input_cost"]
          output_cost = cost_info["output_cost"]
          
          total_cost += model_cost
          display_name = get_model_display_name(model)
          detail = f"{display_name} (Input: {input_tokens} tokens @ ${input_cost:.4f}, Output: {output_tokens} tokens @ ${output_cost:.4f}, Total: ${model_cost:.4f})"
          model_details.append(detail)
  
  # Format the summary paragraph based on single vs. multiple models
  if len(model_details) == 1:
    # Single model - no letter labeling needed
    models_summary = model_details[0]
  elif len(model_details) > 1:
    # Multiple models - use (a), (b), etc. labeling
    models_summary = ', '.join(f"({chr(97 + i)}) {model}" for i, model in enumerate(model_details))
    parts = models_summary.rsplit(', ', 1)
    models_summary = ' and '.join(parts)
  else:
    models_summary = "no model usage was recorded"

  summary_paragraph = (
    f"The total number of tokens used in the chat is {total_input_tokens + total_output_tokens}, "
    f"comprising {total_input_tokens} input tokens and {total_output_tokens} output tokens. "
    f"The total cost for all tokens is ${total_cost:.3f}. "
    f"Costs are distributed across the models as follows: {models_summary}."
)


  # Initialize pdf
  # --------------
  pdf = PDF(user_name, user_course, format="Letter")
  pdf.add_page()
  pdf.set_auto_page_break(auto=True, margin=pdf.margin)

  # Document Title
  # --------------
  pdf.set_font("Arial", "B", 16)
  pdf.cell(0, pdf.margin, f"{user_name}'s Interaction with ChatISA on {pdf.date}", 0, 1, "C")
  pdf.ln(3)

  # Introductory Text
  # -----------------
  verb = "was" if len(subset_models) == 1 else "were"
  
  draw_heading(pdf, "ChatISA's Purpose")
  pdf.multi_cell(
    0, pdf.margin,
    f"The purpose behind ChatISA is to make AI access more inclusive for FSB and Miami University students, with costs covered by an industry sponsor. This chatbot aims to empower students to leverage AI creatively and responsibly. This document includes an export of my conversation with ChatISA for the coursework related to {pdf.user_course}. {formatted_models_with_and} {verb} used to generate the responses.",
    0, "L", True
  )
  draw_divider(pdf)

  # Explanation of document
  # -----------------------
  draw_heading(pdf, "ChatISA's PDF Output Style and Layout")
  pdf.multi_cell(
    0, pdf.margin,
    f"The purpose of the PDF is to provide a clear and well-organized record of the interaction between the student and ChatISA. The student's prompts are highlighted in light red boxes, while ChatISA's responses are formatted with a light gray background for code snippets and a white background for text. This formatting is intended to improve readability and provide a clear visual distinction between the student's messages and ChatISA's responses. The PDF is designed for educational and reference purposes related to the coursework for {user_course}.\nStarting from Page 2, the PDF includes {user_name}'s queries and ChatISA's responses. Additionally, the custom instructions that guide ChatISA's responses are included on the last page of the PDF in the appendix.",
    0, "L", True
  )
  draw_divider(pdf)

  # Cost and token counts
  # ---------------------
  draw_heading(pdf, "Token Counts and Cost Breakdown")
  pdf.multi_cell(
    0, pdf.margin,
    f"{summary_paragraph}",
    0, "L", True
  )
  draw_divider(pdf)

  # Page break before user interaction
  # ----------------------------------
  pdf.add_page()

  # ChatISA's Interaction with the user
  # -----------------------------------
  draw_heading(pdf, f"{pdf.user_name}'s Interaction with ChatISA")

  for message in chat_messages[2:]:
    role = message["role"]
    content = clean_text(re.sub(r"\n\s*\n", "\n", message["content"]))

    if role == "user":
      pdf.set_fill_color(255, 235, 224)
      pdf.multi_cell(0, pdf.margin, f"{pdf.user_name}:", fill=True)
    else:
      pdf.set_fill_color(255, 255, 255)
      pdf.multi_cell(0, pdf.margin, "ChatISA:", fill=True)

    for part in re.split(r"(```\w+?\n.*?```)", content, flags=re.DOTALL):
      if re.match(r"```(\w+)?\n(.*?)```", part, flags=re.DOTALL): # code chunk
        code = re.findall(r"```(\w+)?\n(.*?)```", part, flags=re.DOTALL)[0][1]
        pdf.set_font("Courier", size=10)
        pdf.set_fill_color(230, 230, 230)
        pdf.multi_cell(0, pdf.margin, code, fill=True)
        pdf.set_font("Arial", size=11)
        pdf.ln(5)
      else: # no code chunk - text
        if role == "user":
          pdf.set_fill_color(255, 235, 224)
          pdf.multi_cell(0, pdf.margin, part, fill=True)
          pdf.ln(3)
        else:
          pdf.set_fill_color(255, 255, 255)
          pdf.multi_cell(0, pdf.margin, part, fill=True)
          pdf.ln(6)

  draw_divider(pdf)

  # Page break before appendix
  # --------------------------
  pdf.add_page()

  # Appendix: Custom instructions and default message
  # -------------------------------------------------
  draw_heading(pdf, "Appendix: ChatISA's Custom Instructions and Default User Message")

  for message in chat_messages[:2]:
    role = message["role"]
    content = clean_text(re.sub(r"\n\s*\n", "\n", message["content"]))

    if role == "user":
      pdf.set_fill_color(255, 235, 224)
      pdf.multi_cell(0, pdf.margin, f"{pdf.user_name}: {content}", fill=True)
      pdf.ln(3)
    else:
      pdf.set_fill_color(255, 255, 255)
      pdf.multi_cell(0, pdf.margin, f"ChatISA: {content}", fill=True)
      pdf.ln(6)

  # Save the pdf and return the file path
  # -------------------------------------
  pdf_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
  pdf.output(pdf_output_path)
  
  # Log PDF export (privacy-compliant - no personal info)
  filename = f"{user_course}_{user_name}_chatisa.pdf"
  log_pdf_export(
      page=current_page,
      data={
          "course": user_course,
          "file_type": "pdf"
      }
  )
  
  return pdf_output_path

def create_sandbox_pdf(sandbox_messages, user_name, user_course, total_cost=0.0):
  """
  Create PDF for AI Sandbox interactions with code executions and embedded images.

  Args:
    sandbox_messages: List of messages with structure:
      {
        "role": "user" or "assistant",
        "content": "text content",
        "code_executions": [  # Only for assistant messages
          {
            "code": "python code",
            "outputs": [
              {"type": "image", "image": {"b64_json": "...", "filename": "..."}}
            ]
          }
        ]
      }
    user_name: Student's name
    user_course: Course name
    total_cost: Total cost of API usage
  """

  # Get current page for logging
  import streamlit as st
  current_page = getattr(st.session_state, 'cur_page', 'ai_sandbox')

  # Initialize pdf
  pdf = PDF(user_name, user_course, format="Letter")
  pdf.add_page()
  pdf.set_auto_page_break(auto=True, margin=pdf.margin)

  # Document Title
  pdf.set_font("Arial", "B", 16)
  pdf.cell(0, pdf.margin, f"{user_name}'s AI Sandbox Session on {pdf.date}", 0, 1, "C")
  pdf.ln(3)

  # Introductory Text
  draw_heading(pdf, "AI Sandbox Purpose")
  pdf.multi_cell(
    0, pdf.margin,
    f"The AI Sandbox provides a secure Python code execution environment for computational problem-solving and data analysis. This document exports the conversation between {user_name} and ChatISA's AI Sandbox for coursework related to {user_course}. The GPT-5.2 model with code interpreter was used to generate responses and execute Python code.",
    0, "L", True
  )
  draw_divider(pdf)

  # Explanation of document
  draw_heading(pdf, "PDF Output Style and Layout")
  pdf.multi_cell(
    0, pdf.margin,
    f"The PDF provides a clear record of the interaction. Student prompts appear in light red boxes. ChatISA's text responses have a white background, while executed Python code appears with a gray background. Generated visualizations and graphs are embedded directly in the document. This formatting improves readability and provides clear distinction between prompts, code, and outputs.",
    0, "L", True
  )
  draw_divider(pdf)

  # Cost information
  if total_cost > 0:
    draw_heading(pdf, "Usage Cost")
    pdf.multi_cell(
      0, pdf.margin,
      f"The total cost for this AI Sandbox session is ${total_cost:.4f}. This includes costs for code execution, image processing, and API token usage.",
      0, "L", True
    )
    draw_divider(pdf)

  # Page break before interaction
  pdf.add_page()

  # Sandbox Interaction
  draw_heading(pdf, f"{pdf.user_name}'s AI Sandbox Interaction")

  for message in sandbox_messages:
    role = message["role"]
    content = clean_text(message.get("content", ""))

    # User message
    if role == "user":
      pdf.set_fill_color(255, 235, 224)
      pdf.multi_cell(0, pdf.margin, f"{pdf.user_name}:", fill=True)
      pdf.multi_cell(0, pdf.margin, content, fill=True)
      pdf.ln(3)

    # Assistant message
    else:
      pdf.set_fill_color(255, 255, 255)
      pdf.multi_cell(0, pdf.margin, "ChatISA:", fill=True)

      # Text content
      if content.strip():
        pdf.multi_cell(0, pdf.margin, content, fill=True)
        pdf.ln(3)

      # Code executions
      if "code_executions" in message and message["code_executions"]:
        for execution in message["code_executions"]:
          code = clean_text(execution.get("code", ""))

          # Display code
          if code.strip():
            pdf.set_font("Courier", size=10)
            pdf.set_fill_color(230, 230, 230)
            pdf.multi_cell(0, pdf.margin, code, fill=True)
            pdf.set_font("Arial", size=11)
            pdf.ln(3)

          # Display images from outputs
          if "outputs" in execution and execution["outputs"]:
            for output in execution["outputs"]:
              if output["type"] == "image" and output.get("image", {}).get("b64_json"):
                try:
                  # Decode base64 image
                  import base64
                  from io import BytesIO
                  from PIL import Image

                  image_b64 = output["image"]["b64_json"]
                  image_bytes = base64.b64decode(image_b64)
                  image = Image.open(BytesIO(image_bytes))

                  # Save to temporary file
                  import tempfile
                  temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                  image.save(temp_image.name, format="PNG")
                  temp_image.close()

                  # Calculate image dimensions to fit page width
                  max_width = pdf.w - 2 * pdf.margin
                  img_width = image.width
                  img_height = image.height
                  aspect_ratio = img_height / img_width

                  # Scale to fit within page width
                  display_width = min(max_width, img_width * 0.75)  # 75% of original or page width
                  display_height = display_width * aspect_ratio

                  # Check if image fits on current page (leave margin for bottom)
                  available_height = pdf.h - pdf.get_y() - pdf.margin - 20
                  if display_height > available_height:
                    pdf.add_page()  # Start new page for image

                  # Center the image
                  x_centered = (pdf.w - display_width) / 2

                  # Add image
                  pdf.image(temp_image.name, x=x_centered, w=display_width)
                  pdf.ln(5)

                  # Clean up temp file
                  import os
                  os.unlink(temp_image.name)

                except Exception as img_error:
                  # If image embedding fails, note it
                  pdf.set_font("Arial", "I", 10)
                  pdf.multi_cell(0, pdf.margin, f"[Image could not be embedded: {str(img_error)}]")
                  pdf.set_font("Arial", size=11)
                  pdf.ln(3)

      pdf.ln(3)

  draw_divider(pdf)

  # Save the pdf
  pdf_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
  pdf.output(pdf_output_path)

  # Log PDF export
  filename = f"{user_course}_{user_name}_sandbox.pdf"
  log_pdf_export(
      page=current_page,
      data={
          "course": user_course,
          "file_type": "pdf",
          "sandbox": True
      }
  )

  return pdf_output_path

def draw_divider(pdf):
  y_position = pdf.get_y()+3
  pdf.set_draw_color(200, 16, 45)
  pdf.set_line_width(1)
  pdf.line(pdf.margin, y_position, pdf.w-pdf.margin, y_position)
  pdf.ln(6)

def draw_heading(pdf, text):
  pdf.set_fill_color(255, 255, 255)
  pdf.set_font("Arial", "B", 14)
  pdf.set_text_color(200, 16, 46)
  pdf.multi_cell(0, pdf.margin, text, 0, "L", True)
  pdf.set_font("Arial", size=11)
  pdf.set_text_color(0, 0, 0)

class PDF(FPDF):
  def __init__(self, user_name, user_course, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.user_name = user_name
    self.user_course = user_course
    self.date = datetime.now().strftime("%b %d, %Y")
    self.margin = 10

  def header(self):
    # Text
    self.set_y(8)
    self.set_font("Arial", size=8)
    self.cell(0, self.margin, f"{self.user_name}'s ChatISA interaction for {self.user_course}", 0, 0, "L")
    self.ln(20) # line break

    # FSB logo
    image_width = 33
    x_right_aligned = self.w - image_width - self.margin
    self.image("assets/logo-horizontal-stacked.png", x=x_right_aligned, y=8, w=image_width)

  def footer(self):
    self.set_y(-15)
    self.set_font("Arial", size=8)
    self.cell(0, self.margin, f"Generated with love, yet without guarantees, on {self.date}", 0, 0, "L")
    self.cell(0, self.margin, f"Page {self.page_no()}", 0, 0, "R")
