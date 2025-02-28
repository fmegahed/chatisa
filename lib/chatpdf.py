from datetime import datetime
import re
from fpdf import FPDF
import tempfile

token_cost_rates = {
    # GPT costs per https://openai.com/pricing on 2025-02-28
    'gpt-4o': {'input_cost_per_million_tokens': 2.5, 'output_cost_per_million_tokens': 10},
    'gpt-4o-mini': {'input_cost_per_million_tokens': 0.15, 'output_cost_per_million_tokens': 0.6},
    'gpt-4.5-preview-2025-02-27': {'input_cost_per_million_tokens': 75, 'output_cost_per_million_tokens': 150},
    
    # Anthropic costs per https://www.anthropic.com/pricing#anthropic-api on 2025-02-28
    'claude-3-7-sonnet-20250219': {'input_cost_per_million_tokens': 3, 'output_cost_per_million_tokens': 15},
    
    # Cohere costs per https://cohere.com/pricing on 2025-02-28
    'command-r-plus': {'input_cost_per_million_tokens': 2.5, 'output_cost_per_million_tokens': 10},
    
    # Groq costs per https://wow.groq.com/ on 2025-02-28
    'llama-3.1-8b-instant': {'input_cost_per_million_tokens': 0.05, 'output_cost_per_million_tokens': 0.08},
    'llama-3.3-70b-versatile': {'input_cost_per_million_tokens': 0.59, 'output_cost_per_million_tokens': 0.79},
    'gemma2-9b-it': {'input_cost_per_million_tokens': 0.2, 'output_cost_per_million_tokens': 0.2}
}



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
  for original, replacement in LATIN_REPLACEMENTS.items():
    input_text = input_text.replace(original, replacement)
  return input_text.encode("latin-1", "ignore").decode("latin-1")

def create_pdf(chat_messages, models, token_counts, user_name, user_course):
  
  # Identifying the models used in this conversation (i.e., ones with token_counts > 0)
  subset_models = {model: token_counts[model] for model in models if token_counts[model]['input_tokens'] > 0 or token_counts[model]['output_tokens'] > 0}

  # A formatted string with labels (a), (b), ..., (h)
  formatted_models = ', '.join(f"({chr(97 + i)}) {model}" for i, model in enumerate(subset_models))
  parts = formatted_models.rsplit(', ', 1)
  formatted_models_with_and = ' and '.join(parts)
  
  # A formatted string of models with their token counts in parentheses
  formatted_token_counts = ', '.join(f"({chr(97 + i)}) {model} ({token_counts[model]['input_tokens']}, {token_counts[model]['output_tokens']})" for i, model in enumerate(subset_models))
  parts = formatted_token_counts.rsplit(', ', 1)
  formatted_token_counts_with_and = ' and '.join(parts)
  
  # Calculate total input and output tokens
  total_input_tokens = sum(counts['input_tokens'] for counts in token_counts.values() if counts['input_tokens'] is not None)
  total_output_tokens = sum(counts['output_tokens'] for counts in token_counts.values() if counts['output_tokens'] is not None)
  
  # Breaking down the tokens and costs by model
  total_cost = 0.0
  model_details = []
  
  for model, counts in token_counts.items():
      input_tokens = counts['input_tokens']
      output_tokens = counts['output_tokens']
      
      if output_tokens > 0:
        input_cost = (input_tokens / 1_000_000) * token_cost_rates[model]['input_cost_per_million_tokens']
        output_cost = (output_tokens / 1_000_000) * token_cost_rates[model]['output_cost_per_million_tokens']
        model_cost = input_cost + output_cost
        total_cost += model_cost
        detail = f"{model} (Input: {input_tokens} tokens @ ${input_cost:.4f}, Output: {output_tokens} tokens @ ${output_cost:.4f}, Total: ${model_cost:.4f})"
        model_details.append(detail)
  
  # formatting the summary paragraph similar to formatted models
  models_summary = ', '.join(f"({chr(97 + i)}) {model}" for i, model in enumerate(model_details))
  parts = models_summary.rsplit(', ', 1)
  models_summary = ' and '.join(parts)

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
