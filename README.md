# ChatISA v4.2.0

<div align="center">

![ChatISA Logo](https://github.com/fmegahed/chatisa/assets/22730186/3a0c2839-3384-428c-9aa5-e0cf95ba1296)

**🤖 Educational AI Assistant with Multiple LLM Support**

[![Version](https://img.shields.io/badge/version-4.1-blue.svg)](https://github.com/fmegahed/chatisa)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*Making AI accessible for educational excellence*

</div>

---

## 📖 Overview

ChatISA is an educational Streamlit web application that provides students access to multiple leading AI models through five specialized learning interfaces. Built for responsible academic use, it offers free access to cutting-edge AI technology with costs sponsored by industry partners.

### 🎯 Key Features

- **Multi-Model Support**: Access to 7+ leading AI models (OpenAI, Anthropic, Cohere, Groq)
- **Five Learning Modules**: Coding Companion, Project Coach, Exam Ally, Interview Mentor, AI Comparisons
- **Speech-to-Speech**: Real-time voice interaction using OpenAI Realtime API
- **Document Processing**: Native PDF, image, and document analysis for supported models
- **Model Comparison**: Side-by-side AI responses with minimal system prompts
- **Educational Focus**: Purpose-built for responsible academic use

## 🎥 Demo

<div style="position: relative; padding-bottom: 54.21436004162331%; height: 0;"><iframe src="https://www.loom.com/embed/6a83a717569e46ee80005044384aee53?sid=668a08ff-f6b6-4316-976b-4ab862625df9" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

*See all five learning modules in action and learn how to navigate the interface effectively.*

---

## 🛠️ Five Learning Modules

### 1. 💻 **Coding Companion**
Programming help with educational context. Features code generation, debugging assistance, and concept explanations for R, Python, and data analysis.

### 2. 🎯 **Project Coach**
Team project guidance with specialized coaching roles including Project Manager, Risk Analyst, Team Advisor, and Devil's Advocate.

### 3. 📝 **Exam Ally**
Transform study materials into personalized practice exams. Upload PDFs and generate targeted questions with adaptive difficulty.

### 4. 🎤 **Interview Mentor**
Real-time speech-to-speech interview practice using OpenAI's Realtime API. Upload resume and job descriptions for tailored experiences.

### 5. ⚖️ **AI Comparisons**
Compare AI model responses side-by-side with minimal system prompts. Native support for images, PDFs, and documents (OpenAI and Anthropic models).

---

## 🚀 Quick Start (Development)

### Prerequisites
- **Python 3.11.1-3.12.9**
- **Conda** (recommended for environment management)
- **API Keys** (contact administrators for educational access)

### Installation

1. **Clone and setup**
   ```bash
   git clone https://github.com/fmegahed/chatisa.git
   cd chatisa
   conda create --name chatisa --file requirements.txt
   conda activate chatisa
   ```

2. **Configure API keys**
   Create `.env` file:
   ```bash
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   COHERE_API_KEY=your_cohere_key_here
   GROQ_API_KEY=your_groq_key_here
   ```

3. **Launch the application**
   ```bash
   streamlit run chatgpt.py
   ```

4. **Access ChatISA**
   Open browser to `http://localhost:8501`

---

## 🏢 Production Deployment (Windows Server)

### Server Setup

1. **Environment Setup**
   ```bash
   # Create conda environment in target directory
   conda create --name chatisa --file requirements.txt
   conda activate chatisa
   
   # Navigate to deployment directory
   cd C:\Users\webapp\.conda\chat_isa
   ```

2. **SSL Certificate Configuration**
   Place SSL certificates in `ssl/` directory:
   - `ssl/chatisa.pem` (certificate file)
   - `ssl/chatisapriv.key` (private key file)

3. **Create Startup Script**
   Create `chatisa.bat`:
   ```batch
   @echo off
   cd /d C:\Users\webapp\.conda\chat_isa
   C:\Users\webapp\.conda\envs\chatisa\python.exe -m streamlit run chatgpt.py --server.address chatisa.fsb.miamioh.edu --server.port 443 --server.sslCertFile C:\Users\webapp\.conda\chat_isa\ssl\chatisa.pem --server.sslKeyFile C:\Users\webapp\.conda\chat_isa\ssl\chatisapriv.key
   pause
   ```

### Windows Task Scheduler Configuration

1. **Create Task**
   - Open Task Scheduler
   - Create Basic Task: "ChatIsa Startup"
   - Trigger: "When the computer starts"
   - Action: Start the `chatisa.bat` file

2. **Task Properties**
   ```xml
   <Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
     <RegistrationInfo>
       <URI>\ChatIsa Startup</URI>
     </RegistrationInfo>
     <Triggers>
       <BootTrigger>
         <Enabled>true</Enabled>
       </BootTrigger>
     </Triggers>
     <Settings>
       <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
       <DisallowStartIfOnBatteries>true</DisallowStartIfOnBatteries>
       <StopIfGoingOnBatteries>true</StopIfGoingOnBatteries>
       <AllowHardTerminate>true</AllowHardTerminate>
       <StartWhenAvailable>false</StartWhenAvailable>
     </Settings>
   </Task>
   ```

3. **Access Application**
   - Production URL: `https://chatisa.fsb.miamioh.edu`
   - Auto-starts on server boot
   - Runs on port 443 with SSL

---

## 🏗️ Technical Architecture

### Single-Port Implementation
Version 4.1 eliminates separate FastAPI server by embedding OpenAI Realtime API token generation directly in the Streamlit application:

- **Server-side token minting**: `mint_realtime_client_secret()` in `chatgpt.py`
- **WebRTC frontend**: Embedded HTML/JavaScript for speech-to-speech
- **Client token flow**: Streamlit generates ephemeral tokens for browser WebRTC connection

### Supported Models

| Provider | Model | Vision | PDF | Specialty |
|----------|-------|---------|-----|-----------|
| **OpenAI** | GPT-5 Chat | ✅ | ✅ | General purpose |
| | GPT-5 Mini | ✅ | ✅ | Cost-effective |
| | GPT-4o Realtime | ❌ | ❌ | Speech-to-speech |
| **Anthropic** | Claude Sonnet 4 | ✅ | ✅ | Code & analysis |
| **Cohere** | Command A | ❌ | ❌ | Business writing |
| **Groq** | Llama 3.3 70B | ❌ | ❌ | High performance |
| | Llama 3.1 8B | ❌ | ❌ | Ultra-fast |

### Project Structure

```
chatisa/
├── chatgpt.py              # Main application & token minting
├── config.py               # Centralized configuration
├── pages/                  # Five specialized modules
│   ├── 01_coding_companion.py
│   ├── 02_project_coach.py
│   ├── 03_exam_ally.py
│   ├── 04_interview_mentor.py
│   └── 05_ai_comparisons.py
├── lib/                    # Shared utilities
│   ├── chatgeneration.py   # LLM integration
│   ├── chatpdf.py         # PDF processing
│   ├── sidebar.py         # Common UI components
│   └── speech.py          # Speech functionality
└── ssl/                   # SSL certificates (production)
```

---

## 📚 Usage Guidelines

### For Students
- **Educational Use**: Designed for learning and academic support
- **Get Approval**: Check with instructors before using for coursework
- **Think Critically**: Use AI as a learning tool, not a replacement
- **Academic Integrity**: Follow institutional AI usage policies

### For Educators
- **Set Guidelines**: Establish clear AI usage policies
- **Review Usage**: Monitor student interactions through exports
- **Foster Discussion**: Encourage conversations about AI assistance

---

## 🏢 Institutional Support

**Maintained by Miami University - Farmer School of Business**

- **[Fadel Megahed](https://miamioh.edu/fsb/directory/?up=/directory/megahefm)** - Raymond E. Glos Professor, Information Systems & Analytics
- **[Joshua Ferris](https://miamioh.edu/fsb/directory/?up=/directory/ferrisj2)** - Assistant Professor, Information Systems & Analytics

**Funding Support:**
- U.S. Bank covering API costs for educational use
- Miami University providing server infrastructure and support  
- The Raymond E. Glos Professorship, which provided Fadel time and monetary support to continue to update and mantain this chatbot

---

## 📄 Citation

```bibtex
@misc{megahed2025chatisa,
      title={ChatISA: A Prompt-Engineered, In-House Multi-Modal Generative AI Chatbot for Information Systems Education}, 
      author={Fadel M. Megahed and Ying-Ju Chen and Joshua A. Ferris and Cameron Resatar and Kaitlyn Ross and Younghwa Lee and L. Allison Jones-Farmer},
      year={2025},
      eprint={2407.15010},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2407.15010}, 
}
```

---

## 📞 Support

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/fmegahed/chatisa/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/fmegahed/chatisa/discussions)
- **📧 Institution Setup**: Contact maintainers directly

---

## 📄 License

MIT License - Free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

**Note**: While the software is free, API usage requires valid keys. Educational use is sponsored; commercial users need their own API keys.

---

<div align="center">

**Built with ❤️ for educational excellence**

*ChatISA v4.1 - Making AI accessible, responsible, and educational*

**[⭐ Star us on GitHub](https://github.com/fmegahed/chatisa)**

</div>