# ChatISA v4.0.0

<div align="center">

![ChatISA Logo](https://github.com/fmegahed/chatisa/assets/22730186/3a0c2839-3384-428c-9aa5-e0cf95ba1296)

**🤖 Educational AI Assistant with Multiple LLM Support**

[![Version](https://img.shields.io/badge/version-4.0.0-blue.svg)](https://github.com/fmegahed/chatisa)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*Making AI accessible for Miami University students*

</div>

---

## 📖 Overview

ChatISA is a comprehensive educational AI assistant designed for business analytics and computer science students. It provides free access to multiple leading AI models through five specialized learning interfaces, with costs sponsored by industry partners to ensure equitable access to cutting-edge AI technology.

### 🎯 Key Features

- **🔄 Multi-Model Support**: Access 7+ leading AI models (OpenAI, Anthropic, Cohere, Groq)
- **📚 Five Specialized Modules**: Tailored interfaces for different learning needs
- **💰 Cost Coverage**: Free access with sponsored API costs for educational use
- **📄 Professional Reports**: Comprehensive PDF exports with usage analytics
- **🎤 Voice Integration**: Speech-to-speech capabilities for interview practice
- **🔍 AI Model Comparison**: Side-by-side experimental comparison tool
- **🎓 Educational Focus**: Purpose-built for responsible academic use

---

## 🛠️ Five Learning Modules

### 1. 💻 **Coding Companion**
*Your programming learning partner*

Get personalized programming help with educational context. Features code generation, debugging assistance, and concept explanations optimized for learning R, Python, and data analysis techniques.

- **Specialty**: Programming education and code explanation
- **Best For**: Learning syntax, debugging, understanding algorithms
- **Default Model**: Claude Sonnet 4 (excellent for code analysis)

### 2. 🎯 **Project Coach**
*Comprehensive project guidance with specialized coaching roles*

Navigate team projects with AI coaching across multiple specialized roles including project scoping, risk analysis, team structuring, and reflection guidance.

- **Coaching Roles**: Project Manager, Risk Analyst, Team Advisor, Devil's Advocate
- **Features**: Interactive worksheets, structured guidance, decision frameworks
- **Best For**: Group projects, business case studies, project planning

### 3. 📝 **Exam Ally**
*AI-powered exam preparation from your materials*

Transform your study materials into personalized practice exams. Upload PDFs and generate targeted questions across multiple formats with adaptive difficulty.

- **Question Types**: Multiple choice, short answer, code analysis, data interpretation
- **Features**: PDF processing, adaptive questioning, performance feedback
- **Best For**: Course review, exam preparation, knowledge assessment

### 4. 🎤 **Interview Mentor**
*Professional interview practice with speech-to-speech technology*

Practice interviews with realistic AI conversation using OpenAI's Realtime API. Upload your resume and job descriptions for tailored interview experiences with natural voice interaction.

- **Technology**: Real-time speech-to-speech conversation
- **Features**: Resume analysis, company-specific questions, natural dialogue flow
- **Best For**: Interview preparation, communication skills, professional practice

### 5. ⚖️ **AI Comparisons** ⭐ *NEW in v4.0.0*
*Experimental side-by-side model comparison*

Compare how different AI models respond to the same questions with support for images, PDFs, and documents. Perfect for understanding AI capabilities and differences.

- **Vision Support**: GPT-5 and Claude can analyze images directly
- **Document Support**: Direct PDF reading capabilities for capable models  
- **File Processing**: Automatic handling of various file types by extension
- **Best For**: Understanding AI differences, research, experimental learning

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
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
   ```bash
   # Create .env file with your API keys
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
   - Open your browser to `http://localhost:8502`
   - Start exploring with AI! 🎉

---

## 🏗️ Architecture

### Supported Models

| Provider | Model | Specialty | Vision | PDF |
|----------|-------|-----------|---------|-----|
| **OpenAI** | GPT-5 Chat | General purpose | ✅ | ✅ |
| | GPT-5 Mini | Cost-effective | ✅ | ✅ |
| | GPT-4o Realtime | Speech-to-speech | ❌ | ❌ |
| **Anthropic** | Claude Sonnet 4 | Code & analysis | ✅ | ✅ |
| **Cohere** | Command A | Business writing | ❌ | ❌ |
| **Groq** | Llama 3.3 70B | High performance | ❌ | ❌ |
| | Llama 3.1 8B | Ultra-fast | ❌ | ❌ |

### Core Structure

```
chatisa/
├── chatgpt.py              # Main application entry point
├── config.py               # Centralized configuration
├── pages/                  # Five specialized modules
│   ├── 01_coding_companion.py
│   ├── 02_project_coach.py
│   ├── 03_exam_ally.py
│   ├── 04_interview_mentor.py
│   └── 05_ai_comparisons.py
├── lib/                    # Shared utilities
├── assets/                 # Static resources
└── realtime_server.py      # Speech server
```

---

## 📚 Usage Guidelines

### For Students

- **🎓 Educational Use**: Designed for learning and academic support
- **👨‍🏫 Get Approval**: Always check with instructors before using for coursework
- **🧠 Think Critically**: Use AI as a learning tool, not a replacement for understanding
- **📝 Academic Integrity**: Follow your institution's AI usage policies
- **🔍 Verify Results**: Always evaluate and verify AI-generated content

### For Educators

- **📋 Set Guidelines**: Establish clear AI usage policies for your courses
- **🎯 Align Goals**: Connect AI use with learning objectives
- **📊 Review Usage**: Use PDF exports to understand student interactions
- **🤝 Foster Discussion**: Encourage open conversations about AI assistance

---

## 🏢 Institutional Support

**Maintained by Miami University - Farmer School of Business**

- **[Fadel Megahed](https://miamioh.edu/fsb/directory/?up=/directory/megahefm)** - Raymond E. Glos Professor, Information Systems & Analytics
- **[Joshua Ferris](https://miamioh.edu/fsb/directory/?up=/directory/ferrisj2)** - Assistant Professor, Information Systems & Analytics

**Funding Support:**
- Industry partners covering API costs for educational use
- Miami University providing server infrastructure and support
- Committed to democratizing AI access for educational excellence

---

## 📄 Citation

If you use ChatISA in your research or teaching:

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

*ChatISA v4.0.0 - Making AI accessible, responsible, and educational*

**[⭐ Star us on GitHub](https://github.com/fmegahed/chatisa)**

</div>