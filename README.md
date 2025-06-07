# 🦙📄 Doc Chat with LlamaIndex

A lightweight Gradio-based web app that lets you **chat with your documents in Persian** using:

- 🧠 LlamaIndex
- 🚀 Groq LLaMA3-70B
- 🔍 HuggingFace Embeddings

Just upload a file (PDF, DOCX, Excel, image, etc.), ask your question, and get a rich, Markdown-formatted Farsi answer instantly!

---

## 🚀 Features

- ✅ Supports many file formats: `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.txt`, `.csv`, `.html`, images (`.jpg`, `.png`, etc.)
- 💬 Smart Persian Q&A powered by Groq's LLaMA3-70B
- 🧠 Uses HuggingFace's `all-mpnet-base-v2` for deep semantic understanding
- ⚡ Caches file embeddings for ultra-fast reloading
- 📑 Answers come in structured, beautiful Markdown (great for copy-paste)
- 🖼️ Clean and responsive UI via Gradio

---

## ⚙️ Setup & Run (All-in-One)

```bash
# Clone the project
git clone https://github.com/yourusername/doc-chat-llamaindex.git
cd doc-chat-llamaindex

# Create virtual environment
python -m venv venv

# Activate it (Linux/macOS)
source venv/bin/activate

# Activate it (Windows CMD)
# venv\Scripts\activate

# Install all dependencies
pip install gradio llama-index llama-parse huggingface-hub

# Set your API keys (🔑 Replace with your actual keys!)

# Linux/macOS
export LLAMA_CLOUD_API_KEY=your_llama_key
export GROQ_API_KEY=your_groq_key

# Windows CMD
set LLAMA_CLOUD_API_KEY=your_llama_key
set GROQ_API_KEY=your_groq_key

# Launch the app!
python app.py

```

--

# 🧠 How It Works
Upload a document — it’s parsed via LlamaParse

The text is split into smart chunks for better context

Each chunk is vectorized using HuggingFace’s transformer

LlamaIndex builds a vector index (cached for reuse)

Your question is turned into a structured Persian prompt

Groq's LLaMA3-70B generates a streamed, detailed response in Markdown

--
# 📄 License

MIT License — free to use, modify, and distribute
