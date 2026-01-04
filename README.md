# 📄 PDFChatBot — Intelligent PDF Question Answering System

PDFChatBot is a Python-powered application that transforms any PDF into an interactive chatbot you can ask questions. It’s designed to be easy to use by students, researchers, and non-technical users, while also being powerful enough to support advanced PDF understanding workflows like OCR and semantic analysis. 
GitHub

# 🌟 Key Features
# ✅ Supports Both Text & Image PDFs

Text-based PDFs are fully supported — the system extracts text directly from the file for fast indexing and querying. 
GitHub

Scanned or image-based PDFs can be processed via OCR (Optical Character Recognition) to extract text from scanned pages and images before answering. 
GitHub

# 👉 This means even study notes, old scanned research papers, or textbook screenshots work with the chatbot.

# 🤖 AI-Powered Question Answering

Uses advanced NLP/Language Model techniques to understand user questions in natural language and generate accurate responses based on the content extracted from the PDF. 
GitHub

Works like a conversational assistant — ask questions like “What is the main conclusion on page 12?” or “Summarize this section for me.”

# 💬 Simple & Beautiful UI

Built for non-technical users, the interface is intuitive and clean (using tools like Streamlit / Tkinter in the backend). 
GitHub

No coding knowledge is required to upload a PDF and start asking questions.

# 📌 Great for users who just want answers from their PDF — not complicated setup or technical configs.

# 📚 Optimized for Students & Researchers

Perfect for academic workflows — quickly extract explanations, key points, definitions, formulas, or summaries from long lecture notes and research papers. 
GitHub

Helps with efficient studying, literature review, exam preparation, project work, and research analysis.

# 🔍 Easy Deployment & Use

Set up with just a few commands — install dependencies and run using a simple Python command. 
GitHub

Optionally supports OCR enabled and non-OCR versions, so users can choose based on their needs (scanned vs regular text PDFs). 
GitHub

# 🧠 Flexible Architecture

Built with modular components like:

PDF parsing & OCR

Semantic text querying

Chat interface

Makes it easy to extend — e.g., support summaries, multi-PDF upload, citation tracking, or integration with vector databases.


## Installation
First Install the dependencies.

<code>
  curl -fsSL https://ollama.com/install.sh | sh
  sudo systemctl enable ollama.service
  sudo systemctl start ollama.service
  ollama pull falcon:7b
  sudo apt install tesseract-ocr-eng tesseract-ocr-hin
  pip install -r requirements.txt
</code>

## Run
This one is OCR Enabled.

<code>
  streamlit run pdfchat-v2.py
</code>

This one is Without OCR, Only supports Text containing PDFs, not Images.
<code>
  streamlit run pdfchat-v1.py
</code>


# 📌 Why It’s Useful

✔ Saves hours of manual reading
✔ Makes dense academic content searchable
✔ Designed for students, educators & researchers
✔ Works on diverse PDF types and formats
✔ Minimal setup, no advanced technical skill required
