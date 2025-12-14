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
