
# GPT4All Conversational Retrieval — Streamlit + LlamaCpp + Chroma

A professional, on-prem Retrieval-Augmented Generation (RAG) demo built with Streamlit, LangChain, LlamaCpp (local), and Chroma vectorstore.
The app allows you to upload a PDF, build a vectorstore from document chunks, and interactively ask questions with source citations.

## Files in this repo
- `app_chat_improved.py` — Streamlit application (main)
- `download_model.py` — Download GGUF model from Hugging Face (edit token & path at top)
- `requirements.txt` — Python dependencies
- `FLOWCHART.md` — Mermaid flowchart + short explanation
- `IMAGES.md` — Detailed explanation of included images
- `images/flowchart.png` — Simplified professional flowchart
- `LICENSE`, `.gitignore`

## Quick start (on-prem)
1. Create a virtualenv and activate it:
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the model using `download_model.py` (edit variables at top) or use your preferred method:
```bash
python download_model.py
```

4. Run the Streamlit app:
```bash
streamlit run app_chat_improved.py
```

## Image
![Flowchart](images/flowchart.png)

## Notes
- **Do not** commit large model files to GitHub. Use `.gitignore` (included) to exclude `*.gguf` and `models/`.
- The app uses in-memory Chroma by default — modify the code if you need persistent vectorstores.
- Ensure `llama-cpp-python` or your chosen runtime can load the GGUF model format.

---
