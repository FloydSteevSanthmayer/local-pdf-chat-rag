
# Images included

**images/flowchart.png** — Simplified app flow diagram (professional, easy-to-understand).

## Explanation of the image (flowchart.png)

- **Upload PDF (required):** Visual entry point — user provides a PDF document to analyze.
- **Save PDF to temp file:** Implementation detail — the app writes the uploaded file to a temporary location so PDF loaders can access it.
- **Is vectorstore available?:** Decision node — the app checks whether embeddings/chunks for this PDF were already computed and stored. If yes, it saves time by loading them.
- **Build vectorstore:** When absent, the app loads the PDF, splits text into chunks, generates embeddings (HuggingFace embeddings), and stores vectors in Chroma.
- **Load local LLM (MODEL_PATH):** The app loads your local Llama-compatible GGUF model using the path defined at the top of `app_chat_improved.py`.
- **Retrieval & Answer:** The retriever gets top-k most similar chunks, the LLM composes an answer, and the UI presents the answer plus short source snippets for traceability.

This image is safe to include in your README and documentation. It is provided as a PNG in the `images/` folder for convenient display on GitHub.
