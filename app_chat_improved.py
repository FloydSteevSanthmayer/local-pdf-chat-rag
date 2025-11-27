import os
import tempfile
import logging
from typing import List, Tuple, Any

import streamlit as st
import torch

# LangChain & community imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain

# ---------- USER-CONFIG (top of file) ----------
# Keep the model path at top-level so it's easy to find / edit:
MODEL_PATH = r"C:/Users/LAP14/Downloads/mistral-7b-openorca.Q4_0.gguf"
# ------------------------------------------------

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------
# Cached / helper resources
# -------------------------
@st.cache_resource(show_spinner=False)
def get_device_str() -> str:
    """Return device string for embeddings (cuda if available else cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner=True)
def load_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return a HuggingFaceEmbeddings instance (cached)."""
    device_str = get_device_str()
    emb = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device_str})
    logger.info("Loaded embeddings model %s on %s", model_name, device_str)
    return emb


@st.cache_resource(show_spinner=True)
def load_llm_cached(model_path: str, temperature: float = 0.3, top_p: float = 1.0, n_ctx: int = 2048, streaming: bool = False):
    """Load and cache LlamaCpp LLM. Raise if file missing."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LLM model not found at: {model_path}")
    llm = LlamaCpp(model_path=model_path, temperature=temperature, top_p=top_p, n_ctx=n_ctx, streaming=streaming, verbose=False)
    logger.info("Loaded LlamaCpp model from %s", model_path)
    return llm


@st.cache_resource(show_spinner=True)
def build_vectorstore_from_pdf_cached(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200,
                                      embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Build & cache an in-memory Chroma vectorstore from a PDF.
    If you want persistent storage, create a chromadb.Client with persist_directory and pass to Chroma.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    embeddings = load_embeddings(embeddings_model_name)

    # Build in-memory Chromadb-backed store via LangChain wrapper
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)

    logger.info("Built vectorstore (chunks=%d)", len(chunks))
    return vectordb


# -------------------------
# Session-state helpers
# -------------------------
def init_session_state():
    if "vectorstore_ready" not in st.session_state:
        st.session_state.vectorstore_ready = False
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "llm_loaded" not in st.session_state:
        st.session_state.llm_loaded = False
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "chat_history" not in st.session_state:
        # store as list of dicts: {"user": "...", "assistant": "...", "sources": [...]}
        st.session_state.chat_history: List[dict] = []


# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(page_title="Conversational Retrieval — LlamaCpp + Chroma", layout="wide")
    st.title("Conversational Retrieval — Local LLM + Chroma")

    init_session_state()

    # Sidebar: config
    st.sidebar.header("Configuration")
    st.sidebar.markdown("Model path is pre-set at top of file; override here if needed.")
    model_path_input = st.sidebar.text_input("Local model path (GGUF / llama.cpp)", value=MODEL_PATH)
    embeddings_model_name = st.sidebar.text_input("Embeddings model", value="sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = st.sidebar.number_input("Chunk size (chars)", min_value=256, max_value=20000, value=1000, step=128)
    chunk_overlap = st.sidebar.number_input("Chunk overlap (chars)", min_value=0, max_value=2000, value=200, step=50)
    retriever_k = st.sidebar.number_input("Retriever k (top results)", min_value=1, max_value=20, value=4, step=1)
    st.sidebar.markdown("---")
    st.sidebar.write("Embeddings device:", get_device_str())

    # Upload PDF (required)
    st.subheader("Upload PDF (required)")
    uploaded_file = st.file_uploader("Choose a PDF file to upload", type=["pdf"])
    if uploaded_file is None:
        st.info("Please upload a PDF file to build the vectorstore and enable chat.")
        st.stop()

    # Save upload to temp path
    tmp_dir = tempfile.gettempdir()
    saved_name = f"uploaded_{uploaded_file.name}"
    saved_path = os.path.join(tmp_dir, saved_name)
    try:
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded file saved to: {saved_path}")
    except Exception as e:
        st.error(f"Failed to save uploaded file: {e}")
        st.stop()

    # Build vectorstore if not ready or if settings changed
    recreate_vectorstore = st.sidebar.button("Rebuild vectorstore")
    # We provide a simple key check: if not ready or user requested rebuild, (re)build
    if (not st.session_state.vectorstore_ready) or recreate_vectorstore:
        try:
            with st.spinner("Building vectorstore from PDF (this can take a while)..."):
                vectordb = build_vectorstore_from_pdf_cached(saved_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, embeddings_model_name=embeddings_model_name)
                st.session_state.vectorstore = vectordb
                st.session_state.vectorstore_ready = True
                st.success("Vectorstore ready.")
        except Exception as e:
            st.error(f"Failed to build vectorstore: {e}")
            st.stop()
    else:
        st.info("Using cached vectorstore for this session.")

    # Load LLM if not loaded
    load_model_now = st.sidebar.button("(Re)Load LLM")
    if (not st.session_state.llm_loaded) or load_model_now:
        try:
            with st.spinner("Loading local LLM (may take long)..."):
                llm = load_llm_cached(model_path=model_path_input, temperature=0.3, top_p=1.0, n_ctx=2048, streaming=False)
                st.session_state.llm = llm
                st.session_state.llm_loaded = True
                st.success("LLM loaded.")
        except Exception as e:
            st.error(f"Failed to load LLM: {e}")
            st.stop()
    else:
        st.info("LLM is loaded in session.")

    # Create chain if not present
    if st.session_state.chain is None:
        try:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": retriever_k})
            chain = ConversationalRetrievalChain.from_llm(llm=st.session_state.llm, retriever=retriever, return_source_documents=True)
            st.session_state.chain = chain
            logger.info("ConversationalRetrievalChain created.")
        except Exception as e:
            st.error(f"Failed to create retrieval chain: {e}")
            st.stop()

    # Chat UI
    st.subheader("Chat with your document")
    # Display chat history
    if st.session_state.chat_history:
        for turn in st.session_state.chat_history:
            st.markdown(f"**You:** {turn['user']}")
            st.markdown(f"**Assistant:** {turn['assistant']}")
            # optional: display sources
            sources = turn.get("sources", [])
            if sources:
                st.markdown("**Sources:**")
                for i, s in enumerate(sources, start=1):
                    # each source may be a Document with metadata
                    if hasattr(s, "metadata"):
                        meta = getattr(s, "metadata", {}) or {}
                        src = meta.get("source", "uploaded_pdf")
                        page = meta.get("page", "n/a")
                    else:
                        src, page = "uploaded_pdf", "n/a"
                    st.markdown(f"- Source {i}: {src} (page: {page})")
                    snippet = getattr(s, "page_content", str(s))[:800]
                    st.write(snippet + ("..." if len(snippet) >= 800 else ""))
            st.write("---")

    # Input area
    user_input = st.text_input("Enter your question about the uploaded PDF:", key="user_input")
    ask_button = st.button("Ask")

    if ask_button:
        if not user_input or not user_input.strip():
            st.warning("Please type a question before clicking Ask.")
        else:
            with st.spinner("Running retrieval and generation..."):
                try:
                    # Standard call shape for ConversationalRetrievalChain: pass dict with question + chat_history (many versions accept this)
                    call_result = st.session_state.chain({"question": user_input, "chat_history": [(h["user"], h["assistant"]) for h in st.session_state.chat_history]})
                except Exception as e:
                    # Fallback: try passing only the question
                    try:
                        call_result = st.session_state.chain({"question": user_input})
                    except Exception as e2:
                        st.error(f"Chain call failed: {e} / {e2}")
                        call_result = {"answer": "Error: chain call failed", "source_documents": []}

                # Extract answer & sources in a version-robust way
                answer = call_result.get("answer") or call_result.get("result") or call_result.get("output_text") or str(call_result)
                src_docs = call_result.get("source_documents") or call_result.get("source_docs") or call_result.get("sources") or []

                # Append to chat history for display
                st.session_state.chat_history.append({"user": user_input, "assistant": answer, "sources": src_docs})

                # Clear the input box
                st.session_state.user_input = ""

                # Show immediate answer (also visible in history above)
                st.success("Answer:")
                st.write(answer)

                if src_docs:
                    st.markdown("**Top sources (best-effort):**")
                    for i, doc in enumerate(src_docs, start=1):
                        meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
                        src_name = meta.get("source", "uploaded_pdf")
                        page = meta.get("page", "n/a")
                        st.markdown(f"- Source {i}: {src_name} (page: {page})")
                        snippet = getattr(doc, "page_content", str(doc))[:900]
                        st.write(snippet + ("..." if len(snippet) >= 900 else ""))
                else:
                    st.info("No source documents were returned by the chain for this query.")

    # Footer controls
    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.success("Conversation cleared.")

    if st.button("Reset cached resources (embeddings/LLM/vectorstore)"):
        # Clear cached resources. st.cache_resource cache cannot be selectively cleared from here,
        # but we can reset session state so rebuild occurs next time.
        st.session_state.vectorstore_ready = False
        st.session_state.vectorstore = None
        st.session_state.llm = None
        st.session_state.llm_loaded = False
        st.session_state.chain = None
        st.session_state.chat_history = []
        st.success("Session resources reset. Reload page or click Rebuild / Load LLM to recreate resources.")

    st.info("Tip: If LLM load or embedding creation fails, check that model file exists at the top-of-file MODEL_PATH and that dependencies are installed.")


if __name__ == "__main__":
    main()
