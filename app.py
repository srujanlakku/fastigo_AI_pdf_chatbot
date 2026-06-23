from src.cloud_bootstrap import apply_cloud_runtime_patches

apply_cloud_runtime_patches()

import hashlib

import streamlit as st

from src.chatbot import DocumentChatbot
from src.chunking import create_chunks
from src.config import validate_api_settings
from src.hybrid_retriever import HybridRetriever
from src.pdf_processor import extract_pdf_text
from src.utils import (
    append_chat_history,
    compute_upload_fingerprint,
    escape_html,
    init_session_defaults,
    reset_session,
    validate_pdf_file,
)
from src.vector_store import clear_collection, get_collection, persist_chunks


CUSTOM_CSS = """
<style>
    .main-header {
        background: linear-gradient(120deg, #0f172a 0%, #1d4ed8 100%);
        color: white;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .status-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
    }
    .chat-user {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }
    .chat-assistant {
        background: #f8fafc;
        border-left: 4px solid #0f766e;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }
    .source-card {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .source-meta {
        color: #475569;
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
    }
    .source-excerpt {
        color: #0f172a;
        font-size: 0.95rem;
        line-height: 1.45;
    }
</style>
"""


def _hash_file(uploaded_file) -> str:
    digest = hashlib.sha256()
    name = getattr(uploaded_file, "name", "unknown")
    digest.update(name.encode("utf-8"))
    file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    digest.update(file_bytes)
    return digest.hexdigest()


def _render_sidebar() -> None:
    st.sidebar.markdown("### Session")
    if st.sidebar.button("Reset session", use_container_width=True):
        clear_collection()
        reset_session()
        st.rerun()

    st.sidebar.markdown("### Indexed PDFs")
    indexed_files = st.session_state.get("indexed_files", [])
    if indexed_files:
        for file_name in indexed_files:
            st.sidebar.write(f"- {file_name}")
    else:
        st.sidebar.caption("No documents indexed yet.")

    st.sidebar.markdown("### Status")
    st.sidebar.info(st.session_state.get("index_status", "No documents indexed yet."))


def _index_uploaded_files(file_list) -> None:
    fingerprint = compute_upload_fingerprint(file_list)
    collection = get_collection()
    if (
        fingerprint == st.session_state.get("upload_fingerprint")
        and st.session_state.get("source_chunks")
        and collection.count() > 0
    ):
        st.session_state["index_status"] = "Documents already indexed. Skipping duplicate processing."
        return

    progress = st.progress(0, text="Preparing upload...")
    all_documents = []
    indexed_names = []
    total_files = len(file_list)

    for index, uploaded_file in enumerate(file_list, start=1):
        file_name = getattr(uploaded_file, "name", "unknown.pdf")
        progress.progress(int((index - 1) / total_files * 60), text=f"Extracting text from {file_name}...")
        try:
            file_hash = _hash_file(uploaded_file)
            file_documents = extract_pdf_text(uploaded_file)
            for document in file_documents:
                document["file_hash"] = file_hash
                document["metadata"]["file_hash"] = file_hash
            all_documents.extend(file_documents)
            indexed_names.append(file_name)
        except ValueError as exc:
            st.error(f"Failed to process {file_name}: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error while processing {file_name}: {exc}")

    if not all_documents:
        st.session_state["index_status"] = "Indexing failed. No extractable text found."
        progress.empty()
        return

    progress.progress(70, text="Chunking documents...")
    chunks = create_chunks(all_documents, file_hash=fingerprint)
    if not chunks:
        st.error("No valid text chunks were generated from the uploaded files.")
        st.session_state["index_status"] = "Indexing failed during chunking."
        progress.empty()
        return

    progress.progress(85, text="Generating embeddings and storing vectors...")
    try:
        persist_chunks(chunks)
    except RuntimeError as exc:
        st.error(str(exc))
        st.session_state["index_status"] = "Indexing failed while writing to ChromaDB."
        progress.empty()
        return
    except Exception as exc:
        st.error(f"Embedding or vector storage failed: {exc}")
        st.session_state["index_status"] = "Indexing failed during embedding generation."
        progress.empty()
        return

    st.session_state["source_chunks"] = chunks
    st.session_state["documents"] = all_documents
    st.session_state["indexed_files"] = indexed_names
    st.session_state["upload_fingerprint"] = fingerprint
    st.session_state["index_status"] = f"Indexed {len(indexed_names)} PDF(s) with {len(chunks)} chunks."
    progress.progress(100, text="Indexing complete.")
    st.success(st.session_state["index_status"])


def _render_source_cards(sources) -> None:
    st.markdown("### Sources")
    if not sources:
        st.caption("No source citations available.")
        return
    for source in sources:
        file_name = escape_html(source["file_name"])
        page_number = escape_html(source["page_number"])
        excerpt = escape_html(source["excerpt"])
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-meta"><strong>{file_name}</strong> · Page {page_number}</div>
                <div class="source-excerpt">"{excerpt}"</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_chat_history() -> None:
    history = st.session_state.get("chat_history", [])
    if not history:
        return
    st.markdown("### Conversation")
    for message in history:
        content = escape_html(message["content"])
        if message["role"] == "user":
            st.markdown(f'<div class="chat-user"><strong>You</strong><br>{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="chat-assistant"><strong>Assistant</strong><br>{content}</div>',
                unsafe_allow_html=True,
            )


def main() -> None:
    st.set_page_config(page_title="Fastigo AI PDF Chatbot", page_icon="📄", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_defaults()
    _render_sidebar()

    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin:0;">Fastigo AI PDF Chatbot</h1>
            <p style="margin:0.35rem 0 0 0;">Upload PDFs, ask questions, and get grounded answers with citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        validate_api_settings()
    except EnvironmentError as exc:
        st.error(str(exc))
        st.stop()

    st.markdown('<div class="status-card">Upload one or more PDF files to begin. Indexed documents stay available during this session.</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files for indexing.",
    )

    if uploaded_files:
        file_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
        invalid_files = [file for file in file_list if not validate_pdf_file(file)]
        if invalid_files:
            st.error("One or more uploaded PDFs exceed the allowed file size.")
        else:
            with st.spinner("Processing uploaded PDFs..."):
                _index_uploaded_files(file_list)

    _render_chat_history()

    with st.form("chat_form", clear_on_submit=True):
        query = st.text_input("Ask a question about the uploaded PDFs")
        submitted = st.form_submit_button("Ask", use_container_width=True)

    if submitted and query:
        if not st.session_state.get("source_chunks"):
            st.warning("Please upload and index PDF files before asking a question.")
        else:
            with st.spinner("Searching documents..."):
                try:
                    retriever = HybridRetriever()
                    context = retriever.hybrid_search(query)
                except Exception as exc:
                    st.error(f"Retrieval failed: {exc}")
                    context = []

            if not context:
                st.info("No relevant content was found in the uploaded documents.")
            else:
                chatbot = DocumentChatbot()
                history = st.session_state.get("chat_history", [])
                st.markdown("### Answer")
                answer_placeholder = st.empty()
                answer_text = ""
                try:
                    for token in chatbot.stream_answer(query, context, conversation_history=history):
                        answer_text += token
                        answer_placeholder.markdown(answer_text)
                    append_chat_history("user", query)
                    append_chat_history("assistant", answer_text)
                    _render_source_cards(chatbot.format_sources(context))
                except RuntimeError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Chat generation failed: {exc}")


if __name__ == "__main__":
    main()
