import hashlib
import html
import os
from typing import Any, Dict, List

import streamlit as st


def safe_filename(filename: str) -> str:
    return os.path.basename(filename)


def validate_pdf_file(file: Any) -> bool:
    if hasattr(file, "name"):
        size_mb = getattr(file, "size", None)
        if size_mb is not None:
            return size_mb / (1024 * 1024) <= int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    return True


def compute_upload_fingerprint(files: List[Any]) -> str:
    digest = hashlib.sha256()
    for uploaded_file in files:
        name = getattr(uploaded_file, "name", "unknown")
        size = getattr(uploaded_file, "size", 0)
        digest.update(name.encode("utf-8"))
        digest.update(str(size).encode("utf-8"))
        current_pos = uploaded_file.tell() if hasattr(uploaded_file, "tell") else 0
        file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        digest.update(file_bytes)
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(current_pos)
    return digest.hexdigest()


def append_chat_history(role: str, content: str) -> None:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    st.session_state["chat_history"].append({"role": role, "content": content})


def reset_session() -> None:
    for key in [
        "chat_history",
        "source_chunks",
        "documents",
        "indexed_files",
        "upload_fingerprint",
        "index_status",
        "last_error",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def init_session_defaults() -> None:
    defaults = {
        "chat_history": [],
        "source_chunks": [],
        "documents": [],
        "indexed_files": [],
        "upload_fingerprint": "",
        "index_status": "No documents indexed yet.",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def escape_html(value: object) -> str:
    return html.escape(str(value), quote=True)
