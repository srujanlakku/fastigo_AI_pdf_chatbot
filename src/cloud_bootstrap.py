"""Runtime patches for Streamlit Cloud and other constrained Linux deployments."""

from __future__ import annotations

import os
import sys

_PATCHES_APPLIED = False


def apply_cloud_runtime_patches() -> None:
    """Apply environment and sqlite patches before ChromaDB imports."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return
    _PATCHES_APPLIED = True

    # Protobuf / OpenTelemetry compatibility for ChromaDB on Streamlit Cloud.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")

    # ChromaDB requires sqlite3 >= 3.35; Streamlit Cloud ships an older stdlib sqlite.
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass
