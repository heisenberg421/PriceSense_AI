from __future__ import annotations
import os
from pathlib import Path

__all__ = ["get_openai_api_key", "get_embedding_model"]


def _load_dotenv(dotenv_path: Path | str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _get_streamlit_secret(key: str) -> str:
    """
    Try to read a value from st.secrets (Streamlit Community Cloud).
    Returns empty string if streamlit is not available or key not set.
    Safe to call outside a Streamlit context.

    Uses key-in-secrets check before access to avoid FileNotFoundError
    (no secrets file) and KeyError (key not present) both cleanly.
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key]).strip()
        return ""
    except Exception:
        return ""


def get_openai_api_key() -> str:
    """
    Return the OpenAI API key.
    Priority: st.secrets → environment variable → .env file
    """
    secret = _get_streamlit_secret("OPENAI_API_KEY")
    if secret:
        return secret
    _load_dotenv()
    return os.environ.get("OPENAI_API_KEY", "").strip()


def get_embedding_model() -> str:
    """
    Return the HuggingFace embedding model name.
    Priority: st.secrets → environment variable → .env file → default
    """
    secret = _get_streamlit_secret("EMBEDDING_MODEL")
    if secret:
        return secret
    _load_dotenv()
    return os.environ.get("EMBEDDING_MODEL", "multi-qa-mpnet-base-dot-v1").strip()