"""One chat call that works with Ollama, Groq or Gemini: change three settings in .env, not the code."""
import os
from pathlib import Path

import httpx


def _load_env() -> None:
    """Read KEY=VALUE lines from .env (if there is one) without overriding real environment variables."""
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def settings() -> tuple[str, str, str]:
    _load_env()
    base = os.environ.get("LLM_BASE_URL")
    key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")

    # Fallback to Streamlit secrets when deployed on Streamlit Community Cloud
    if not key or not model:
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                base = base or st.secrets.get("LLM_BASE_URL")
                key = key or st.secrets.get("LLM_API_KEY")
                model = model or st.secrets.get("LLM_MODEL")
        except Exception:
            pass

    return (base or "http://localhost:11434/v1", key or "", model or "")


def chat(messages: list[dict], **options) -> str:
    """Send chat messages, get the reply text. `options` go straight to the API (temperature, max_tokens...)."""
    base, key, model = settings()
    if not model:
        raise RuntimeError("Set LLM_MODEL in .env (see .env.example).")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = httpx.post(f"{base.rstrip('/')}/chat/completions", headers=headers, timeout=120,
                   json={"model": model, "messages": messages, **options})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
