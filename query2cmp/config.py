"""Configuration loaded from environment and .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from current directory or project root
load_dotenv()
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)


def get_config() -> dict:
    """Return configuration from environment variables."""
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_base_url": base_url or "https://api.openai.com/v1",
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "mcp_server": os.getenv("MCP_SERVER_URL", ""),
        "mcp_language": os.getenv("MCP_LANGUAGE", "").strip(),
        "llm_backend": os.getenv("LLM_BACKEND", "openai").lower(),
        "verbose": os.getenv("QUERY2MCP_VERBOSE", "false").lower() in ("true", "1", "yes"),
    }
