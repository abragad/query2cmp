"""LLM backends for query2cmp."""

from .base import ToolCallResult, LLMBackend
from .openai_backend import OpenAIBackend

__all__ = [
    "ToolCallResult",
    "LLMBackend",
    "OpenAIBackend",
]

try:
    from .apple_backend import AppleFMBackend

    __all__.append("AppleFMBackend")
except ImportError:
    AppleFMBackend = None
