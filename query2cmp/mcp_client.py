"""MCP client supporting stdio (local) and streamable HTTP (remote)."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client


def _normalize_server_spec(server_spec: str) -> str:
    """Strip trailing parenthetical comments (e.g. ' (5 tools)') from URLs."""
    s = server_spec.strip()
    if " (" in s and (s.startswith("http://") or s.startswith("https://")):
        s = s.split(" (", 1)[0].rstrip()
    return s


def _is_http_url(server_spec: str) -> bool:
    """Return True if server_spec looks like an HTTP(S) URL."""
    s = server_spec.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def _is_local_script(server_spec: str) -> bool:
    """Return True if server_spec looks like a local script path."""
    p = Path(server_spec)
    return p.suffix in (".py", ".js") and (p.exists() or server_spec.endswith((".py", ".js")))


@asynccontextmanager
async def mcp_session(server_spec: str) -> AsyncIterator[ClientSession]:
    """
    Connect to an MCP server and yield a ClientSession.

    server_spec can be:
    - An HTTP(S) URL (e.g. https://example.com/mcp) → streamable HTTP
    - A path to a .py or .js script → stdio subprocess
    """
    server_spec = _normalize_server_spec(server_spec)

    if _is_http_url(server_spec):
        async with streamable_http_client(url=server_spec) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
    elif _is_local_script(server_spec):
        command = "python" if server_spec.endswith(".py") else "node"
        params = StdioServerParameters(command=command, args=[server_spec])
        stdio_transport = stdio_client(params)
        async with stdio_transport as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
    else:
        raise ValueError(
            f"Invalid MCP server spec: {server_spec!r}. "
            "Use an HTTP(S) URL or a path to a .py/.js script."
        )
