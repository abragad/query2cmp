"""Pipeline: natural language query → LLM → MCP tool selection and invocation."""

import asyncio
import json
import time
from typing import Any

from .llm_backends.base import LLMBackend, ToolCallResult
from .mcp_client import mcp_session

WAITING_INTERVAL = 3  # seconds between "still waiting" messages


async def run_pipeline(
    query: str,
    *,
    backend: LLMBackend,
    mcp_server_spec: str,
    mcp_language: str = "",
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """
    Run the full pipeline: query → [translate] → LLM → MCP invocation → results.

    Returns a list of result dicts (one per tool call).
    """
    results: list[dict[str, Any]] = []

    def log(msg: str, data: Any = None) -> None:
        if verbose:
            print(msg)
            if data is not None:
                print(json.dumps(data, indent=2, default=str))

    if mcp_language:
        log("\n--- Step 0: Translating query to MCP language ---")
        log(f"Target language: {mcp_language}")
        log(f"Original query: {query!r}")
        query = await backend.translate(query, mcp_language)
        log(f"Translated query: {query!r}")

    log("\n--- Step 1: Connecting to MCP server ---")
    log(f"MCP server spec: {mcp_server_spec}")

    async with mcp_session(mcp_server_spec) as session:
        log("\n--- Step 2: Listing available MCP tools ---")
        list_response = await session.list_tools()
        tools = list_response.tools
        tool_names = [t.name for t in tools]
        log(f"Available tools: {tool_names}")

        if not tools:
            log("No tools available on this MCP server.")
            return results

        async def mcp_call_tool(tool_name: str, arguments: dict) -> Any:
            if verbose:
                log("  4a. Sending request to MCP server...")
                start = time.perf_counter()

            async def _heartbeat() -> None:
                """Print periodic 'still waiting' while the call is in progress."""
                while True:
                    await asyncio.sleep(WAITING_INTERVAL)
                    elapsed = time.perf_counter() - start
                    if verbose:
                        print(f"     ... still waiting ({elapsed:.1f}s)")

            call_task = asyncio.create_task(session.call_tool(tool_name, arguments))
            heartbeat_task = asyncio.create_task(_heartbeat()) if verbose else None
            try:
                result = await call_task
            finally:
                if heartbeat_task:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

            if verbose:
                elapsed = time.perf_counter() - start
                log(f"  4b. Response received ({elapsed:.2f}s)")
                n_parts = len(result.content) if hasattr(result, "content") else 0
                log(f"  4c. Parsing result ({n_parts} content part(s))...")
            return result

        backend_results = await backend.run(
            query=query,
            tools=tools,
            mcp_call_tool=mcp_call_tool,
            verbose=verbose,
        )

        for r in backend_results:
            results.append(r.to_dict())

    return results
