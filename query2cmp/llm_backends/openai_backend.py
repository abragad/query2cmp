"""OpenAI-compatible API backend."""

import json
from typing import Any

from openai import OpenAI

from .base import ToolCallResult


def _tools_for_openai(tools: list) -> list[dict]:
    """Convert MCP tool list to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema if t.inputSchema else {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


SYSTEM_PROMPT = """You are an assistant that helps users by calling MCP (Model Context Protocol) tools.
Given the user's natural language query, choose the most appropriate tool and provide the correct arguments.
Only call tools when the query clearly requires it. Be precise with argument values."""


class OpenAIBackend:
    """LLM backend using OpenAI-compatible API."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    async def answer_natural_language(self, query: str, results: list[dict]) -> str:
        """Turn query + tool results into a natural language answer."""
        results_str = json.dumps(results, indent=2, default=str)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Given the user's query and the results from MCP tool invocations, "
                    "provide a clear, natural language answer. Be concise but complete. Answer in the same language as the query.",
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nTool results:\n{results_str}",
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()

    async def translate(self, query: str, target_language: str) -> str:
        """Translate query to the target language."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a translator. Your ONLY task is to translate the user's text to the specified language. "
                    "Do NOT answer the question. Do NOT add information. Do NOT interpret or fulfill the request. "
                    "Output ONLY the translated text, nothing else. If the user asks for a recommendation, translate that request literally.",
                },
                {
                    "role": "user",
                    "content": f"Translate this to {target_language}: {query}",
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()

    async def run(
        self,
        query: str,
        tools: list[Any],
        mcp_call_tool: Any,
        verbose: bool = False,
    ) -> list[ToolCallResult]:
        results: list[ToolCallResult] = []

        def log(msg: str, data: Any = None) -> None:
            if verbose:
                print(msg)
                if data is not None:
                    print(json.dumps(data, indent=2, default=str))

        log("\n--- Step 3: Sending query to LLM for tool selection ---")
        log(f"Query: {query!r}")
        log("Tools sent to LLM:", [{"name": t.name, "description": (t.description or "")[:80]} for t in tools])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            tools=_tools_for_openai(tools),
            tool_choice="auto",
        )

        choice = response.choices[0]
        log(f"LLM response (role={choice.message.role}):")
        if choice.message.content:
            log(f"  Content: {choice.message.content}")
        if choice.message.tool_calls:
            log(f"  Tool calls: {[(tc.function.name, tc.function.arguments) for tc in choice.message.tool_calls]}")

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    tool_args = {}

                log("\n--- Step 4: Invoking MCP tool ---")
                log(f"Tool: {tool_name}")
                log(f"Arguments: {tool_args}")

                result = await mcp_call_tool(tool_name, tool_args)
                result_parts = []
                for c in result.content:
                    if hasattr(c, "model_dump"):
                        result_parts.append(c.model_dump())
                    elif hasattr(c, "text"):
                        result_parts.append({"type": "text", "text": c.text})
                    else:
                        result_parts.append(str(c))
                results.append(ToolCallResult(tool=tool_name, arguments=tool_args, result=result_parts))
                log(f"Result: {result_parts}")
        else:
            log("\n--- Step 4: No tool call in LLM response ---")
            if choice.message.content:
                log(f"LLM message: {choice.message.content}")
                results.append(
                    ToolCallResult(tool=None, arguments={}, result=[{"type": "message", "text": choice.message.content}])
                )

        return results
