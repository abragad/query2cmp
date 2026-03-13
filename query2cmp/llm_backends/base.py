"""Base LLM backend interface."""

from typing import Any, Protocol


class ToolCallResult:
    """Result of a single tool invocation or LLM message."""

    def __init__(
        self,
        tool: str | None,
        arguments: dict[str, Any],
        result: list[Any],
    ):
        self.tool = tool
        self.arguments = arguments
        self.result = result

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"tool": self.tool, "arguments": self.arguments, "result": self.result}
        if self.tool is None and self.result:
            first = self.result[0]
            msg = first.get("text", first) if isinstance(first, dict) else first
            d["message"] = msg
        return d


class LLMBackend(Protocol):
    """Protocol for LLM backends that select and optionally execute MCP tools."""

    async def translate(self, query: str, target_language: str) -> str:
        """Translate query to the target language. Used when MCP tools use a specific language."""
        ...

    async def answer_natural_language(self, query: str, results: list[dict[str, Any]]) -> str:
        """Turn query + tool results into a natural language answer."""
        ...

    async def run(
        self,
        query: str,
        tools: list[Any],
        mcp_call_tool: Any,
        verbose: bool = False,
    ) -> list[ToolCallResult]:
        """
        Process query with the LLM and return tool call results.

        :param query: Natural language query
        :param tools: Available MCP tools
        :param mcp_call_tool: Async callable (tool_name, arguments) -> CallToolResult
        :param verbose: Whether to log steps
        :return: List of tool call results (may be empty if no tool was called)
        """
        ...
