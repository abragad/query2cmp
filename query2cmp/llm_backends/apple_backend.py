"""Apple Foundation Model (on-device) backend.

Requires: macOS 26+, Apple Silicon, Apple Intelligence enabled, apple-fm-sdk.
"""

import json
import sys

# Apple on-device model has a limited context window. Truncate large tool results
# before feeding back to the model to avoid ExceededContextWindowSizeError.
MAX_TOOL_RESULT_CHARS = 4000
from typing import Any, Optional

from .base import ToolCallResult

try:
    import apple_fm_sdk as fm
    from apple_fm_sdk.generation_guide import GenerationGuide
    from apple_fm_sdk.generation_property import Property
    from apple_fm_sdk.generation_schema import GenerationSchema

    _APPLE_FM_AVAILABLE = True
except ImportError:
    _APPLE_FM_AVAILABLE = False
    fm = None


def _json_schema_type_to_python(prop_schema: dict) -> type:
    """Map JSON Schema type to Python type (Apple FM supports str, int, float, bool, List[str])."""
    t = prop_schema.get("type", "string")
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        from typing import List

        return List[str]
    return str


def _json_schema_to_properties(input_schema: dict) -> list[Property]:
    """Convert MCP JSON Schema to Apple FM Property list."""
    if not input_schema or input_schema.get("type") != "object":
        return []

    props_schema = input_schema.get("properties") or {}
    required = set(input_schema.get("required") or [])

    properties = []
    for name, prop_def in props_schema.items():
        if not isinstance(prop_def, dict):
            continue
        prop_type = _json_schema_type_to_python(prop_def)
        description = prop_def.get("description") or ""
        guides = []

        if "enum" in prop_def:
            guides.append(GenerationGuide.anyOf([str(v) for v in prop_def["enum"]]))
        elif "anyOf" in prop_def:
            enum_vals = []
            for opt in prop_def["anyOf"]:
                if isinstance(opt, dict) and "const" in opt:
                    enum_vals.append(str(opt["const"]))
                elif isinstance(opt, dict) and "enum" in opt:
                    enum_vals.extend(str(v) for v in opt["enum"])
            if enum_vals:
                guides.append(GenerationGuide.anyOf(enum_vals))

        if name not in required:
            from typing import Optional

            prop_type = Optional[prop_type]

        properties.append(Property(name=name, type_class=prop_type, description=description or None, guides=guides))

    return properties


def _create_mcp_tool_class(
    mcp_tool: Any,
    mcp_call_tool: Any,
    results: list[ToolCallResult],
    verbose: bool,
) -> type:
    """Create a dynamic Apple FM Tool class for an MCP tool."""
    input_schema = mcp_tool.inputSchema or {"type": "object", "properties": {}}
    properties = _json_schema_to_properties(input_schema)

    if not properties:
        properties = [Property(name="query", type_class=str, description="The query or input", guides=[])]

    type_name = f"Args_{mcp_tool.name}"
    dummy_type = type(type_name, (), {})

    schema = GenerationSchema(
        type_class=dummy_type,
        description=mcp_tool.description or f"Arguments for {mcp_tool.name}",
        properties=properties,
    )

    async def call(self, args) -> str:
        arg_dict = {}
        for prop in properties:
            try:
                val = args.value(prop.type_class, for_property=prop.name)
                if val is not None:
                    arg_dict[prop.name] = val
            except Exception:
                pass

        if verbose:
            print(f"\n--- Step 4: Invoking MCP tool ---")
            print(f"Tool: {mcp_tool.name}")
            print(f"Arguments: {arg_dict}")

        result = await mcp_call_tool(mcp_tool.name, arg_dict)
        result_parts = []
        for c in result.content:
            if hasattr(c, "model_dump"):
                result_parts.append(c.model_dump())
            elif hasattr(c, "text"):
                result_parts.append({"type": "text", "text": c.text})
            else:
                result_parts.append(str(c))

        results.append(ToolCallResult(tool=mcp_tool.name, arguments=arg_dict, result=result_parts))
        if verbose:
            print(f"Result: {result_parts}")

        result_str = json.dumps(result_parts)
        if len(result_str) > MAX_TOOL_RESULT_CHARS:
            n = len(result_parts)
            for keep in range(min(20, n), 0, -1):
                candidate = json.dumps(result_parts[:keep])
                if len(candidate) < MAX_TOOL_RESULT_CHARS - 60:
                    result_str = candidate + (
                        f" ... [truncated: {n} items total, showing first {keep}]"
                    )
                    break
            else:
                result_str = f"[Tool returned {n} items, too large for model context. See full result above.]"
        return result_str

    schema_ref = [schema]

    @property
    def arguments_schema(self):
        return schema_ref[0]

    tool_class = type(
        f"MCPTool_{mcp_tool.name}",
        (fm.Tool,),
        {
            "name": mcp_tool.name,
            "description": mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            "arguments_schema": arguments_schema,
            "call": call,
        },
    )
    return tool_class


class AppleFMBackend:
    """LLM backend using Apple's on-device Foundation Model."""

    def __init__(self):
        if not _APPLE_FM_AVAILABLE:
            raise ImportError(
                "apple-fm-sdk is required for Apple backend. Install with: pip install apple-fm-sdk. "
                "Requires macOS 26+, Apple Silicon, and Apple Intelligence enabled."
            )
        self.model = fm.SystemLanguageModel(
            guardrails=fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS,
        )
        is_available, reason = self.model.is_available()
        if not is_available:
            raise RuntimeError(f"Apple Foundation Model not available: {reason}")

    async def answer_natural_language(self, query: str, results: list) -> str:
        """Turn query + tool results into a natural language answer."""
        results_str = json.dumps(results, indent=2, default=str)
        if len(results_str) > MAX_TOOL_RESULT_CHARS:
            results_str = (
                results_str[: MAX_TOOL_RESULT_CHARS - 60]
                + f"\n... [truncated, {len(results_str)} chars total]"
            )
        session = fm.LanguageModelSession(
            model=self.model,
            instructions="You are a helpful assistant. Given the user's query and the results from MCP tool invocations, "
            "provide a clear, natural language answer. Be concise but complete. Answer in the same language as the query.",
        )
        response = await session.respond(
            prompt=f"Query: {query}\n\nTool results:\n{results_str}"
        )
        return (response or "").strip()

    async def translate(self, query: str, target_language: str) -> str:
        """Translate query to the target language."""
        try:
            session = fm.LanguageModelSession(
                model=self.model,
                instructions="You are a translator. Your ONLY task is to translate the user's text to the specified language. "
                "Do NOT answer the question. Do NOT add information. Do NOT interpret or fulfill the request. "
                "Output ONLY the translated text, nothing else. If the user asks for a recommendation, translate that request literally.",
            )
            response = await session.respond(
                prompt=f"Translate this to {target_language}: {query}"
            )
            return (response or "").strip()
        except fm.GuardrailViolationError:
            print(
                "Warning: Translation skipped (guardrail). Using original query.",
                file=sys.stderr,
            )
            return query

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

        log("\n--- Step 3: Sending query to Apple Foundation Model ---")
        log(f"Query: {query!r}")
        log("Tools sent to LLM:", [{"name": t.name, "description": (t.description or "")[:80]} for t in tools])

        fm_tools = [_create_mcp_tool_class(t, mcp_call_tool, results, verbose)() for t in tools]

        session = fm.LanguageModelSession(
            model=self.model,
            instructions="You are an assistant that helps users by calling MCP tools. "
            "Given the user's query, choose the most appropriate tool and provide the correct arguments. "
            "Only call tools when the query clearly requires it.",
            tools=fm_tools,
        )

        try:
            response = await session.respond(prompt=query)
            if not results and response:
                log("\n--- Step 4: No tool call in model response ---")
                log(f"Model message: {response}")
                results.append(
                    ToolCallResult(tool=None, arguments={}, result=[{"type": "message", "text": str(response)}])
                )
        except fm.ToolCallError as e:
            if verbose:
                print(f"\nTool call error: {e.tool_name} - {e.underlying_error}")
            raise

        return results
