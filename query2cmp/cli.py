"""Command-line interface for query2cmp."""

import argparse
import asyncio
import json
import sys
import traceback

from .config import get_config
from .pipeline import run_pipeline


def _create_backend(config: dict, args: argparse.Namespace):
    """Create LLM backend from config and args."""
    backend_name = (args.backend or config.get("llm_backend") or "openai").lower()

    if backend_name == "apple":
        try:
            from .llm_backends.apple_backend import AppleFMBackend
        except ImportError as e:
            raise ImportError(
                "Apple backend requires apple-fm-sdk. Install with: pip install apple-fm-sdk\n"
                "Requires macOS 26+, Apple Silicon, and Apple Intelligence enabled."
            ) from e
        return AppleFMBackend()
    else:
        from openai import OpenAI

        from .llm_backends.openai_backend import OpenAIBackend

        api_key = config.get("openai_api_key") or ""
        if not api_key:
            print(
                "Error: OPENAI_API_KEY is required for OpenAI backend. "
                "Add it to .env or set the environment variable.",
                file=sys.stderr,
            )
            sys.exit(1)

        client = OpenAI(
            api_key=api_key,
            base_url=config.get("openai_base_url") or None,
        )
        model = args.model or config.get("openai_model", "gpt-4o-mini")
        return OpenAIBackend(client=client, model=model)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Natural language query → LLM → MCP invocation. "
        "Accepts a text query, processes it via an LLM, and invokes the appropriate MCP tool."
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural language query (any language). If omitted, read from stdin.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Explain each step in detail (overrides QUERY2MCP_VERBOSE).",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose output (overrides QUERY2MCP_VERBOSE).",
    )
    parser.add_argument(
        "--mcp",
        metavar="URL_OR_SCRIPT",
        help="MCP server URL (http(s)://...) or path to .py/.js script (overrides MCP_SERVER_URL).",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "apple"],
        help="LLM backend: openai (API) or apple (on-device Foundation Model). Overrides LLM_BACKEND.",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        help="LLM model name for OpenAI backend (overrides OPENAI_MODEL).",
    )
    parser.add_argument(
        "--mcp-language",
        metavar="LANG",
        help="Translate query to this language before tool selection (e.g. en, English). Overrides MCP_LANGUAGE.",
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Output results as JSON only (no verbose explanation).",
    )
    parser.add_argument(
        "-t",
        "--text",
        action="store_true",
        help="Feed results to LLM for a natural language answer instead of raw JSON.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full error details: exception type, chain, and traceback.",
    )
    args = parser.parse_args()

    config = get_config()

    mcp_server = args.mcp or config.get("mcp_server", "")
    if not mcp_server:
        print(
            "Error: MCP server not specified. Set MCP_SERVER_URL in .env or use --mcp.",
            file=sys.stderr,
        )
        return 1

    query = args.query
    if query is None:
        query = sys.stdin.read().strip()
    if not query:
        print("Error: No query provided. Pass as argument or via stdin.", file=sys.stderr)
        return 1

    verbose = config.get("verbose", False)
    if args.verbose:
        verbose = True
    elif args.no_verbose:
        verbose = False

    mcp_language = (args.mcp_language or config.get("mcp_language", "")).strip()

    try:
        backend = _create_backend(config, args)
    except (ImportError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    backend_name = (args.backend or config.get("llm_backend") or "openai").lower()
    if verbose and not args.json:
        print("=" * 60)
        print("query2cmp: Natural Language → LLM → MCP")
        print("=" * 60)
        print(f"Query: {query!r}")
        print(f"Backend: {backend_name}")
        if backend_name == "openai":
            print(f"Model: {args.model or config.get('openai_model', 'gpt-4o-mini')}")
        print(f"MCP server: {mcp_server}")
        if mcp_language:
            print(f"MCP language: {mcp_language}")
        print()

    try:
        results = asyncio.run(
            run_pipeline(
                query,
                backend=backend,
                mcp_server_spec=mcp_server,
                mcp_language=mcp_language,
                verbose=verbose and not args.json,
            )
        )
    except Exception as e:
        def _exc_message(exc: Exception) -> str:
            if isinstance(exc, BaseExceptionGroup):
                return _exc_message(exc.exceptions[0]) if exc.exceptions else str(exc)
            return str(exc)

        def _format_error(dest=sys.stderr, prefix_newline: bool = False) -> None:
            msg = _exc_message(e)
            lines = (["\n"] if prefix_newline else []) + ["--- Error ---", f"{type(e).__name__}: {msg}"]
            exc = e
            chain = []
            while getattr(exc, "__cause__", None) or getattr(exc, "__context__", None):
                next_exc = exc.__cause__ or exc.__context__
                if next_exc and next_exc not in chain:
                    chain.append(next_exc)
                    lines.append(f"  Caused by: {type(next_exc).__name__}: {next_exc}")
                    exc = next_exc
                else:
                    break
            if args.debug:
                lines.append("")
                lines.append("Traceback:")
                lines.append(traceback.format_exc())
            for line in lines:
                print(line, file=dest)

        if verbose or args.debug:
            _format_error(
                sys.stdout if verbose and not args.json else sys.stderr,
                prefix_newline=verbose and not args.json,
            )
        else:
            print(f"Error: {_exc_message(e)}", file=sys.stderr)
        return 1

    if args.text:
        if verbose and not args.json:
            print("\n--- Step 5: Generating natural language answer ---")
        answer = asyncio.run(backend.answer_natural_language(query, results))
        print(answer)
    elif args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        if verbose:
            print("\n" + "=" * 60)
            print("Final results")
            print("=" * 60)
        for r in results:
            if r.get("tool"):
                print(f"\nTool: {r['tool']}")
                print(f"Arguments: {r.get('arguments', {})}")
                for part in r.get("result", []):
                    if isinstance(part, dict) and part.get("type") == "text":
                        print(f"Result: {part.get('text', part)}")
                    else:
                        print(f"Result: {part}")
            elif r.get("message"):
                print(f"\nMessage: {r['message']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
