#!/usr/bin/env python3
"""Minimal MCP server for testing query2cmp. Run with: python examples/simple_mcp_server.py"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Simple Test Server")


@mcp.tool()
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    mcp.run(transport="stdio")
