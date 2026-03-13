# query2cmp

A command-line tool that takes a natural language query (in any language), processes it via an LLM (local or remote), and invokes the appropriate tool on a remote MCP server. Built for testing and education.

## Flow

```
Natural language query → LLM (tool selection) → MCP tool invocation → Results
```

## Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `LLM_BACKEND` | `openai` (default) or `apple` |
| `OPENAI_API_KEY` | API key (required for OpenAI backend) |
| `OPENAI_BASE_URL` | Base URL (default: `https://api.openai.com/v1`). Use for Azure, Ollama, LiteLLM, etc. |
| `OPENAI_MODEL` | Model name (default: `gpt-4o-mini`) |
| `MCP_SERVER_URL` | MCP server: HTTP(S) URL or path to `.py`/`.js` script |
| `MCP_LANGUAGE` | Translate query to this language before tool selection (e.g. `en`, `English`). Empty = no translation |
| `QUERY2MCP_VERBOSE` | Set to `true` to explain each step in detail |

### OpenAI-compatible endpoints

The tool uses the OpenAI Python client, so it works with any OpenAI-compatible API:

- **OpenAI**: `OPENAI_BASE_URL=https://api.openai.com/v1`
- **Azure OpenAI**: `OPENAI_BASE_URL=https://<resource>.openai.azure.com/openai/deployments/<deployment>`
- **Ollama**: `OPENAI_BASE_URL=http://localhost:11434/v1`, `OPENAI_MODEL=llama3.2`
- **LiteLLM**: `OPENAI_BASE_URL=http://localhost:4000/v1`

### Apple on-device Foundation Model

Use `LLM_BACKEND=apple` or `--backend apple` to use Apple's on-device Foundation Model on macOS. No API key required.

**Requirements:**
- macOS 26.0+ (Tahoe)
- Apple Silicon Mac
- Apple Intelligence enabled
- `pip install apple-fm-sdk`

```bash
# Install with Apple support
pip install -e ".[apple]"

# Use Apple backend
query2cmp --backend apple --mcp examples/simple_mcp_server.py "Greet Alice"
```

## Usage

```bash
# Basic usage (uses .env)
query2cmp "Navigate to example.com"

# Verbose mode: explain each step
query2cmp -v "What's the weather in London?"

# Override MCP server
query2cmp --mcp http://localhost:8000/mcp "List my tasks"

# Override model
query2cmp --model gpt-4o "Search for Python tutorials"

# Use Apple on-device model (macOS 26+, Apple Silicon)
query2cmp --backend apple "Greet Bob"

# Translate query to English before tool selection (for MCP servers with English tool descriptions)
query2cmp --mcp-language en "Saluta Mario"
query2cmp --mcp-language English "Aggiungi 3 e 5"

# Natural language answer (feed results to LLM for a readable response)
query2cmp --text "Ultimo evento effettuato"

# JSON output only
query2cmp -j "Open google.com"

# Query from stdin
echo "Navigate to github.com" | query2cmp
```

## Verbose mode

With `-v` or `QUERY2MCP_VERBOSE=true`, the tool prints each step:

0. **Translating query** (if `MCP_LANGUAGE` set) – Original query → translated query
1. **Connecting to MCP server** – URL or script path
2. **Listing available MCP tools** – Tools exposed by the server
3. **Sending query to LLM** – Query and tools sent for tool selection
4. **Invoking MCP tool** – Tool name, arguments, and result
   - 4a. Sending request to MCP server
   - 4b. Response received (with timing)
   - 4c. Parsing result
5. **Generating natural language answer** (with `--text`) – Query + results → LLM → readable answer

## MCP server types

- **Remote HTTP**: Use a full URL, e.g. `http://localhost:8000/mcp` (Streamable HTTP)
- **Local stdio**: Use a path to a `.py` or `.js` script, e.g. `./server.py`

## Example: local MCP server

A minimal test server is included. For stdio servers, query2cmp spawns the process automatically:

```bash
query2cmp --mcp examples/simple_mcp_server.py -v "Greet Alice"
query2cmp --mcp examples/simple_mcp_server.py -v "Add 3 and 5"
```

## License

GPL v3
