# Semantic-RTS Command Reference

## Phase 1 — Knowledge Base Construction

### 1. Checkout a Defects4J project

```bash
# Checkout a specific bug version (e.g., Cli bug 1, fixed version)
defects4j checkout -p Cli -v 1f -w /tmp/Cli_1f
```

### 2. Set up the Python environment (WSL / Linux)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a Linux-native venv and install dependencies
cd /path/to/semantic-rts
uv venv .venv-linux --python 3.11
source .venv-linux/bin/activate
uv pip install -e ".[dev]"
```

### 3. Configure API key

```bash
# Copy example env file and add your Google AI Studio key
cp .env.example .env
# Edit .env and set: GOOGLE_API_KEY=your_key_here
```

### 4. Build the Knowledge Base

```bash
cd /path/to/semantic-rts
source .venv-linux/bin/activate

# Full build (resume by default — skips already-processed tests)
srts build --project-path /tmp/Cli_1f --project Cli

# Limit API calls (useful for testing)
srts build --project-path /tmp/Cli_1f --project Cli --max-requests 20

# Clean build — ignore existing KB and reprocess everything
srts build --project-path /tmp/Cli_1f --project Cli --force

# Clean build from scratch (also clears stale embedding cache)
rm -rf data/kb/Cli data/cache/embeddings
srts build --project-path /tmp/Cli_1f --project Cli --force
```

### 5. Verify the KB

```bash
# Check files produced
ls -lh data/kb/Cli/
# Expected: index.faiss, index.meta.json, tests.jsonl

# Check how many tests are indexed
python3 -c "
import json
with open('data/kb/Cli/index.meta.json') as f:
    meta = json.load(f)
print(f'Tests indexed : {meta[\"n_tests\"]}')
print(f'Embedding dim : {meta[\"dim\"]}')
"

# Inspect a sample test entry
python3 -c "
import json
with open('data/kb/Cli/tests.jsonl') as f:
    print(json.dumps(json.loads(f.readline()), indent=2))
"
```

---

## Configuration

Key settings in `config/default.yaml`:

| Setting | Value | Notes |
|---|---|---|
| `chat_model` | `gemini-3.1-flash-lite-preview` | 15 RPM, 500 RPD on free tier |
| `embedding_model` | `gemini-embedding-001` | 3072-d vectors |
| `rate_limit_rpm` | `9` | Conservative — stays under 15 RPM limit |
| `max_retries` | `5` | 429 waits 60s; 503 waits up to 30s |
| `embedding_dim` | `3072` | Must match embedding model output |
