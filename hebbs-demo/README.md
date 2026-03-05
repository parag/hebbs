# hebbs-demo: AI Sales Intelligence Agent

A reference application demonstrating HEBBS's full capabilities -- compound learning, real-time memory surfacing, multi-strategy recall, and graph-based reasoning -- through an AI Sales Intelligence Agent.

## Quick Start

```bash
# 1. Install
cd hebbs-demo
pip install -e .

# 2. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# 3. Run interactive mode
hebbs-demo interactive
```

That's it. OpenAI handles all LLM calls (conversation, memory extraction, reflection) and embeddings by default.

---

## Embedding Providers

Embeddings are the most critical cost and performance lever in HEBBS. Every `remember()`, `recall()`, `subscribe()`, and `reflect()` operation depends on embedding quality. Choosing the right provider directly impacts retrieval accuracy, latency, and operating cost.

### Available Providers

| Provider | Model | Dimensions | Latency | Cost | Quality (MTEB) | Offline |
|----------|-------|-----------|---------|------|----------------|---------|
| **ONNX** (local) | BGE-small-en-v1.5 | 384 | ~3ms | Free | ~62% | Yes |
| **ONNX** (local) | BGE-large-en-v1.5 | 1024 | ~15ms | Free | ~67% | Yes |
| **OpenAI** | text-embedding-3-small | 1536 | ~100-300ms | $0.02/1M tokens | ~62% | No |
| **OpenAI** | text-embedding-3-large | 3072 | ~100-300ms | $0.13/1M tokens | ~64% | No |
| **Ollama** (local) | nomic-embed-text | 768 | ~10ms | Free | ~63% | Yes |

### How to Switch Embedding Providers

Edit the `[embedding]` section in your config file:

**Use local ONNX embeddings (free, offline, fastest):**

```toml
[embedding]
provider = "onnx"

[embedding.onnx]
model = "bge-small-en-v1.5"    # 384-dim, 3ms, free
```

**Use OpenAI embeddings (best for production quality):**

```toml
[embedding]
provider = "openai"

[embedding.openai]
api_key_env = "OPENAI_API_KEY"
model = "text-embedding-3-small"    # 1536-dim, $0.02/1M tokens
```

**Use OpenAI large embeddings (highest quality):**

```toml
[embedding]
provider = "openai"

[embedding.openai]
api_key_env = "OPENAI_API_KEY"
model = "text-embedding-3-large"    # 3072-dim, $0.13/1M tokens
```

**Use Ollama local embeddings (free, better quality than BGE-small):**

```toml
[embedding]
provider = "ollama"

[embedding.ollama]
base_url = "http://localhost:11434"
model = "nomic-embed-text"    # 768-dim, free, requires Ollama
```

### Important: Switching Requires a Fresh Database

Different embedding providers produce vectors with different dimensions and different vector spaces. You **cannot** mix embeddings from different providers in the same database.

When you change `[embedding].provider` or `[embedding.*.model]`:

```bash
# Delete the old database
rm -rf ./hebbs-data

# Run with new embeddings -- database will be recreated
hebbs-demo interactive --config configs/openai.toml
```

The demo app validates this automatically on startup. If you try to open a database created with a different embedder, it will refuse with a clear error message telling you which embedder the database was created with.

### Cost Comparison: What Embeddings Actually Cost

For a typical sales agent session (50 conversations, ~100 memories each = 5,000 memories):

| Provider | Embedding Cost | Recall Cost (per query) | Monthly Cost (1K queries/day) |
|----------|---------------|------------------------|------------------------------|
| ONNX BGE-small | $0 | $0 | **$0** |
| OpenAI 3-small | ~$0.05 (ingest) | ~$0.001 | **~$1.50** |
| OpenAI 3-large | ~$0.33 (ingest) | ~$0.006 | **~$9.50** |

ONNX is 33x-330x cheaper. The question is whether the quality difference justifies the cost.

### When to Use Which

| Scenario | Recommended Embedding | Why |
|----------|----------------------|-----|
| **Development and testing** | ONNX BGE-small | Free, offline, fast, deterministic |
| **Customer demos** | OpenAI text-embedding-3-small | Slightly better quality, familiar brand |
| **Production (cost-sensitive)** | ONNX BGE-small or BGE-large | Zero marginal cost, data stays local |
| **Production (quality-sensitive)** | OpenAI text-embedding-3-small | Best price/quality ratio with API |
| **Edge deployment** | ONNX BGE-small | Must work offline, no API dependency |
| **Embedding comparison demo** | Both | Run `compare-embeddings` to show HEBBS multi-strategy recall compensates for smaller embeddings |

---

## LLM Providers

### How to Switch LLM Providers

LLM providers are configured separately for each role:

```toml
[llm]
conversation_provider = "openai"       # generates agent responses
conversation_model = "gpt-4o"
extraction_provider = "openai"         # extracts structured memories from conversations
extraction_model = "gpt-4o-mini"
```

**Switch conversation to Anthropic:**

```toml
[llm]
conversation_provider = "anthropic"
conversation_model = "claude-sonnet-4-20250514"

[llm.anthropic]
api_key_env = "ANTHROPIC_API_KEY"
```

**Switch everything to Ollama (local, free, requires Ollama installed):**

```toml
[llm]
conversation_provider = "ollama"
conversation_model = "llama3.2"
extraction_provider = "ollama"
extraction_model = "llama3.2"

[llm.ollama]
base_url = "http://localhost:11434"

[hebbs.reflect]
provider = "ollama"
model = "llama3.2"
```

Note: the reflect pipeline LLM is configured under `[hebbs.reflect]`, not `[llm]`, because it runs inside the HEBBS engine (Rust side), not in the Python demo app.

### Pre-Built Config Profiles

```bash
# OpenAI everything (default -- just needs OPENAI_API_KEY)
hebbs-demo interactive --config configs/openai.toml

# Best demo quality (GPT-4o + OpenAI embeddings)
hebbs-demo interactive --config configs/demo-quality.toml

# Local-only, no API keys needed (requires Ollama)
hebbs-demo interactive --config configs/local.toml

# Cost-optimized (GPT-4o-mini + local ONNX embeddings)
hebbs-demo interactive --config configs/cost-optimized.toml
```

---

## Commands

### Interactive Mode

```bash
# Start a conversation with the AI sales agent
hebbs-demo interactive

# Use a specific config
hebbs-demo interactive --config configs/demo-quality.toml

# Control observability panel detail level
hebbs-demo interactive --quiet      # agent responses only
hebbs-demo interactive --normal     # collapsed HEBBS panels (default)
hebbs-demo interactive --verbose    # expanded HEBBS panels with full details
```

### Scripted Scenarios

```bash
# Run all 7 scenarios
hebbs-demo scenarios --all

# Run a specific scenario
hebbs-demo scenarios --run discovery_call
hebbs-demo scenarios --run reflect_learning
hebbs-demo scenarios --run subscribe_realtime
```

### Provider Comparisons

```bash
# Compare LLM providers (fixed embedding)
hebbs-demo compare-llm

# Compare embedding providers (fixed LLM) -- ONNX vs OpenAI
hebbs-demo compare-embeddings --configs configs/onnx-embed.toml,configs/openai-embed.toml

# Compare all embedding providers
hebbs-demo compare-embeddings --configs configs/onnx-embed.toml,configs/openai-embed.toml,configs/ollama-embed.toml
```

---

## Observability: Seeing HEBBS Work

Every conversation turn shows a HEBBS activity panel. Title bars are always visible; details expand/collapse with `d` key or `--verbose` flag.

**Collapsed (default):**

```
[+] REMEMBER   1 memory stored (importance: 0.8, entity: acme_corp)              3.2ms
[+] RECALL     5 memories retrieved (strategy: Similarity+Temporal)               4.1ms
[+] SUBSCRIBE  1 memory surfaced (confidence: 0.84)                               0.8ms
                                                                          total:  8.1ms
```

**Expanded (press `d` or use `--verbose`):**

```
[-] RECALL     5 memories retrieved (strategy: Similarity+Temporal)               4.1ms
    │ 0.91  "Beta Inc had identical SOC 2 blockers"                    [Episode]
    │ 0.87  "Gamma Corp resolved compliance via Vanta partnership"     [Episode]
    │ 0.83  "Fintech prospects prioritize compliance over cost"        [Insight]
    │ 0.79  "Last call with Acme: discussed pricing timeline"          [Episode]
    │ 0.71  "Delta Corp compliance timeline was 6 months"              [Episode]
```

---

## Environment Variables

| Variable | Required | Used By |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes (for default config) | LLM conversation, extraction, reflect, and OpenAI embeddings |
| `ANTHROPIC_API_KEY` | Only if using Anthropic | LLM conversation, extraction |

---

## Project Structure

```
hebbs-demo/
  pyproject.toml              # Dependencies: hebbs, openai, anthropic, click, rich
  README.md                   # This file
  hebbs_demo/
    cli.py                    # Click CLI: interactive, scenarios, compare-llm, compare-embeddings
    agent.py                  # SalesAgent: conversation loop with HEBBS integration
    memory_manager.py         # Memory extraction, structured remember(), context building
    llm_client.py             # Unified LLM interface wrapping openai/anthropic/ollama
    config.py                 # TOML config loading, env var resolution
    display.py                # DisplayManager: collapsible Rich panels for HEBBS activity
    scenarios/                # 7 scripted scenarios (discovery, objection, multi-session, ...)
    reports/                  # Comparison report generators
  tests/                      # pytest suite
  configs/                    # Pre-built config profiles
    openai.toml               # Default: OpenAI for LLM + embeddings
    demo-quality.toml         # Best quality: GPT-4o + OpenAI embeddings
    local.toml                # Local-only: Ollama + ONNX (requires Ollama)
    cost-optimized.toml       # Budget: GPT-4o-mini + local ONNX embeddings
    openai-embed.toml         # OpenAI embeddings (for compare-embeddings)
    onnx-embed.toml           # ONNX BGE-small embeddings (for compare-embeddings)
```
