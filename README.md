# HEBBS

**The memory engine for AI agents.** One binary. Sub-10ms recall. Agents that actually learn.

HEBBS is a memory primitive purpose-built for AI agents. It replaces the patchwork of vector databases, key-value stores, and graph databases that agent developers cobble together today with a single, fast, embeddable engine.

Vector search tells your agent what's *similar*. HEBBS tells your agent what *happened*, what *caused* it, and what *worked before*.

```bash
curl -sSf https://hebbs.dev/install | sh
hebbs-server
```

---

## Why HEBBS Exists

Current "memory" solutions for AI agents are storage dressed up as intelligence. They solve narrow retrieval problems but miss the deeper cognitive capabilities agents actually need.

| Approach | What it does | What it misses |
|---|---|---|
| Conversation History | Append-only log, truncate at window | No importance weighting, no consolidation, context just gets cut |
| Vector DB / RAG | Similarity retrieval over chunks | One retrieval path, no decay, no structural consolidation |
| Redis / KV Cache | Fast storage of computed results | No semantic understanding, manual key management for everything |
| Knowledge Graphs | Structured relationships | Hard to populate automatically, rigid schema, no temporal context |

HEBBS moves beyond storage into cognition: importance-driven encoding, multi-path recall, episodic-to-semantic consolidation, native decay and reinforcement, and revision over append.

---

## Quick Start

### Install

```bash
# macOS / Linux
curl -sSf https://hebbs.dev/install | sh

# Docker
docker run -p 6380:6380 hebbs/hebbs

# Or embed as a library (no separate process)
pip install hebbs
```

### Connect

```python
from hebbs import HEBBS

e = HEBBS("localhost:6380")
```

### Remember

```python
e.remember(
    experience="Prospect mentioned competitor contract expires March 15",
    importance=0.95,
    context={"prospect_id": "acme", "stage": "discovery", "signal": "urgency"}
)
```

### Recall

```python
# What happened with this prospect? (Temporal)
history = e.recall(cue={"prospect_id": "acme"}, strategy="temporal")

# How should I handle this objection? (Similarity)
responses = e.recall(cue="we built this in-house", strategy="similarity")

# Why did the last similar deal fall through? (Causal)
causes = e.recall(cue="deal lost after pricing", strategy="causal")

# I've never sold to healthcare -- what's transferable? (Analogical)
patterns = e.recall(cue="healthcare compliance objection", strategy="analogical")
```

### Subscribe (Associative / Real-time)

```python
# The engine pushes relevant memories as your agent processes input.
# No explicit recall needed -- knowledge surfaces automatically.

for memory in e.subscribe(input_stream=call_transcript, threshold=0.8):
    inject_into_agent_context(memory)
```

### Reflect

```python
# Configure background consolidation. HEBBS learns while your agent sleeps.
e.reflect_policy({
    "triggers": [
        {"type": "threshold", "new_memories": 50},
        {"type": "schedule", "interval": "daily"},
        {"type": "recall_failure", "confidence_below": 0.3},
        {"type": "metric_drift", "metric": "conversion_rate", "delta": 0.2}
    ],
    "strategy": "hybrid",
    "scope": "incremental"
})

# Query distilled knowledge
insights = e.insights(filter={"topic": "objection handling", "min_confidence": 0.8})
```

---

## The API

Nine operations. Three groups.

### Write

| Operation | What it does |
|---|---|
| `remember(experience, importance, context)` | Store a memory with importance scoring and structured context. |
| `revise(memory_id, new_evidence)` | Update a belief. Replaces, not appends. |
| `forget(criteria)` | Prune by staleness, irrelevance, or compliance (GDPR). |

### Read

| Operation | What it does |
|---|---|
| `recall(cue, strategy)` | Retrieve memories by similarity, time, causation, or analogy. |
| `prime(context)` | Pre-load relevant context before an agent turn. For frameworks. |
| `subscribe(input_stream, threshold)` | Real-time push. The engine surfaces memories as they become relevant. |

### Consolidate

| Operation | What it does |
|---|---|
| `reflect_policy(config)` | Configure automatic background consolidation triggers. |
| `reflect(scope)` | Manual trigger for on-demand consolidation. |
| `insights(filter)` | Query distilled knowledge produced by reflection. |

---

## Four Recall Strategies

Most memory systems give you one retrieval mode: similarity search. HEBBS gives you four.

| Strategy | Question it answers | Example |
|---|---|---|
| **Similarity** | "What looks like this?" | Finding relevant objection responses |
| **Temporal** | "What happened, in order?" | Reconstructing a prospect's full history |
| **Causal** | "What led to this outcome?" | Understanding why a deal was lost |
| **Analogical** | "What's structurally similar in a different domain?" | Applying finance patterns to healthcare |

All four run against a single engine. No fan-out across services.

---

## Performance

Benchmarked on a single `c6g.large` instance (2 vCPU, 4GB RAM) with 10M stored memories.

| Operation | p50 | p99 |
|---|---|---|
| `remember` | 0.8ms | 4ms |
| `recall` (similarity) | 2ms | 8ms |
| `recall` (temporal) | 0.5ms | 2ms |
| `recall` (causal) | 4ms | 12ms |
| `recall` (multi-strategy) | 6ms | 18ms |
| `subscribe` (event-to-push) | 1ms | 5ms |

### Scalability

| Memories | `recall` p99 (similarity) | `recall` p99 (temporal) |
|---|---|---|
| 100K | 3ms | 0.6ms |
| 1M | 5ms | 0.8ms |
| 10M | 8ms | 1.2ms |
| 100M | 12ms | 2.0ms |

---

## Architecture

```text
┌──────────────────────────────────────────────────────┐
│                    Client SDKs                       │
│            Python  │  TypeScript  │  Rust            │
├──────────────────────────────────────────────────────┤
│               gRPC  │  HTTP/REST                     │
├──────────────────────────────────────────────────────┤
│                                                      │
│                 Core Engine (Rust)                    │
│                                                      │
│  ┌────────────┐ ┌────────────┐ ┌──────────────────┐ │
│  │  Remember   │ │   Recall   │ │ Reflect Pipeline │ │
│  │  Engine     │ │   Engine   │ │ (background)     │ │
│  │            │ │            │ │                  │ │
│  │ • encode   │ │ • prime    │ │ • cluster (Rust) │ │
│  │ • score    │ │ • query    │ │ • propose (LLM)  │ │
│  │ • index    │ │ • subscribe│ │ • validate (LLM) │ │
│  │ • decay    │ │ • merge    │ │ • store insights │ │
│  └─────┬──────┘ └─────┬──────┘ └────────┬─────────┘ │
│        │              │                 │            │
│  ┌─────┴──────────────┴─────────────────┴─────────┐ │
│  │              Index Layer                        │ │
│  │   Temporal (B-tree) │ Vector (HNSW) │ Graph    │ │
│  └──────────────────────┬──────────────────────────┘ │
│                         │                            │
│  ┌──────────────────────┴──────────────────────────┐ │
│  │         Storage Engine (RocksDB)                 │ │
│  │         Column Families per index type           │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  ┌─────────────────────┐  ┌────────────────────────┐ │
│  │ Embedding Engine    │  │ LLM Provider Interface │ │
│  │ (ONNX Runtime,      │  │ (Anthropic, OpenAI,    │ │
│  │  built-in default)  │  │  Ollama -- pluggable)  │ │
│  └─────────────────────┘  └────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Built with:**
- **Rust** -- no GC pauses, single static binary, C-level performance
- **RocksDB** -- embedded LSM storage, proven by TiKV and CockroachDB
- **HNSW** -- logarithmic-scaling vector index for similarity and analogical recall
- **ONNX Runtime** -- built-in CPU embeddings (<5ms), zero external API dependencies
- **gRPC** -- bidirectional streaming for real-time `subscribe` channels

---

## Deployment

### Standalone Server (the Redis model)

```bash
hebbs-server --port 6380 --data ./hebbs-data
```

### Embedded Library (the SQLite model)

```python
from hebbs import HEBBS

e = HEBBS.open("./agent-memory")  # No separate process
e.remember(...)
```

### Edge Mode (robots, laptops, workstations)

Same API, different configuration. A Jetson Orin, MacBook, or Intel laptop runs the complete engine including local reflection with on-device LLMs.

---

## Client Libraries

| Language | Package | Status |
|---|---|---|
| Python | `pip install hebbs` | Stable |
| TypeScript | `npm install @hebbs/client` | Stable |
| Rust | `hebbs` crate | Stable |
| Go | `go get hebbs.dev/client` | Beta |

Python supports both server mode (gRPC) and embedded mode (PyO3, no separate process).

---

## Use Cases

**Voice Sales Agents** -- The most demanding test for agentic memory. A sales agent that remembers prospect history across calls, handles objections with proven responses, and learns which pitches convert over time.

**Customer Support** -- Recall past tickets for the same customer, surface solutions from similar issues, reduce escalations through consolidated troubleshooting knowledge.

**Coding Agents** -- Remember what approaches worked in this codebase, recall past debugging sessions, avoid repeating failed strategies.

**Robotics** -- Warehouse robots that learn navigation patterns, share blocked-aisle knowledge across a fleet, and reflect on operational efficiency. All running fully offline on edge hardware.

**Personal Assistants** -- Remember preferences, learn routines, pick up context across conversations.

---

## Contributing

We welcome contributions across the stack. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

All contributors must sign our [Contributor License Agreement](CLA.md) before their first PR can be merged. It's a one-time thing -- the CLA bot will walk you through it on your first pull request.

---

## License

HEBBS uses a dual-license model.

**The engine** (hebbs-core, hebbs-storage, hebbs-index, hebbs-embed, hebbs-reflect, hebbs-server, hebbs-cli) is licensed under [BSL 1.1](LICENSE-BSL). This is the same license used by CockroachDB, Sentry, and Terraform. You can use it freely in production. You can read the source, modify it, self-host it, build on top of it. The only thing you can't do is take HEBBS and offer it as a hosted service to third parties. Every version converts to Apache 2.0 after four years.

**Client libraries and protocol definitions** (hebbs-client, hebbs-proto, hebbs-ffi) are licensed under [Apache 2.0](LICENSE-APACHE). The code you import into your projects is fully open source with no restrictions.

Educational institutions and non-profit organizations can use the full engine without restriction.

If you need a different licensing arrangement, reach out at parag@hebbs.ai.

---

*Agents deserve better than a vector database and a prayer.*
