---
name: vector-memory
description: Vector-based persistent memory for Hermes using Qdrant — stores and retrieves memories with semantic search, supports explicit saves and session summaries across sessions.
version: 1.0.0
author: Hermes Agent
license: MIT
dependencies: [qdrant-client>=1.7.0, sentence-transformers>=2.2.0]
metadata:
  hermes:
    tags: [memory, vector-database, qdrant, embeddings, persistent-memory, RAG]
    related_skills: [session_search]
prerequisites:
  environment:
    - QDRANT_HOST
    - QDRANT_PORT
    - QDRANT_COLLECTION
    - BOT_ID
    - EMBEDDING_MODEL (optional, default: all-MiniLM-L6-v2; 384dim, tested with sentence-transformers local embeddings)
    - EMBEDDING_PROVIDER (optional, default: local; other: openai, cohere)
    - OPENAI_API_KEY (if using openai embeddings)
    - COHERE_API_KEY (if using cohere embeddings)
  packages: [qdrant-client, sentence-transformers]
---

# Vector Memory

Persistent, semantic memory for Hermes using Qdrant vector database. Enables cross-session recall via embedding-based similarity search.

## Two Components

This skill ships as two pieces that work together:

1. **`references/vector_memory.py`** — Pure Python library (importable, no Hermes deps). Used by the plugin and for standalone scripting.

2. **`plugins/memory/vector-memory/`** — MemoryProvider plugin that wires vector-memory into Hermes's memory system automatically. Handles session hooks, tool schemas, and lifecycle.

**The skill doc is for reference and standalone use. The plugin is what makes it run inside Hermes.**

---

## When to Use

- User says "remember this" or "save this for later"
- Saving preferences, facts, or context across sessions
- Finding relevant past context when starting a new session on a topic
- Session-end automatic summary storage
- Any time the user explicitly requests something be saved to memory

## When NOT to Use

- Short-term working memory within a session → use LLM context
- Ephemeral notes → use the built-in `memory` tool instead
- Simple key-value retrieval → use the built-in `memory` tool instead

## Architecture

```
Hermes Instance                    Central Qdrant
┌────────────────────────┐        ┌──────────────────────┐
│  plugins/             │        │  Collection:          │
│  memory/              │        │  hermes_memory_<bot>   │
│  vector-memory/        │        │                       │
│  └── __init__.py      │        │  Points:              │
│     (MemoryProvider)  │        │  • id (uuid)          │
│                        │        │  • vector (1024dim)   │
│  skills/              │        │  • payload:           │
│  memory/              │        │    text, type, tags,  │
│  vector-memory/       │        │    timestamp, meta    │
│  └── references/      │        └──────────────────────┘
│     vector_memory.py  │
└────────────────────────┘
```

---

## Installation

### 1. Install dependencies

```bash
pip install qdrant-client sentence-transformers
```

### 2. Deploy to all servers

**Option A — Shared volume (recommended for k8s):**
Mount `~/.hermes/` from a shared storage so both the skill and plugin are accessible everywhere.

**Option B — Git sync:**
Push `~/.hermes/skills/memory/vector-memory/` and `~/.hermes/hermes-agent/plugins/memory/vector-memory/` to a git repo, clone on each server.

### 3. Configure per bot

Set environment variables for each Hermes instance:

```bash
# Required — point all bots to the same central Qdrant
export QDRANT_HOST=10.0.1.50
export QDRANT_PORT=6333
export QDRANT_COLLECTION=hermes_memory

# Per-bot — this is the namespace
export BOT_ID=alpha-bot-1
```

### 4. Activate the plugin

```bash
hermes memory setup
# Select "vector-memory" as the memory provider
```

Or set in config:
```yaml
memory:
  provider: vector-memory
```

### 5. One-time collection setup *(run once per bot)*

```bash
python ~/.hermes/skills/memory/vector-memory/references/setup_once.py
```

> **Idempotent & safe** — creates the Qdrant collection, verifies connectivity,
> and writes a lock file at `~/.hermes/bots/<bot_id>/.vector_memory.lock`.
> Re-running does nothing (refuses with a clear error). To re-run after a
> Qdrant migration, pass `--force`.

---

## Files

- **[setup_once.py](references/setup_once.py)** — One-time setup tool with idempotency guard
- **[init_collection.py](references/init_collection.py)** — Called automatically by setup_once.py
- **[vector_memory.py](references/vector_memory.py)** — Core `VectorMemory` library class + CLI
- **Plugin:** `~/.hermes/hermes-agent/plugins/memory/vector-memory/__init__.py` — MemoryProvider implementation

---

## Tools Available in Chat
```
Search: "What language does the user prefer?"
```
Returns ranked memories by semantic similarity.

### `vm_save` — Explicit save
```
Save: "User prefers responses in British English"
Type: preference
Tags: ["language", "writing-style"]
```
Stores verbatim — no LLM extraction needed.

### `vm_list` — View all memories
```
List type: preference
Limit: 10
```
Shows stored memories, newest first.

### `vm_delete` — Remove a memory
```
memory_id: "uuid-here"
```

---

## Automatic Behaviour

### Session end → summary stored
When a session ends (exit, /reset, timeout), `on_session_end` fires automatically:
- Messages are concatenated and summarized
- Topics are extracted (simple keyword)
- Stored as `memory_type=session_summary`

### Pre-compression → context preserved
When context compression triggers, `on_pre_compress` fires:
- Key session context is included in the compression summary
- Nothing is lost when context is truncated

### Built-in memory writes → auto-mirrored
When the user saves something via the built-in `memory` tool, `on_memory_write` fires:
- That memory is automatically also stored in vector memory
- No duplicate tool calls needed

### Turn start → prefetch
At the start of each turn, relevant memories are prefetched in the background and injected as context.

---

## Multi-Bot Isolation

Each bot uses its own collection:

```
hermes_memory_alpha_bot_1    ← Bot 1's memories
hermes_memory_beta_bot_2     ← Bot 2's memories
hermes_memory_trading_bot    ← Trading bot's memories
```

The `BOT_ID` env var determines the namespace. Bots never see each other's memories.

---

## Embedding Providers

**Local (default — BGE-m3):**
```bash
export EMBEDDING_PROVIDER=local
export EMBEDDING_MODEL=BAAI/bge-m3
```

**OpenAI:**
```bash
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export EMBEDDING_MODEL=text-embedding-3-small
```

**Cohere:**
```bash
export EMBEDDING_PROVIDER=cohere
export COHERE_API_KEY=...
export EMBEDDING_MODEL=embed-english-v3.0
```

---

## Scaling to 10+ Servers

| Concern | Solution |
|---|---|
| **Bot isolation** | Collection per `BOT_ID` |
| **Skill/plugin sync** | Shared volume or git push |
| **Single Qdrant endpoint** | All bots point to `QDRANT_HOST` |
| **Connection pooling** | HTTP session reuse (built into qdrant-client) |
| **HA** | Deploy Qdrant as a 3-node cluster |

---

## Files

- **[vector_memory.py](references/vector_memory.py)** — Core `VectorMemory` library class + CLI
- **[init_collection.py](references/init_collection.py)** — One-time collection setup
- **Plugin:** `~/.hermes/hermes-agent/plugins/memory/vector-memory/__init__.py` — MemoryProvider implementation
