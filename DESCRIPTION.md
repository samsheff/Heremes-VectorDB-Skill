# Vector Memory

**Persistent, semantic memory for Hermes using Qdrant.**

Stores memories with embeddings for cross-session recall. Supports explicit saves, session summaries, and semantic search.

## Quick Start

1. Set env vars: `QDRANT_HOST`, `QDRANT_PORT`, `BOT_ID`
2. Run `python references/init_collection.py` to create the collection
3. Import and use:

```python
from vector_memory import store_memory, search_memories

store_memory("User prefers concise responses", memory_type="preference")
results = search_memories("What are the user's communication preferences?")
```

## Multi-Bot Deployment

Each bot uses its own collection (namespaced by `BOT_ID`). Point all bots to the same Qdrant server — isolation is automatic.

See `SKILL.md` for full documentation.
