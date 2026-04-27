"""
Vector Memory — Qdrant-backed semantic memory for Hermes.

Usage:
    from vector_memory import VectorMemory, store_memory, search_memories, list_memories

    vm = VectorMemory()
    vm.store("User prefers British English", memory_type="preference", tags=["language"])
    results = vm.search("What language does the user prefer?")
"""

from __future__ import annotations

import os
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_BASE = os.getenv("QDRANT_COLLECTION", "hermes_memory")
BOT_ID = os.getenv("BOT_ID", "default")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

# Derived
COLLECTION_NAME = f"{QDRANT_COLLECTION_BASE}_{BOT_ID.replace('-', '_').replace(' ', '_')}"

# Embedding dimension by model (approximate — verify for your model)
EMBEDDING_DIM = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "embed-english-v3.0": 1024,
}.get(EMBEDDING_MODEL, 1024)


# ---------------------------------------------------------------------------
# Memory types
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    PROJECT = "project"
    SESSION_SUMMARY = "session_summary"
    MANUAL = "manual"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    id: str
    text: str
    memory_type: str
    tags: list[str]
    timestamp: str
    metadata: dict

    def to_payload(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "memory_type": self.memory_type,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_point(cls, point: dict) -> "MemoryRecord":
        payload = point.get("payload", {})
        return cls(
            id=payload.get("id", point.get("id", "")),
            text=payload.get("text", ""),
            memory_type=payload.get("memory_type", "manual"),
            tags=payload.get("tags", []),
            timestamp=payload.get("timestamp", ""),
            metadata=payload.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _get_embedder():
    """Lazy-load the appropriate embedding model."""
    if EMBEDDING_PROVIDER == "openai":
        return _OpenAIEmbedder()
    elif EMBEDDING_PROVIDER == "cohere":
        return _CohereEmbedder()
    else:
        return _LocalEmbedder()


class _LocalEmbedder:
    """Local sentence-transformers embedder."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        self._dim = EMBEDDING_DIM

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()


class _OpenAIEmbedder:
    """OpenAI API embedder."""

    def __init__(self):
        import openai
        self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = EMBEDDING_MODEL

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(
            model=self._model,
            input=text,
            encoding_format="float",
        )
        return resp.data[0].embedding


class _CohereEmbedder:
    """Cohere API embedder."""

    def __init__(self):
        import cohere
        self._client = cohere.Client(os.getenv("COHERE_API_KEY"))
        self._model = EMBEDDING_MODEL

    def embed(self, text: str) -> list[float]:
        resp = self._client.embed(
            texts=[text],
            model=self._model,
            input_type="search_document",
        )
        return resp.embeddings[0]


# ---------------------------------------------------------------------------
# Qdrant client (lazy singleton)
# ---------------------------------------------------------------------------

_qdrant_client = None


def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            timeout=10,
            prefer_grpc=True,
        )
    return _qdrant_client


# ---------------------------------------------------------------------------
# VectorMemory class
# ---------------------------------------------------------------------------

class VectorMemory:
    """
    Qdrant-backed semantic memory store.

    Each bot uses its own collection (derived from BOT_ID).
    """

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        embedding_provider: str = EMBEDDING_PROVIDER,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._embedding_model = embedding_model
        self._embedding_provider = embedding_provider
        self._embedder = None
        self._qdrant = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = _get_embedder()
        return self._embedder

    @property
    def qdrant(self):
        if self._qdrant is None:
            self._qdrant = _get_qdrant()
        return self._qdrant

    # ---- Store ----

    def store(
        self,
        text: str,
        memory_type: str = "manual",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Store a memory in Qdrant.

        Returns: {"success": True, "memory_id": <uuid>, "score": 1.0}
        """
        memory_id = str(uuid.uuid4())
        vector = self.embedder.embed(text)
        now = datetime.now(timezone.utc).isoformat()

        point = {
            "id": memory_id,
            "vector": vector,
            "payload": {
                "id": memory_id,
                "text": text,
                "memory_type": memory_type,
                "tags": tags or [],
                "timestamp": now,
                "metadata": metadata or {},
            },
        }

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

        logger.info(f"[vector-memory] Stored memory {memory_id} ({memory_type}): {text[:80]}")
        return {"success": True, "memory_id": memory_id, "score": 1.0}

    # ---- Search ----

    def search(
        self,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None,
        score_threshold: float = 0.7,
    ) -> list[dict]:
        """
        Semantic search over memories.

        Returns list of dicts with keys: id, text, memory_type, tags, timestamp, score
        """
        query_vector = self.embedder.embed(query)

        filter_cond = None
        if memory_type:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            filter_cond = Filter(
                must=[
                    FieldCondition(
                        key="memory_type",
                        match=MatchValue(value=memory_type),
                    )
                ]
            )

        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_cond,
            score_threshold=score_threshold,
            with_payload=True,
        )

        out = []
        for r in results:
            payload = r.payload
            out.append({
                "id": payload.get("id", r.id),
                "text": payload.get("text", ""),
                "memory_type": payload.get("memory_type", "manual"),
                "tags": payload.get("tags", []),
                "timestamp": payload.get("timestamp", ""),
                "metadata": payload.get("metadata", {}),
                "score": r.score,
            })
        return out

    # ---- List ----

    def list(
        self,
        limit: int = 100,
        memory_type: Optional[str] = None,
    ) -> list[dict]:
        """
        List recent memories, newest first.
        """
        filter_cond = None
        if memory_type:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            filter_cond = Filter(
                must=[
                    FieldCondition(
                        key="memory_type",
                        match=MatchValue(value=memory_type),
                    )
                ]
            )

        results = self.qdrant.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_cond,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        points = results[0] if isinstance(results, tuple) else results

        # Sort by timestamp descending
        def get_ts(p):
            return p.payload.get("timestamp", "") if p.payload else ""

        out = []
        for p in sorted(points, key=get_ts, reverse=True):
            if p.payload:
                out.append({
                    "id": p.payload.get("id", p.id),
                    "text": p.payload.get("text", ""),
                    "memory_type": p.payload.get("memory_type", "manual"),
                    "tags": p.payload.get("tags", []),
                    "timestamp": p.payload.get("timestamp", ""),
                    "metadata": p.payload.get("metadata", {}),
                })
        return out

    # ---- Delete ----

    def delete(self, memory_id: str) -> dict:
        """Delete a memory by ID."""
        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=[memory_id],
        )
        logger.info(f"[vector-memory] Deleted memory {memory_id}")
        return {"success": True, "memory_id": memory_id}

    # ---- Session summaries ----

    def store_session_summary(
        self,
        session_id: str,
        summary: str,
        topics: Optional[list[str]] = None,
        turn_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Store an end-of-session summary."""
        meta = metadata or {}
        meta["session_id"] = session_id
        meta["turn_count"] = turn_count
        return self.store(
            text=f"Session {session_id}: {summary}",
            memory_type="session_summary",
            tags=topics or [],
            metadata=meta,
        )

    def get_session_context(
        self,
        current_topic: str,
        limit: int = 3,
    ) -> list[dict]:
        """
        Retrieve relevant session summaries for context injection.
        Filters to session_summary type only.
        """
        return self.search(
            query=current_topic,
            limit=limit,
            memory_type="session_summary",
            score_threshold=0.65,
        )


# ---------------------------------------------------------------------------
# Convenience functions (singleton instance)
# ---------------------------------------------------------------------------

_vm: Optional[VectorMemory] = None


def _get_vm() -> VectorMemory:
    global _vm
    if _vm is None:
        _vm = VectorMemory()
    return _vm


def store_memory(
    text: str,
    memory_type: str = "manual",
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Store a memory using the default instance."""
    return _get_vm().store(text, memory_type, tags, metadata)


def search_memories(
    query: str,
    limit: int = 5,
    memory_type: Optional[str] = None,
    score_threshold: float = 0.7,
) -> list[dict]:
    """Search memories using the default instance."""
    return _get_vm().search(query, limit, memory_type, score_threshold)


def list_memories(
    limit: int = 100,
    memory_type: Optional[str] = None,
) -> list[dict]:
    """List memories using the default instance."""
    return _get_vm().list(limit, memory_type)


def delete_memory(memory_id: str) -> dict:
    """Delete a memory using the default instance."""
    return _get_vm().delete(memory_id)


def store_session_summary(
    session_id: str,
    summary: str,
    topics: Optional[list[str]] = None,
    turn_count: int = 0,
    metadata: Optional[dict] = None,
) -> dict:
    """Store a session summary using the default instance."""
    return _get_vm().store_session_summary(session_id, summary, topics, turn_count, metadata)


def get_recent_session_context(
    current_topic: str,
    limit: int = 3,
) -> list[dict]:
    """Get relevant past session context for the current topic."""
    return _get_vm().get_session_context(current_topic, limit)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vector Memory CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_store = sub.add_parser("store", help="Store a memory")
    p_store.add_argument("text")
    p_store.add_argument("--type", "-t", default="manual")
    p_store.add_argument("--tags", "-g", nargs="*", default=[])

    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query")
    p_search.add_argument("--limit", "-l", type=int, default=5)
    p_search.add_argument("--type", "-t", default=None)

    p_list = sub.add_parser("list", help="List memories")
    p_list.add_argument("--limit", "-l", type=int, default=50)
    p_list.add_argument("--type", "-t", default=None)

    p_delete = sub.add_parser("delete", help="Delete a memory")
    p_delete.add_argument("memory_id")

    args = parser.parse_args()

    if args.cmd == "store":
        result = store_memory(args.text, args.type, args.tags)
        print(json.dumps(result, indent=2))
    elif args.cmd == "search":
        results = search_memories(args.query, args.limit, args.type)
        for r in results:
            print(f"[{r['score']:.3f}] [{r['memory_type']}] {r['text'][:100]}")
    elif args.cmd == "list":
        memories = list_memories(args.limit, args.type)
        for m in memories:
            print(f"[{m['memory_type']}] {m['timestamp']} — {m['text'][:80]}")
    elif args.cmd == "delete":
        print(json.dumps(delete_memory(args.memory_id), indent=2))
    else:
        parser.print_help()
