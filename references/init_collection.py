#!/usr/bin/env python3
"""
Initialize a Qdrant collection for vector memory.

Usage:
    python init_collection.py [--bot-id my-bot] [--recreate]

Run once per bot to create its collection.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, OptimizersConfig, HnswConfig


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_BASE = os.getenv("QDRANT_COLLECTION", "hermes_memory")


def get_collection_name(bot_id: str) -> str:
    safe = bot_id.replace("-", "_").replace(" ", "_").replace("/", "_")
    return f"{QDRANT_COLLECTION_BASE}_{safe}"


def get_embedding_dim() -> int:
    model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    dims = {
        "BAAI/bge-m3": 1024,
        "BAAI/bge-large-en-v1.5": 1024,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "embed-english-v3.0": 1024,
    }
    return dims.get(model, 1024)


def init_collection(
    bot_id: str,
    host: str = QDRANT_HOST,
    port: int = QDRANT_PORT,
    recreate: bool = False,
) -> None:
    client = QdrantClient(host=host, port=port, prefer_grpc=True)
    collection_name = get_collection_name(bot_id)
    dim = get_embedding_dim()

    collections = [c.name for c in client.get_collections().collections]
    exists = collection_name in collections

    if exists and recreate:
        print(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name=collection_name)
        exists = False

    if not exists:
        print(f"Creating collection: {collection_name} (dim={dim})")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            ),
            optimizers_config=OptimizersConfig(
                indexing_threshold=0,  # Index everything for speed
            ),
            hnsw_config=HnswConfig(
                m=16,
                ef_construct=200,
            ),
        )
        print(f"✓ Collection '{collection_name}' created")
    else:
        print(f"Collection '{collection_name}' already exists (use --recreate to overwrite)")

    # Verify
    info = client.get_collection(collection_name=collection_name)
    print(f"  Vectors: {info.vectors_count}")
    print(f"  Points:  {info.points_count}")
    print(f"  Status:  {info.status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Qdrant collection for vector memory")
    parser.add_argument(
        "--bot-id",
        default=os.getenv("BOT_ID", "default"),
        help="Bot ID (default: from BOT_ID env var)",
    )
    parser.add_argument(
        "--host",
        default=QDRANT_HOST,
        help=f"Qdrant host (default: {QDRANT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=QDRANT_PORT,
        help=f"Qdrant port (default: {QDRANT_PORT})",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection if it exists",
    )
    args = parser.parse_args()

    print(f"Connecting to Qdrant at {args.host}:{args.port}...")
    try:
        init_collection(
            bot_id=args.bot_id,
            host=args.host,
            port=args.port,
            recreate=args.recreate,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
