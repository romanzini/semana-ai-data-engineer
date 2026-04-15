"""ShopAgent Day 2 — Ingest reviews JSONL into Qdrant (The Memory)."""

import os
import uuid
from pathlib import Path

import qdrant_client
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.readers.json import JSONReader
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

Settings.llm = Anthropic(model="claude-sonnet-4-20250514")


def ingest_reviews(
    jsonl_path: str | None = None,
    qdrant_url: str | None = None,
    collection_name: str | None = None,
) -> qdrant_client.QdrantClient:
    reviews_dir = PROJECT_ROOT / "gen" / "data" / "reviews"
    qdrant_url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    collection_name = collection_name or os.environ.get("QDRANT_COLLECTION", "shopagent_reviews")
    embed_model_name = os.environ.get("QDRANT_LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    # mcp-server-qdrant names vectors as "fast-{short-name}" (e.g. fast-bge-base-en-v1.5)
    vector_name = "fast-" + embed_model_name.split("/")[-1]

    embed_model = FastEmbedEmbedding(model_name=embed_model_name)

    # Read all reviews*.jsonl files from the directory — new files are picked up automatically
    if jsonl_path:
        jsonl_files = [Path(jsonl_path)]
    else:
        jsonl_files = sorted(reviews_dir.glob("reviews*.jsonl"))

    reader = JSONReader(is_jsonl=True, clean_json=True)
    documents = []
    for f in jsonl_files:
        documents.extend(reader.load_data(input_file=str(f)))
    print(f"Loaded {len(documents)} reviews from {len(jsonl_files)} file(s)")

    texts = [doc.text for doc in documents]
    print("Generating embeddings...")
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)

    client = qdrant_client.QdrantClient(url=qdrant_url)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"Dropped existing collection '{collection_name}'")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={vector_name: VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)},
    )
    print(f"Created collection '{collection_name}' with named vector '{vector_name}'")

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector={vector_name: emb},
            payload={"document": doc.text, **doc.metadata},
        )
        for doc, emb in zip(documents, embeddings)
    ]
    client.upsert(collection_name=collection_name, points=points)
    print(f"Indexed {len(points)} reviews into Qdrant '{collection_name}'")

    return client


if __name__ == "__main__":
    client = ingest_reviews()

    embed_model_name = os.environ.get("QDRANT_LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    vector_name = "fast-" + embed_model_name.split("/")[-1]
    collection_name = os.environ.get("QDRANT_COLLECTION", "shopagent_reviews")
    embed_model = FastEmbedEmbedding(model_name=embed_model_name)

    test_query = "Clientes reclamando de entrega"
    query_vector = embed_model.get_text_embedding(test_query)
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=vector_name,
        limit=5,
    )
    print(f"\nTest query: '{test_query}'")
    for i, point in enumerate(results.points, 1):
        text = point.payload.get("document", "")[:100]
        print(f"  [{i}] score={point.score:.3f}  {text}...")
