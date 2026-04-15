"""ShopAgent Day 2 — Semantic search on The Memory (Qdrant) without re-indexing."""

import os
from pathlib import Path

import qdrant_client
from dotenv import load_dotenv
from llama_index.embeddings.fastembed import FastEmbedEmbedding

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

QUERIES = [
    "Clientes reclamando de entrega atrasada",
    "Reviews positivos sobre qualidade do produto",
    "Problemas com pagamento ou frete",
]


def build_retriever(
    qdrant_url: str | None = None,
    collection_name: str | None = None,
    similarity_top_k: int = 5,
):
    qdrant_url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    collection_name = collection_name or os.environ.get("QDRANT_COLLECTION", "shopagent_reviews")
    embed_model_name = os.environ.get("QDRANT_LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    # mcp-server-qdrant names vectors as "fast-{short-name}"
    vector_name = "fast-" + embed_model_name.split("/")[-1]

    client = qdrant_client.QdrantClient(url=qdrant_url)
    embed_model = FastEmbedEmbedding(model_name=embed_model_name)
    return client, embed_model, collection_name, vector_name, similarity_top_k


def run_query(client, embed_model, collection_name, vector_name, top_k, query: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"QUERY: {query}")
    print("=" * 60)

    query_vector = embed_model.get_text_embedding(query)
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=vector_name,
        limit=top_k,
    )
    print(f"\nTop {len(results.points)} semantic matches:")
    for i, point in enumerate(results.points, 1):
        text = point.payload.get("document", "")[:140]
        print(f"  [{i}] score={point.score:.3f}  {text}")


if __name__ == "__main__":
    print("Connecting to Qdrant collection 'shopagent_reviews'...")
    client, embed_model, collection_name, embed_model_name, top_k = build_retriever(similarity_top_k=5)
    print("Retriever ready.\n")

    for query in QUERIES:
        run_query(client, embed_model, collection_name, embed_model_name, top_k, query)

    print(f"\n{'=' * 60}")
    print("Notice: 'demorou 15 dias' and 'nao recebi' both match")
    print("'entrega atrasada' — SQL LIKE cannot do this.")
    print("=" * 60)
