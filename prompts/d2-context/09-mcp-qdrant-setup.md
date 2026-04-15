Configure MCP Qdrant to connect Claude Code to The Memory:

Add to your Claude Code MCP settings:
```json
{
  "mcpServers": {
    "qdrant": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "shopagent_reviews",
        "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"
      }
    }
  }
}
```

> **Importante:** `QDRANT_LOCAL_EMBEDDING_MODEL` deve usar o mesmo modelo configurado em `ingest_reviews.py` (`FastEmbedEmbedding`). O `mcp-server-qdrant` nomeia os vetores como `fast-{short-name}` — ex: `BAAI/bge-base-en-v1.5` → `fast-bge-base-en-v1.5`. O `ingest_reviews.py` aplica essa mesma convenção ao criar a coleção, garantindo compatibilidade.

Then ask Claude Code: "Quais clientes reclamam de entrega atrasada?"
Claude should search Qdrant semantically and return real review matches.
