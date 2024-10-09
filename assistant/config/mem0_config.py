from assistant.config.settings import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
)

config = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URI,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": QDRANT_COLLECTION_NAME,
            "url": QDRANT_URL,
            "api_key": QDRANT_API_KEY,
        },
    },
    "version": "v1.1",
}
