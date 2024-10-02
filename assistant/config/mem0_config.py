from assistant.config.settings import QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL

config = {
    "llm": {
        "provider": "litellm",
        "config": {
            "model": "gemini/gemini-1.5-flash-002",
            "temperature": 0.2,
            "max_tokens": 1024,
        },
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
            "embedding_dims": 768,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": QDRANT_COLLECTION_NAME,
            "url": QDRANT_URL,
            "api_key": QDRANT_API_KEY,
            "embedding_model_dims": 768,
        },
    },
    "version": "v1.1",
}
