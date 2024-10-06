import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GITHUB_PERSONAL_TOKEN = os.getenv("GITHUB_PERSONAL_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES_DB_PATH = os.path.join(BASE_DIR, "data", "faces-db")
CLONE_BASE_DIR = os.getenv("CLONE_BASE_DIR")

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

VECTOR_STORE_PROVIDER = "qdrant"
VECTOR_STORE_COLLECTION_NAME = QDRANT_COLLECTION_NAME
VECTOR_STORE_URL = QDRANT_URL
VECTOR_STORE_API_KEY = QDRANT_API_KEY

VERSION = "v1.1"
USER_ID = "divyanshu"
