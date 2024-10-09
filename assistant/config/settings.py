import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_PERSONAL_TOKEN = os.getenv("GITHUB_PERSONAL_TOKEN")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES_DB_PATH = os.path.join(BASE_DIR, "data", "faces-db")
CLONE_BASE_DIR = os.getenv("CLONE_BASE_DIR")

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

VERSION = "v1.1"
USER_ID = "divyanshu"
