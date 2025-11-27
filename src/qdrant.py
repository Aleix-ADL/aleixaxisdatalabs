import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv

load_dotenv()

# Qdrant URL + API key from .env
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Create client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def create_taxonomy_collection(collection_name="taxonomy"):
    """
    Creates Qdrant collection for storing taxonomy embeddings.
    """
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1536,  # embedding size for text-embedding-3-small
            distance=Distance.COSINE
        )
    )
    print(f"Collection '{collection_name}' created successfully!")
