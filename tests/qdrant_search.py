import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from src.embeddings import get_embedding

load_dotenv()

COLLECTION = os.getenv("QDRANT_COLLECTION", "taxonomy_index")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

query = "Restaurants in Berlin"
vector = get_embedding(query)

results = client.search(
    collection_name=COLLECTION,
    query_vector=vector,
    limit=5
)

for r in results:
    print(r.payload["name"], r.score)
