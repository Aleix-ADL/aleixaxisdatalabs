import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

from qdrant_client import QdrantClient
from src.embeddings import get_embedding

load_dotenv()

COLLECTION = os.getenv("QDRANT_COLLECTION", "taxonomy")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

query = "Burger restaurants in Berlin"
vector = get_embedding(query)

results = client.query_points(
    collection_name=COLLECTION,
    query=vector,
    limit=5
).points

for r in results:
    print(r.payload["name"], r.score)
