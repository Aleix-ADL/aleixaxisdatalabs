import sys
import os

# Allow imports from /src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from src.taxonomy_loader import load_taxonomy
from src.embeddings import get_embedding
from src.qdrant import create_taxonomy_collection


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "taxonomy")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def index_taxonomy():
    create_taxonomy_collection(COLLECTION_NAME)
    print("Loading taxonomy...")
    items = load_taxonomy()

    points = []

    for i, item in enumerate(items):
        # vector = get_embedding(item["name"])  # embedding text
        vector = get_embedding(item["text"])


        points.append(
            PointStruct(
                id=i,  # <-- FIXED: numeric ID, required by Qdrant
                vector=vector,
                payload=item  # include full taxonomy item
            )
        )

    print(f"Inserting {len(points)} items into Qdrant...")

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print("Done! Taxonomy has been indexed.")


if __name__ == "__main__":
    index_taxonomy()
