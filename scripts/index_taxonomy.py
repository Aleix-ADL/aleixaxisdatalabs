import sys
import os

# Allow imports from /src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from datetime import datetime, timezone

from src.taxonomy_loader import load_taxonomy
from src.embeddings import get_embedding, get_embedding_batch
from src.qdrant import create_taxonomy_collection


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
MODEL = os.getenv("EMBEDDING_MODEL")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def index_taxonomy():
    # Create collection
    create_taxonomy_collection(COLLECTION_NAME)

    # Load taxonomy nodes
    print("Loading taxonomy...")
    items = load_taxonomy()
    print(f"Loaded {len(items)} nodes.")

    texts = [item["text"] for item in items]
    embeddings = get_embedding_batch(texts)

    points = []

    for i, (item, vector) in enumerate(zip(items, embeddings)):
        payload = {
            **item,
            "taxonomy_id": item["id"],                # original string ID
            "model": MODEL,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

        points.append(PointStruct(
            id=i,                # numeric ID required by Qdrant
            vector=vector,
            payload=payload
        ))

    # Old code no batch embedding 
        # for i, item in enumerate(items):

        #     vector = get_embedding(item["text"])

        #     # Create payload with metadata
        #     payload = item.copy()  # keep all original fields
        #     payload.update({
        #         "taxonomy_id": item["id"],
        #         "model": MODEL,
        #         "updated_at": datetime.now(timezone.utc).isoformat()
        #     })

        #     points.append(
        #         PointStruct(
        #             id=i,  # <-- FIXED: numeric ID, required by Qdrant
        #             vector=vector,
        #             payload=payload  # include full taxonomy item
        #         )
        #     )

    print(f"Inserting {len(points)} items into Qdrant...")

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print("Done! Taxonomy has been indexed.")


if __name__ == "__main__":
    index_taxonomy()
