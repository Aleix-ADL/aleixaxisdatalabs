import pytest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from src.embeddings import get_embedding
from src.qdrant import client  # your initialized Qdrant client


COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "taxonomy")


@pytest.mark.parametrize("query, expected_top", [
    ("Frankfurt restaurants", "Restaurant"),
    ("Gourmet burgers", "Burger restaurant"),
    ("Hotdogs stands", "Hotdog restaurant"),
    ("Sushi places", "Sushi restaurant"),
    ("Italian food", "Italian restaurant"),
])
def test_taxonomy_search_accuracy(query, expected_top):
    """
    Tests if Qdrant returns the correct taxonomy category for a given query.
    """

    # 1. Embed the query
    vector = get_embedding(query)

    # 2. Search Qdrant (top 5)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=5
    ).points


    assert len(results) > 0, "Qdrant returned no results."

    top_result_name = results[0].payload.get("name")

    print("\nQuery:", query)
    print("Expected:", expected_top)
    print("Top match:", top_result_name)

    # 3. Check if correct taxonomy is ranked #1
    assert expected_top.lower() in top_result_name.lower(), (
        f"Query '{query}' should match '{expected_top}' "
        f"but got '{top_result_name}'"
    )
