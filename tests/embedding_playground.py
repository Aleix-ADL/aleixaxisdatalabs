import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.embeddings import get_embedding, get_embedding_batch
from itertools import combinations


TEXTS = [
    "Pasta restaurants",
    "Hotdog stands",
    "Event agencies",
    "Email marketing SaaS for ecommerce",
    "Burger places",
    "Sushi restaurants near me",
    "Italian restaurants",
    "Wedding planners",
    "CRM SaaS for sales teams",
    "SaaS analytics platform",
    "Chicago pizza places",
    "Fast food",
    "Taco restaurants",
    "Enterprise recruiting software",
    "Local bakeries",
    "Steakhouse restaurants",
    "Yoga studios",
    "Ecommerce marketing automation tools",
    "PR agencies for tech startups",
    "Comedy events"
]

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def run_evaluation():
    print("Embedding all texts...")
    # embeddings = [get_embedding(t) for t in TEXTS]
    embeddings = get_embedding_batch(TEXTS)
    
    assert len(embeddings) == len(TEXTS), "Mismatch: embeddings count does not match texts!"

    # Validation: embedding dimensions
    expected_dim = len(embeddings[0])
    for emb in embeddings:
        assert len(emb) == expected_dim, "Mismatch: embedding dimension inconsistent!"

    report = []

    for i, text in enumerate(TEXTS):
        sims = []
        for j, other in enumerate(TEXTS):
            if i == j:
                continue
            score = cosine(embeddings[i], embeddings[j])
            sims.append((other, score))

        sims.sort(key=lambda x: x[1], reverse=True)
        top3 = sims[:3]

        report.append({
            "text": text,
            "top3": top3,
        })

    return report


def generate_markdown_output(report):
    lines = ["# Embedding Similarity Report\n"]

    for item in report:
        lines.append(f"## Query: **{item['text']}**")
        for other, score in item["top3"]:
            lines.append(f"- {score:.4f} — {other}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    report = run_evaluation()
    md = generate_markdown_output(report)

    with open("embedding_similarity_report.md", "w") as f:
        f.write(md)

    print("\nReport saved to embedding_similarity_report.md")
