import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.embeddings import get_embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 1. Your evaluation dataset
texts = [
    "Frankfurt restaurants",
    "Hotdog stands",
    "Fast food restaurants",
    "Gourmet burger restaurants",
    "Event agencies",
    "Corporate event planners",
    "Email marketing SaaS for ecommerce",
    "CRM software for ecommerce",
    "Dentists",
    "Dental clinics",
    "Physiotherapy clinics",
    "Cybersecurity consulting companies",
    "Managed IT services",
    "Cloud consulting agencies",
    "Yoga studios",
    "Pilates studios",
    "Gyms and fitness centers",
    "Clothing stores",
    "Shoe stores",
    "Sports stores"
]

print("Embedding all items...")
embeddings = [get_embedding(t) for t in texts]

# 2. Compute similarities for each text
for i, text in enumerate(texts):
    sims = []
    for j, other in enumerate(texts):
        if i == j: 
            continue
        sim = cosine_similarity(embeddings[i], embeddings[j])
        sims.append((other, sim))

    # Sort by similarity descending
    sims.sort(key=lambda x: x[1], reverse=True)
    top3 = sims[:3]

    print("\n---")
    print(f"TEXT: {text}")
    print("Top 3 matches:")
    for t, s in top3:
        print(f"  → {t}  (score: {s:.3f})")
