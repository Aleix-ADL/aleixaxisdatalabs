# Embedding Service Usage Guidelines

## 1. When to Call the Embedding Service
Use the embedding service to obtain vector representations of text for tasks such as:

- Semantic search
- Text clustering
- Similarity comparisons
- Recommendation systems

**Best practices:**

- Avoid excessive calls in tight loops; batch requests or cache embeddings when possible.
- Use embeddings only when you need semantic understanding or similarity scoring.

---

## 2. Input Length Limits
- Maximum input length is determined by `MAX_INPUT_LENGTH` (default: 5000 characters, configurable via environment variable).
- Longer texts are automatically truncated.
- For very long documents, consider splitting into smaller chunks and embedding each chunk separately.

---

## 3. Cost Considerations
- Each embedding request may incur a cost depending on the provider and model used (e.g., OpenAI charges per 1,000 tokens).
- Larger models produce higher-dimensional embeddings and are more expensive.
- Cost-saving tips:
  - Use smaller embedding models when high precision is not required.
  - Cache embeddings for repeated or common texts.
  - Minimize unnecessary calls to the embedding service.

---

## 4. Example Usage

```python
from embedding_service import get_embedding

text = "This is a sample document."

# Get embedding using the default model
embedding_vector = get_embedding(text)

# Get embedding using a specific model
embedding_vector_small = get_embedding(text, model="text-embedding-3-small")

print(len(embedding_vector))  # Example: 1536 floats
