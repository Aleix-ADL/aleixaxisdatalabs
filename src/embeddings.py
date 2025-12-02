import os
import time
from dotenv import load_dotenv
import openai
from typing import Optional

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
MAX_INPUT_LENGTH = int(os.getenv("EMBEDDING_MAX_LENGTH", 5000))  # characters
MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", 5))
BASE_BACKOFF = float(os.getenv("EMBEDDING_BACKOFF", 1.0))  # seconds

# Initialize client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Internal error types ---
class EmbeddingServiceError(Exception):
    """Base exception for embedding service errors."""
    pass

class RateLimitError(EmbeddingServiceError):
    """Raised when API rate limit is exceeded."""
    pass

class NetworkError(EmbeddingServiceError):
    """Raised for transient network errors."""
    pass

class TimeoutError(EmbeddingServiceError):
    """Raised when the request times out."""
    pass


# --- Helper functions ---
def _truncate(text: str) -> str:
    """Truncation strategy: take first MAX_INPUT_LENGTH characters."""
    return text[:MAX_INPUT_LENGTH] if len(text) > MAX_INPUT_LENGTH else text

# --- Main function ---
def get_embedding(text: str, model: Optional[str] = None) -> list[float]:
    """
    Returns an embedding vector for the given text with:
    - Truncation
    - Retry on transient errors
    - Exponential backoff
    - Internal error types
    """
    model_to_use = model or DEFAULT_EMBEDDING_MODEL
    text = _truncate(text)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(
                model=model_to_use,
                input=text
            )
            return response.data[0].embedding

        except openai.error.RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise RateLimitError("Rate limit exceeded and max retries reached.") from e

        except openai.error.Timeout as e:
            if attempt == MAX_RETRIES:
                raise TimeoutError("Request timed out and max retries reached.") from e

        except openai.error.APIConnectionError as e:
            if attempt == MAX_RETRIES:
                raise NetworkError("Transient network error and max retries reached.") from e

        except Exception as e:
            # Catch-all for unexpected errors
            raise EmbeddingServiceError("Unexpected error in embedding service.") from e

        # Exponential backoff before retrying
        sleep_time = BASE_BACKOFF * (2 ** (attempt - 1))
        time.sleep(sleep_time)

from typing import List, Optional

def get_embedding_batch(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """
    Returns embeddings for a list of texts using OpenAI.
    - Preserves truncation, retries, exponential backoff, and internal error types.
    """
    model_to_use = model or DEFAULT_EMBEDDING_MODEL
    texts = [_truncate(text) for text in texts]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(model=model_to_use, input=texts)
            return [r.embedding for r in response.data]

        except openai.error.RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise RateLimitError("Rate limit exceeded and max retries reached.") from e

        except openai.error.Timeout as e:
            if attempt == MAX_RETRIES:
                raise TimeoutError("Request timed out and max retries reached.") from e

        except openai.error.APIConnectionError as e:
            if attempt == MAX_RETRIES:
                raise NetworkError("Transient network error and max retries reached.") from e

        except Exception as e:
            raise EmbeddingServiceError("Unexpected error in batch embedding.") from e

        # Exponential backoff before retrying
        sleep_time = BASE_BACKOFF * (2 ** (attempt - 1))
        time.sleep(sleep_time)
