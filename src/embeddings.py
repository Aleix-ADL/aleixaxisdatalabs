import os
from dotenv import load_dotenv
from openai import OpenAI

# load environment variables
load_dotenv()

# initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """
    Returns an embedding vector for a given input text using OpenAI.
    """
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding
