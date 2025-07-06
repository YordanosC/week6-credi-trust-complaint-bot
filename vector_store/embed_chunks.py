# embed_chunks.py
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings  # Uncomment if using HF

def get_embedding_model():
    # OpenAI (requires API key set via env)
    return OpenAIEmbeddings()

    # For HuggingFace (optional):
    # return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
