# src/rag_pipeline.py
from src.retrieve_context import load_vector_store, retrieve_relevant_chunks
from src.generate_answer import generate_answer

def rag_pipeline(question, vector_store_path="vector_store/faiss_index"):
    db = load_vector_store(vector_store_path)
    chunks = retrieve_relevant_chunks(question, db)
    answer = generate_answer(question, chunks)
    return answer, chunks
