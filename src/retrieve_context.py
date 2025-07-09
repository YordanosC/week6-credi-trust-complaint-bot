# src/retrieve_context.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

def load_vector_store(path="vector_store/faiss_index"):
    embedding_model = OpenAIEmbeddings()
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)

def retrieve_relevant_chunks(question, db, k=5):
    return db.similarity_search(question, k=k)
