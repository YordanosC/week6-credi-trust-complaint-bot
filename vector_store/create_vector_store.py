# create_vector_store.py
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks, embedding_model, save_path="faiss_index"):
    db = FAISS.from_texts(chunks, embedding_model)
    db.save_local(save_path)
    print(f"✅ Vector store saved to '{save_path}'")
    return db
