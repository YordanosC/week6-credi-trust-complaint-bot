# main_task2.py
import pandas as pd
from chunk_text import chunk_text
from embed_chunks import get_embedding_model
from create_vector_store import create_vector_store

df = pd.read_csv("../notebooks/cleaned_data.csv")
print("Columns in DataFrame:", df.columns.tolist())
print(df.head())  # optional: preview the data


# Step 3: Chunking
print("🔹 Chunking text...")
chunks = chunk_text()

# Step 4: Embedding
print("🔹 Getting embedding model...")
embedding_model = get_embedding_model()

# Step 5: Index and save
print("🔹 Creating vector store...")
create_vector_store(chunks, embedding_model)
