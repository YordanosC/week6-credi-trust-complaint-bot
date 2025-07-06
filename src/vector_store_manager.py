from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2, IndexIDMap # For FAISS, more robust for IDs
import numpy as np
import pandas as pd
import os
import pickle # To save and load FAISS index

class VectorStoreManager:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2',
                 chunk_size=500, chunk_overlap=50):
        self.embedding_model = SentenceTransformer(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len, # Use character length for splitting
            add_start_index=True,
        )
        self.vector_store = None
        self.document_map = {} # To store original text and metadata for each chunk ID

    def create_chunks(self, text, metadata):
        """Creates text chunks from a single document with associated metadata."""
        chunks = self.text_splitter.create_documents([text])
        # Add original metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(metadata) # Combine existing (start_index) with new metadata
            chunk.metadata['chunk_id'] = f"{metadata['Complaint ID']}_{i}" # Unique ID for each chunk
        return chunks

    def embed_and_index(self, df_cleaned, id_column='Complaint ID',
                        product_column='Product', text_column='cleaned_narrative'):
        """
        Processes a DataFrame of cleaned complaints, creates embeddings,
        and builds the FAISS vector store.
        """
        print("Starting embedding and indexing process...")
        all_embeddings = []
        current_id = 0 # Unique ID for FAISS index (sequential)

        for index, row in df_cleaned.iterrows():
            original_text = row[text_column]
            metadata = {
                'Complaint ID': row[id_column],
                'Product': row[product_column],
                # Add other relevant metadata if needed
            }
            chunks = self.create_chunks(original_text, metadata)

            for chunk in chunks:
                chunk_text = chunk.page_content
                embedding = self.embedding_model.encode(chunk_text)
                all_embeddings.append(embedding)

                # Store the actual chunk text and metadata associated with a simple ID
                self.document_map[current_id] = {
                    'text': chunk_text,
                    'metadata': chunk.metadata
                }
                current_id += 1

        if not all_embeddings:
            print("No embeddings generated. Check input data or chunking.")
            return

        # Convert list of embeddings to a NumPy array
        embeddings_np = np.array(all_embeddings).astype('float32')

        # Create FAISS index
        dimension = embeddings_np.shape[1]
        self.vector_store = IndexIDMap(IndexFlatL2(dimension)) # Use IndexIDMap for custom IDs
        self.vector_store.add_with_ids(embeddings_np, np.arange(len(embeddings_np))) # Add with sequential IDs

        print(f"FAISS index created with {len(all_embeddings)} embeddings.")
        print(f"FAISS index size: {self.vector_store.ntotal}")


    def save_vector_store(self, faiss_path, doc_map_path):
        """Saves the FAISS index and the document map."""
        if self.vector_store is None:
            print("Vector store not created yet. Call embed_and_index() first.")
            return

        os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
        with open(faiss_path, 'wb') as f:
            pickle.dump(self.vector_store, f)
        print(f"FAISS index saved to {faiss_path}")

        with open(doc_map_path, 'wb') as f:
            pickle.dump(self.document_map, f)
        print(f"Document map saved to {doc_map_path}")

    def load_vector_store(self, faiss_path, doc_map_path):
        """Loads the FAISS index and the document map."""
        try:
            with open(faiss_path, 'rb') as f:
                self.vector_store = pickle.load(f)
            print(f"FAISS index loaded from {faiss_path}")

            with open(doc_map_path, 'rb') as f:
                self.document_map = pickle.load(f)
            print(f"Document map loaded from {doc_map_path}")
            return True
        except FileNotFoundError:
            print(f"Error: Vector store files not found at {faiss_path} or {doc_map_path}.")
            return False

    def retrieve_documents(self, query_text, top_k=5, product_filter=None):
        """
        Retrieves top_k most relevant document chunks based on a query.
        Optionally filters by product category.
        Returns a list of dictionaries with chunk text and metadata.
        """
        if self.vector_store is None:
            print("Vector store not loaded. Call load_vector_store() first.")
            return []

        query_embedding = self.embedding_model.encode(query_text).astype('float32')
        query_embedding = np.array([query_embedding]) # FAISS expects a 2D array

        # Perform the search
        distances, faiss_ids = self.vector_store.search(query_embedding, top_k*5) # Retrieve more than top_k for filtering

        retrieved_chunks = []
        for i, faiss_id in enumerate(faiss_ids[0]): # faiss_ids is 2D, get the first row
            if faiss_id == -1: # FAISS returns -1 for empty slots
                continue

            doc_info = self.document_map.get(faiss_id)
            if doc_info:
                # Apply product filter if specified
                if product_filter and doc_info['metadata'].get('Product') != product_filter:
                    continue # Skip this chunk if it doesn't match the filter

                retrieved_chunks.append({
                    'text': doc_info['text'],
                    'metadata': doc_info['metadata']
                })
                if len(retrieved_chunks) >= top_k: # Stop once top_k are found after filtering
                    break
        return retrieved_chunks

# Example of how this class would be used
# if __name__ == "__main__":
#     # First, ensure data is processed
#     from data_processor import ComplaintDataProcessor
#     processor = ComplaintDataProcessor('../data/complaints.csv')
#     df_cleaned = processor.load_data()
#     target_products = ['Credit card', 'Personal loan', 'Buy Now, Pay Later (BNPL)', 'Savings account', 'Money transfer']
#     df_cleaned = processor.filter_data(target_products)
#     df_cleaned = processor.apply_cleaning()
#
#     manager = VectorStoreManager()
#     manager.embed_and_index(df_cleaned)
#     manager.save_vector_store('../vector_store/faiss_index.bin', '../vector_store/doc_map.pkl')
#
#     # To load and use later:
#     # loaded_manager = VectorStoreManager()
#     # loaded_manager.load_vector_store('../vector_store/faiss_index.bin', '../vector_store/doc_map.pkl')
#     # chunks = loaded_manager.retrieve_documents("Why are people unhappy with BNPL?")
#     # for chunk in chunks:
#     #     print(chunk['text'][:100], chunk['metadata']['Product'])