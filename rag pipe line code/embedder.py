import faiss
import numpy as np
import os
import pickle

def embed_chunks(chunks, embedder):
    return embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

def store_embeddings_faiss(embeddings, chunks, index_path, mapping_path):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)

    with open(mapping_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"✅ Stored FAISS index at: {index_path}\n✅ Stored chunk mapping at: {mapping_path}")

def load_embeddings_faiss(index_path, mapping_path):
    index = faiss.read_index(index_path)
    with open(mapping_path, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks
