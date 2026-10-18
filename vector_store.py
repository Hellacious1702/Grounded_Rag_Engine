import chromadb
from chromadb.utils import embedding_functions
from chunker import create_overlapping_chunks , DocumentChunk
from typing import List

local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="tech_knowledge_base",
    embedding_function=local_ef,
    metadata={"hnsw:space": "cosine"}
)

def add_chunks_to_vector_db(chunks: List[DocumentChunk]):
    """
    Extracts text, metadata dicts, and unique IDs from Pydantic objects
    and indexes them into ChromaDB.
    """
    
    texts = [chunk.text for chunk in chunks]
    
    metadatas = [chunk.metadata.model_dump() for chunk in chunks]
    
    ids = [chunk.metadata.chunk_id for chunk in chunks]
    
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Successfully indexed {len(chunks)} Pydantic chunks into ChromaDB.")
    
def search_vector_db(query:str,top_k:int = 2):
    """
    Converts query text to a vector and retrieves top_k closest matching chunks.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents","metadatas","distances"]
    )
    return results


if __name__ == "__main__":
    sample_text=(
        "The authentication microservice handles user login and OAuth2 token issuance. "
        "Tokens expire after 60 minutes and must be refreshed using the refresh endpoint. "
        "If a token is compromised, administrators can revoke it using the revocation API. "
        "Database connections use connection pooling with a maximum of 50 concurrent connections. "
        "Idle connections are closed after 300 seconds to conserve memory resources."
    )
    
    pydantic_chunks = create_overlapping_chunks(
        text=sample_text,
        source_name="system_architecture.md",
        category="SystemDesign",
        chunk_size=20,
        overlap=5
    )
    
    print("\n----Step 1: Ingesting Chunks -----")
    add_chunks_to_vector_db(pydantic_chunks)
    
    test_query = "How long do user session keys stay active before expiring?"
    print(f"\n--- Step 2: Querying DB for: '{test_query}' ---")
    
    search_results = search_vector_db(test_query , top_k=2)
    
    docs = search_results["documents"][0]
    metas = search_results["metadatas"][0]
    distances = search_results["distances"][0]
    
    for i in range(len(docs)):
        print(f"\nMatch Rank {i+1}")
        print(f"Chunk ID : {metas[i]['chunk_id']}")
        print(f"Distance : {distances[i]:.4f} (Lower = More Similar)")
        print(f"Text     : \"{docs[i]}\"")
        