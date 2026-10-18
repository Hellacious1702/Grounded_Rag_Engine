from typing import List
from pydantic import BaseModel , Field

class ChunkMetadata(BaseModel):
    chunk_id: str = Field(description="Unique identifier for the chunk, e.g., auth_doc_chunk_0")
    source: str = Field(description="Filename or document origin")
    category: str = Field(description="Topic classification tag")
    word_count: int = Field(ge=0 , description="Number of words in this chunk (must be >= 0)")
    
class DocumentChunk(BaseModel):
    text: str = Field(description="The actual text content of the chunk")
    metadata: ChunkMetadata = Field(description="Nested metadata object associated with this chunk")
    
def create_overlapping_chunks(
    text: str,
    source_name: str,
    category: str,
    chunk_size: int = 200,
    overlap: int = 20
) -> List[DocumentChunk]:
    """
    Splits text into overlapping chunks and validates each chunk against Pydantic schemas.
    """
    words = text.split()
    chunks: List[DocumentChunk] = []
    chunk_index = 0
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = (" ").join(chunk_words)
        
        chunk_id = f"{source_name}_chunk_{chunk_index}"
        
        metadata_object = ChunkMetadata(
            chunk_id = chunk_id,
            source=source_name,
            category=category,
            word_count=len(chunk_words)
        )
        
        chunk_object = DocumentChunk(
            text=chunk_text,
            metadata=metadata_object
        )
        
        chunks.append(chunk_object)
        
        chunk_index += 1
        start += (chunk_size - overlap)
        
    return chunks


if __name__ == "__main__":
    sample_text = (
        "The authentication microservice handles user login and OAuth2 token issuance. "
        "Tokens expire after 60 minutes and must be refreshed using the refresh endpoint. "
        "If a token is compromised, administrators can revoke it using the revocation API. "
        "Database connections use connection pooling with a maximum of 50 concurrent connections. "
        "Idle connections are closed after 300 seconds to conserve memory resources."
    )
    
    test_chunks = create_overlapping_chunks(
        text = sample_text,
        source_name = "auth_service.md",
        category="security",
        chunk_size=15,
        overlap=5
    )
    
    print(f"Generated {len(test_chunks)} Validated Pydantic Chunks: \n")
    for chunk in test_chunks:
        print(f"ID : {chunk.metadata.chunk_id}")
        print(f"Text: \"{chunk.text}\"")
        print(f"Dict Export: {chunk.model_dump()}\n" + "-"*40)
        
        
    