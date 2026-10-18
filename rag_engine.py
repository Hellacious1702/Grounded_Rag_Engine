import os 
from openai import OpenAI
from chunker import create_overlapping_chunks
from vector_store import add_chunks_to_vector_db , search_vector_db
from dotenv import load_dotenv

load_dotenv()

nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_NIM_API_KEY")
)

def generate_grounded_response(user_query: str , max_distance_threshold: float = 0.60):
    
    raw_results = search_vector_db(query=user_query , top_k=3)
    
    docs = raw_results['documents'][0]
    metas = raw_results['metadatas'][0]
    distances = raw_results['distances'][0]
    
    valid_chunks = []
    
    for doc , meta , dist in zip(docs,metas,distances):
        if dist < max_distance_threshold:
            valid_chunks.append(
                {
                    "text" : doc,
                    "chunk_id" : meta["chunk_id"],
                    "source" : meta["source"],
                    "distance" : dist
                }
            )
            
    if not valid_chunks:
        return "INSUFFICIENT_CONTEXT: No relevant knowledge found in database." , []
    
    formatted_context = ""

    for c in valid_chunks:
        formatted_context += f"--- BEGIN CHUNK [{c['chunk_id']}] (Source: {c['source']}) ---\n"
        formatted_context += f"{c['text']}\n"
        formatted_context += f"--- END CHUNK ---\n\n"
        
    system_prompt = """
        You are a Technical Support AI. Answer the user's question using ONLY the provided Context.

        STRICT GROUNDING RULES:
        1. Every factual claim MUST include an inline citation of the Chunk ID, e.g., [sys_spec.md_chunk_0].
        2. Do NOT use external knowledge or make assumptions.
        3. If the answer cannot be directly proven by the Context, respond ONLY with: "INSUFFICIENT_CONTEXT".
    """
    
    user_prompt = f"Context:\n{formatted_context}\nQuestion: {user_query}"
    
    response = nim_client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=[
            {
                "role":"system",
                "content":system_prompt
            },
            {
                "role":"user",
                "content":user_prompt
            }
        ],
        temperature=0.0
    )
    
    answer = response.choices[0].message.content
    return answer , valid_chunks

if __name__ == "__main__":
    # Sample Knowledge Base
    kb_text = (
        "The microservice architecture utilizes PostgreSQL version 15.4 for primary relational storage. "
        "Max database connections are capped at 100 for production clusters. "
        "User sessions use OAuth2 Bearer tokens which expire exactly 60 minutes after generation. "
        "Password reset requests send a single-use token expiring after 15 minutes."
    )

    print("--- Step 1: Ingesting Knowledge Base ---")
    chunks = create_overlapping_chunks(
        text=kb_text,
        source_name="sys_spec.md",
        category="architecture",
        chunk_size=20,
        overlap=5
    )
    add_chunks_to_vector_db(chunks)

    # Test 1: Grounded Query (Should succeed with citations)
    q1 = "When do OAuth2 bearer tokens expire?"
    print(f"\n--- Test 1: Query: '{q1}' ---")
    ans1, ctx1 = generate_grounded_response(q1)
    print(f"LLM Answer:\n{ans1}")

    # Test 2: Out-Of-Domain Query (Should trigger INSUFFICIENT_CONTEXT)
    q2 = "What is the policy for employee hardware refunds?"
    print(f"\n--- Test 2: Query: '{q2}' ---")
    ans2, ctx2 = generate_grounded_response(q2)
    print(f"LLM Answer:\n{ans2}")
    
    
