import streamlit as st

from chunker import create_overlapping_chunks
from rag_engine import generate_grounded_response
from vector_store import collection

st.set_page_config(page_title="Grounded RAG Knowledge Engine", page_icon="🔎", layout="wide")

DEFAULT_KB = (
    "The microservice architecture utilizes PostgreSQL version 15.4 for primary relational storage. "
    "Max database connections are capped at 100 for production clusters. "
    "User sessions use OAuth2 Bearer tokens which expire exactly 60 minutes after generation. "
    "Password reset requests send a single-use token expiring after 15 minutes."
)


def index_text(text: str, source: str, category: str, chunk_size: int, overlap: int) -> int:
    chunks = create_overlapping_chunks(text, source, category, chunk_size, overlap)
    collection.upsert(
        documents=[c.text for c in chunks],
        metadatas=[c.metadata.model_dump() for c in chunks],
        ids=[c.metadata.chunk_id for c in chunks],
    )
    return len(chunks)


if "indexed" not in st.session_state:
    index_text(DEFAULT_KB, "sys_spec.md", "architecture", 20, 5)
    st.session_state.indexed = ["sys_spec.md"]

st.title("🔎 Grounded RAG Knowledge Engine")
st.caption(
    "Overlapping chunks → Pydantic validation → MiniLM embeddings → ChromaDB (cosine) → "
    "distance filter → NVIDIA NIM with chunk-ID citations"
)

with st.sidebar:
    st.header("Knowledge base")
    st.write("Indexed:", ", ".join(st.session_state.indexed))
    st.write(f"Chunks in DB: **{collection.count()}**")

    with st.expander("Add a document"):
        src = st.text_input("Source name", "my_doc.md")
        cat = st.text_input("Category", "general")
        body = st.text_area("Text", height=140)
        c1, c2 = st.columns(2)
        size = c1.number_input("Chunk size (words)", 5, 500, 40)
        ovl = c2.number_input("Overlap (words)", 0, 100, 8)
        if st.button("Index document") and body.strip() and ovl < size:
            n = index_text(body, src, cat, int(size), int(ovl))
            if src not in st.session_state.indexed:
                st.session_state.indexed.append(src)
            st.success(f"Indexed {n} chunks from {src}")
            st.rerun()

    st.header("Retrieval")
    threshold = st.slider("Max cosine distance", 0.1, 1.0, 0.60, 0.05)

st.subheader("Ask a question")
examples = ["When do OAuth2 bearer tokens expire?", "What is the policy for employee hardware refunds?"]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state.q = ex

question = st.text_input("Question", key="q", placeholder="Ask about the indexed documents")

if question:
    with st.spinner("Retrieving and generating..."):
        answer, chunks = generate_grounded_response(question, max_distance_threshold=threshold)

    if answer.strip().startswith("INSUFFICIENT_CONTEXT"):
        st.warning(answer)
    else:
        st.success(answer)

    st.subheader(f"Retrieved context ({len(chunks)} chunks passed the filter)")
    for c in chunks:
        with st.expander(f"[{c['chunk_id']}] · distance {c['distance']:.3f}"):
            st.write(c["text"])
            st.caption(f"Source: {c['source']}")
