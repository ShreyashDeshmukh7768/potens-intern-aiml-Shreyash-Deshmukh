from pathlib import Path
import time

import streamlit as st

PDF_PATH = Path("data/LangChain.pdf")
RETRIEVER_TYPE = "Semantic Similarity Search"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "documents"
PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_NAME = "gemini-2.5-flash"
EXAMPLE_QUESTIONS = [
    "What is LangChain?",
    "How does LangChain compare to LangGraph?",
    "What are the main benefits of using LangChain?",
]


def configure_page() -> None:
    """Configure the Streamlit page layout and visual theme."""
    st.set_page_config(
        page_title="Explainable RAG Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    style = '''
    <style>
    .stApp {
        background: linear-gradient(135deg, #07111f, #0b1527);
        color: #e6f1ff;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .card {
        background: rgba(14, 27, 46, 0.92);
        border: 1px solid rgba(34, 211, 238, 0.18);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 30px rgba(2, 8, 23, 0.25);
        margin-bottom: 0.8rem;
    }
    .stButton > button {
        border-radius: 999px;
        border: 1px solid rgba(34, 211, 238, 0.35);
        background: linear-gradient(90deg, #0f766e, #2563eb);
        color: white;
    }
    </style>
    '''
    st.markdown(style, unsafe_allow_html=True)


def render_sidebar(state: dict) -> None:
    """Render the sidebar with project configuration and pipeline status."""
    with st.sidebar:
        st.markdown("## 🧠 Explainable RAG")
        st.caption("Production-style retrieval and grounding workflow")

        st.markdown("### System Overview")
        st.write("**LLM:** Gemini 2.5 Flash")
        st.write(f"**Embedding model:** {EMBEDDING_MODEL}")
        st.write("**Vector DB:** ChromaDB")
        st.write(f"**Retriever:** {RETRIEVER_TYPE}")
        st.write(f"**Chunk size:** {CHUNK_SIZE}")
        st.write(f"**Chunk overlap:** {CHUNK_OVERLAP}")

        st.divider()
        st.markdown("### Runtime Snapshot")
        if state.get("documents") is not None:
            st.metric("Loaded documents", len(state["documents"]))
        else:
            st.metric("Loaded documents", 0)
        if state.get("chunks") is not None:
            st.metric("Total chunks", len(state["chunks"]))
        else:
            st.metric("Total chunks", 0)
        if state.get("retrieved_chunks") is not None:
            st.metric("Retrieved chunks", len(state["retrieved_chunks"]))
        else:
            st.metric("Retrieved chunks", 0)

        st.divider()
        st.markdown("### Future Enhancements")
        st.checkbox("Cross Encoder Re-ranking", value=False, disabled=True)
        st.checkbox("Recursive Context Tree Selection (RCTS)", value=False, disabled=True)
        st.checkbox("Contradiction Detection", value=False, disabled=True)
        st.checkbox("Multi-Query Retrieval", value=False, disabled=True)
        st.checkbox("Query Expansion", value=False, disabled=True)


def render_header() -> None:
    """Render the main header and lead copy."""
    st.markdown("# 🔎 Explainable RAG Dashboard")
    st.markdown(
        "A transparent, engineering-focused interface for grounding answers in retrieved evidence."
    )


def render_query_panel(state: dict) -> None:
    """Render the question input, examples, and generation button."""
    st.markdown("## ✍️ Query")
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        question = st.text_area(
            "Ask a question about the indexed PDF",
            value=state.get("question", "What is LangChain?"),
            height=140,
            placeholder="Enter a grounded question...",
        )

        example_cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, example in zip(example_cols, EXAMPLE_QUESTIONS):
            if col.button(example, use_container_width=True):
                state["question"] = example
                st.rerun()

        generate_col, clear_col = st.columns([1, 1])
        with generate_col:
            generate_clicked = st.button("Generate Answer", use_container_width=True)
        with clear_col:
            if st.button("Clear", use_container_width=True):
                state.clear()
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        if generate_clicked:
            state["question"] = question
            run_pipeline(state)


def render_pipeline_visualization() -> None:
    """Render a conceptual RAG pipeline flow."""
    st.markdown("## 🔄 RAG Pipeline")
    stages = [
        ("PDF Loading", "Load and validate the source document."),
        ("Chunking", "Split the document into context windows."),
        ("Embedding", "Create vector representations."),
        ("Semantic Retrieval", "Find relevant chunks for the query."),
        ("Prompt Building", "Assemble grounded context and instructions."),
        ("Gemini Generation", "Generate an answer from the prompt."),
        ("Citation Formatting", "Map evidence back to source chunks."),
    ]

    cols = st.columns(len(stages))
    for col, (title, description) in zip(cols, stages):
        with col:
            st.markdown(
                f"<div class='card'><h4>{title}</h4><p>{description}</p></div>",
                unsafe_allow_html=True,
            )


def render_retrieved_chunks(chunks: list) -> None:
    """Render retrieved chunks in expandable cards."""
    if not chunks:
        return

    st.markdown("## 📦 Retrieved Chunks")
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata or {}
        with st.expander(
            f"Chunk {index}: {metadata.get('chunk_id', f'chunk-{index}')}",
            expanded=index == 1,
        ):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            chunk_cols = st.columns([1, 1, 1, 1])
            chunk_cols[0].metric("Chunk ID", metadata.get("chunk_id", "n/a"))
            chunk_cols[1].metric("Page", metadata.get("page", "n/a"))
            chunk_cols[2].metric("Chunk Index", metadata.get("chunk_index", "n/a"))
            chunk_cols[3].metric("Characters", len(chunk.page_content))
            st.markdown("**Preview**")
            st.write(chunk.page_content[:300])
            st.markdown("**Metadata**")
            st.json(metadata, expanded=False)
            st.markdown("</div>", unsafe_allow_html=True)


def render_answer(answer: str, citations: list[dict]) -> None:
    """Render the generated answer and its citations."""
    st.markdown("## ✅ Grounded Answer")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(answer)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("## 🧾 Citations")
    for citation in citations:
        st.markdown(
            f"<div class='card'><strong>{citation['label']}</strong><br/>{citation['details']}</div>",
            unsafe_allow_html=True,
        )


def render_prompt_viewer(prompt: str) -> None:
    """Render the exact prompt passed to the LLM."""
    with st.expander("Prompt Viewer", expanded=False):
        st.code(prompt, language="text")


def render_analytics(state: dict) -> None:
    """Render latency and corpus statistics."""
    st.markdown("## 📈 Analytics")
    analytics_cols = st.columns(6)
    analytics_cols[0].metric("Retrieval latency", state.get("retrieval_latency", "—"))
    analytics_cols[1].metric("Generation latency", state.get("generation_latency", "—"))
    analytics_cols[2].metric("Total pages", state.get("page_count", 0))
    analytics_cols[3].metric("Total chunks", state.get("chunk_count", 0))
    analytics_cols[4].metric("Retrieved chunks", state.get("retrieved_count", 0))
    analytics_cols[5].metric("Prompt length", state.get("prompt_length", 0))


def build_citations(chunks: list) -> list[dict]:
    """Build citation cards from retrieved chunks."""
    citations: list[dict] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata or {}
        citations.append(
            {
                "label": f"Citation {index}",
                "details": (
                    f"Chunk ID: {metadata.get('chunk_id', 'n/a')} | "
                    f"Page: {metadata.get('page', 'n/a')} | "
                    f"Source: {metadata.get('source', 'n/a')}"
                ),
            }
        )
    return citations


def run_pipeline(state: dict) -> None:
    """Run the RAG pipeline and populate the session state."""
    try:
        from src.embeddings.vector_store import create_vector_store
        from src.llm.generator import generate_answer
        from src.loaders.pdf_loader import load_pdf
        from src.processors.text_splitter import split_documents
        from src.prompts.prompt_template import build_prompt
        from src.retrievers.semantic_retriever import retrieve_documents

        with st.spinner("Running the explainable RAG pipeline..."):
            pipeline_progress = st.progress(0)
            with st.status("Loading PDF and preparing the corpus", expanded=True) as status:
                start_time = time.perf_counter()
                documents = load_pdf(PDF_PATH)
                state["documents"] = documents
                pipeline_progress.progress(0.2)
                st.write("PDF loaded successfully.")

                chunk_start = time.perf_counter()
                chunks = split_documents(documents)
                state["chunks"] = chunks
                pipeline_progress.progress(0.4)
                st.write("Documents were chunked successfully.")

                create_vector_store(chunks)
                pipeline_progress.progress(0.6)
                st.write("Vector store was created successfully.")

                retrieval_start = time.perf_counter()
                retrieved_chunks = retrieve_documents(state.get("question", "What is LangChain?"), k=3)
                state["retrieved_chunks"] = retrieved_chunks
                retrieval_latency = round(time.perf_counter() - retrieval_start, 3)
                state["retrieval_latency"] = f"{retrieval_latency}s"
                pipeline_progress.progress(0.8)
                st.write("Top-k semantic retrieval completed.")

                prompt = build_prompt(state.get("question", "What is LangChain?"), retrieved_chunks)
                state["prompt"] = prompt
                state["prompt_length"] = len(prompt)
                generation_start = time.perf_counter()
                answer = generate_answer(prompt)
                generation_latency = round(time.perf_counter() - generation_start, 3)
                state["answer"] = answer
                state["generation_latency"] = f"{generation_latency}s"
                state["page_count"] = len(documents)
                state["chunk_count"] = len(chunks)
                state["retrieved_count"] = len(retrieved_chunks)
                state["citations"] = build_citations(retrieved_chunks)
                state["total_latency"] = round(time.perf_counter() - start_time, 3)
                pipeline_progress.progress(1.0)
                status.update(label="Pipeline completed", state="complete")
    except FileNotFoundError as exc:
        st.error(f"PDF file not found: {exc}")
    except ValueError as exc:
        st.error(f"Validation error: {exc}")
    except RuntimeError as exc:
        st.error(f"Runtime error: {exc}")
    except Exception as exc:  # pragma: no cover - defensive handling
        st.error(f"Unexpected error: {exc}")


def main() -> None:
    """Render the explainable RAG dashboard interface."""
    configure_page()
    inject_css = None
    del inject_css
    if "question" not in st.session_state:
        st.session_state["question"] = "What is LangChain?"
    if "documents" not in st.session_state:
        st.session_state["documents"] = None
    if "chunks" not in st.session_state:
        st.session_state["chunks"] = None
    if "retrieved_chunks" not in st.session_state:
        st.session_state["retrieved_chunks"] = None
    if "answer" not in st.session_state:
        st.session_state["answer"] = None
    if "prompt" not in st.session_state:
        st.session_state["prompt"] = None
    if "citations" not in st.session_state:
        st.session_state["citations"] = []
    if "page_count" not in st.session_state:
        st.session_state["page_count"] = 0
    if "chunk_count" not in st.session_state:
        st.session_state["chunk_count"] = 0
    if "retrieved_count" not in st.session_state:
        st.session_state["retrieved_count"] = 0
    if "retrieval_latency" not in st.session_state:
        st.session_state["retrieval_latency"] = "—"
    if "generation_latency" not in st.session_state:
        st.session_state["generation_latency"] = "—"
    if "prompt_length" not in st.session_state:
        st.session_state["prompt_length"] = 0

    state = st.session_state
    render_sidebar(state)
    render_header()
    render_query_panel(state)
    render_pipeline_visualization()

    if state.get("answer"):
        render_answer(state["answer"], state.get("citations", []))
        render_prompt_viewer(state.get("prompt", ""))
        render_retrieved_chunks(state.get("retrieved_chunks", []))
        render_analytics(state)


if __name__ == "__main__":
    main()
