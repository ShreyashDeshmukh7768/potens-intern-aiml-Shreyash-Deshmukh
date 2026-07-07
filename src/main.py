from pathlib import Path

from src.embeddings.vector_store import (
    COLLECTION_NAME,
    PERSIST_DIRECTORY,
    create_vector_store,
)
from src.loaders.pdf_loader import load_pdf
from src.processors.text_splitter import split_documents
from src.retrievers.semantic_retriever import retrieve_documents


def main() -> None:
    """Verify the document indexing and retrieval pipeline end to end."""
    pdf_path = Path("data/LangChain.pdf")
    question = "What is LangChain?"

    try:
        documents = load_pdf(pdf_path)
        chunks = split_documents(documents)
        create_vector_store(chunks)
    except FileNotFoundError as exc:
        print(f"PDF file not found: {exc}")
        return
    except ValueError as exc:
        print(f"Validation error: {exc}")
        return
    except RuntimeError as exc:
        print(f"Runtime error: {exc}")
        return
    except Exception as exc:  # pragma: no cover
        print(f"Unexpected error: {exc}")
        return

    print("PDF loaded successfully")
    print(f"Total pages loaded: {len(documents)}")
    print(f"Total chunks generated: {len(chunks)}")
    print("Vector store created successfully")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Persistence directory: {PERSIST_DIRECTORY}")
    print()

    retrieved_chunks = retrieve_documents(question, k=3)

    print("=" * 60)
    print("Semantic Retrieval Test")
    print("=" * 60)
    print(f"Query: {question}")
    print(f"Retrieved {len(retrieved_chunks)} chunks")
    print()

    for index, chunk in enumerate(retrieved_chunks, start=1):
        print("----------------------------------------")
        print(f"Rank: {index}")
        print(f"Chunk ID: {chunk.metadata.get('chunk_id')}")
        print(f"Page: {chunk.metadata.get('page')}")
        print(f"Chunk Index: {chunk.metadata.get('chunk_index')}")
        print(f"Character Count: {len(chunk.page_content)}")
        print(f"Preview: {chunk.page_content[:300]}")
        print()


if __name__ == "__main__":
    main()