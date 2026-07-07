"""End-to-end verification of the RAG pipeline."""

from pathlib import Path

from src.embeddings.vector_store import (
    COLLECTION_NAME,
    PERSIST_DIRECTORY,
    create_vector_store,
)
from src.llm.generator import generate_answer
from src.loaders.pdf_loader import load_pdf
from src.processors.text_splitter import split_documents
from src.prompts.prompt_template import build_prompt
from src.retrievers.semantic_retriever import retrieve_documents


def main() -> None:
    """Verify the complete RAG pipeline end to end."""
    pdf_path = Path("data/LangChain.pdf")
    question = "Who won the FIFA World Cup in 2022?"

    try:
        # Step 1: Load the PDF
        documents = load_pdf(pdf_path)

        # Step 2: Split into chunks
        chunks = split_documents(documents)

        # Step 3: Create the vector store
        create_vector_store(chunks)

        # Step 4: Retrieve relevant chunks
        retrieved_chunks = retrieve_documents(question, k=3)

        # Step 5: Build the prompt
        prompt = build_prompt(question, retrieved_chunks)

        # Step 6: Generate the final answer
        answer = generate_answer(prompt)

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

    print("=" * 60)
    print("RAG Pipeline Verification")
    print("=" * 60)
    print()

    print("PDF loaded successfully")
    print(f"Total pages loaded: {len(documents)}")
    print(f"Total chunks generated: {len(chunks)}")
    print("Vector store created successfully")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Persistence directory: {PERSIST_DIRECTORY}")

    print()
    print("=" * 60)
    print("Question")
    print("=" * 60)
    print(question)

    print()
    print("=" * 60)
    print("Retrieved Chunks")
    print("=" * 60)
    print(f"Top {len(retrieved_chunks)} chunks retrieved.")

    for index, chunk in enumerate(retrieved_chunks, start=1):
        print(f"\nChunk {index}")
        print(f"Chunk ID      : {chunk.metadata.get('chunk_id')}")
        print(f"Page          : {chunk.metadata.get('page')}")
        print(f"Chunk Index   : {chunk.metadata.get('chunk_index')}")
        print(f"Characters    : {len(chunk.page_content)}")
        print(f"Preview       : {chunk.page_content[:200]}...")

    print()
    print("=" * 60)
    print("Generated Answer")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()