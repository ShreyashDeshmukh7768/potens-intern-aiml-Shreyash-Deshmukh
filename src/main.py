from pathlib import Path

from src.citations.formatter import format_answer
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
    """Run the complete RAG pipeline end to end."""

    pdf_path = Path("data/LangChain.pdf")
    question = "how to make dosa?"

    try:
        # Step 1: Load PDF
        documents = load_pdf(pdf_path)

        # Step 2: Split into chunks
        chunks = split_documents(documents)

        # Step 3: Build vector database
        create_vector_store(chunks)

        # Step 4: Retrieve relevant chunks
        retrieved_chunks = retrieve_documents(question, k=3)

        # Step 5: Build prompt
        prompt = build_prompt(question, retrieved_chunks)

        # Step 6: Generate answer
        answer = generate_answer(prompt)

        # Step 7: Attach citations
        final_response = format_answer(answer, retrieved_chunks)

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

    print("=" * 70)
    print("RAG Pipeline Summary")
    print("=" * 70)
    print(f"PDF: {pdf_path.name}")
    print(f"Total Pages           : {len(documents)}")
    print(f"Total Chunks          : {len(chunks)}")
    print(f"Vector Collection     : {COLLECTION_NAME}")
    print(f"Persistence Directory : {PERSIST_DIRECTORY}")
    print()

    print("=" * 70)
    print("User Question")
    print("=" * 70)
    print(question)
    print()

    print("=" * 70)
    print(f"Top {len(retrieved_chunks)} Retrieved Chunks")
    print("=" * 70)

    for index, chunk in enumerate(retrieved_chunks, start=1):
        print(f"Chunk {index}")
        print(f"Chunk ID      : {chunk.metadata.get('chunk_id')}")
        print(f"Page          : {chunk.metadata.get('page')}")
        print(f"Chunk Index   : {chunk.metadata.get('chunk_index')}")
        print(f"Characters    : {len(chunk.page_content)}")
        print(
            f"Preview       : "
            f"{chunk.page_content[:200].replace(chr(10), ' ')}..."
        )
        print()

    print("=" * 70)
    print("Grounded Response")
    print("=" * 70)
    print(final_response)


if __name__ == "__main__":
    main()