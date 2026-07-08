from pathlib import Path

from src.citations.formatter import format_answer
from src.embeddings.vector_store import (
    COLLECTION_NAME,
    PERSIST_DIRECTORY,
    create_vector_store,
)
from src.llm.generator import generate_answer
from src.loaders.pdf_loader import load_all_pdfs
from src.processors.text_splitter import split_documents
from src.prompts.prompt_template import build_prompt
from src.retrievers.semantic_retriever import retrieve_documents


def main() -> None:
    """Run the complete multi-document RAG pipeline end to end."""

    data_directory = Path("data")
    question = "How does Retrieval-Augmented Generation (RAG) work?"

    try:
        # Step 1: Load all PDFs
        documents = load_all_pdfs(data_directory)

        # Step 2: Split documents into chunks
        chunks = split_documents(documents)

        # Step 3: Create vector database
        create_vector_store(chunks)

        # Step 4: Retrieve relevant chunks
        retrieved_chunks = retrieve_documents(question, k=3)

        # Step 5: Build prompt
        prompt = build_prompt(question, retrieved_chunks)

        # Step 6: Generate answer
        answer = generate_answer(prompt)

        # Step 7: Format answer with citations
        final_response = format_answer(answer, retrieved_chunks)

    except FileNotFoundError as exc:
        print(f"File not found: {exc}")
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

    pdf_files = sorted(data_directory.glob("*.pdf"))

    print("=" * 70)
    print("MULTI-DOCUMENT RAG PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Data Directory        : {data_directory}")
    print(f"PDFs Indexed          : {len(pdf_files)}")
    print(f"Total Pages Loaded    : {len(documents)}")
    print(f"Total Chunks Created  : {len(chunks)}")
    print(f"Vector Collection     : {COLLECTION_NAME}")
    print(f"Persistence Directory : {PERSIST_DIRECTORY}")
    print()

    print("=" * 70)
    print("USER QUESTION")
    print("=" * 70)
    print(question)
    print()

    print("=" * 70)
    print(f"SEMANTIC RETRIEVAL RESULTS (Top {len(retrieved_chunks)})")
    print("=" * 70)

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.metadata

        print(f"Retrieved Result #{index}")
        print("-" * 70)
        print(f"Source File   : {Path(metadata.get('source', 'Unknown')).name}")
        print(f"Chunk ID      : {metadata.get('chunk_id')}")
        print(f"Page          : {metadata.get('page')}")
        print(f"Chunk Index   : {metadata.get('chunk_index')}")
        print(f"Characters    : {len(chunk.page_content)}")
        print(
            "Preview       : "
            f"{chunk.page_content[:200].replace(chr(10), ' ')}..."
        )
        print()

    print("=" * 70)
    print("PROMPT STATISTICS")
    print("=" * 70)
    print(f"Prompt Length : {len(prompt)} characters")
    print()

    print("=" * 70)
    print("GROUNDED RESPONSE")
    print("=" * 70)
    print(final_response)
    print()

    print("=" * 70)
    print("RESPONSE STATISTICS")
    print("=" * 70)
    print(f"Response Length : {len(answer)} characters")


if __name__ == "__main__":
    main()