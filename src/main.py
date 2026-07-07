from pathlib import Path

from src.embeddings.vector_store import (
    COLLECTION_NAME,
    PERSIST_DIRECTORY,
    create_vector_store,
)
from src.loaders.pdf_loader import load_pdf
from src.processors.text_splitter import split_documents


def main() -> None:
    """Verify the document indexing pipeline end to end."""
    pdf_path = Path("data/LangChain.pdf")

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


if __name__ == "__main__":
    main()