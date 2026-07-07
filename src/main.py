from pathlib import Path

from src.loaders.pdf_loader import load_pdf
from src.processors.text_splitter import split_documents


def main() -> None:
    """Verify PDF loading and chunking end to end."""
    pdf_path = Path("data/LangChain.pdf")

    try:
        documents = load_pdf(pdf_path)
    except FileNotFoundError as exc:
        print(f"PDF file not found: {exc}")
        return
    except RuntimeError as exc:
        print(f"Unable to load PDF: {exc}")
        return
    except Exception as exc:  # pragma: no cover - defensive error handling
        print(f"Unexpected error: {exc}")
        return

    chunks = split_documents(documents)

    print("PDF loaded successfully")
    print(f"Total pages loaded: {len(documents)}")
    print(f"Total chunks generated: {len(chunks)}")

    if chunks:
        first_chunk = chunks[0]
        print(f"chunk_id: {first_chunk.metadata.get('chunk_id')}")
        print(f"page: {first_chunk.metadata.get('page')}")
        print(f"chunk_index: {first_chunk.metadata.get('chunk_index')}")
        print(f"character count: {len(first_chunk.page_content)}")
        print("\nPreview:")
        print(first_chunk.page_content[:300])


if __name__ == "__main__":
    main()
