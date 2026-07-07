from pathlib import Path

from src.loaders.pdf_loader import load_pdf

def main() -> None:
    """Verify that the PDF loader works end to end."""
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

    print("PDF loaded successfully.")
    print(f"Total pages loaded: {len(documents)}")

    if documents:
        first_page = documents[0]
        print(f"First page metadata: {first_page.metadata}")
        preview = first_page.page_content[:300]
        print(f"First page preview: {preview}")


if __name__ == "__main__":
    main()
