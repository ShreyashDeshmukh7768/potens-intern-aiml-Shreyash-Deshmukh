"""Utilities for loading PDF documents."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a single PDF file and return LangChain Document objects.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of LangChain Document objects.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If the PDF cannot be loaded.
    """
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        loader = PyPDFLoader(str(path))
        return loader.load()
    except Exception as exc:
        raise RuntimeError(f"Unable to load PDF '{path}'.") from exc


def load_all_pdfs(data_directory: str | Path) -> list[Document]:
    """Load every PDF from a directory.

    Args:
        data_directory: Directory containing PDF files.

    Returns:
        A combined list of LangChain Document objects from all PDFs.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no PDF files are found.
        RuntimeError: If any PDF fails to load.
    """
    directory = Path(data_directory)

    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in '{directory}'."
        )

    all_documents: list[Document] = []

    for pdf_file in pdf_files:
        documents = load_pdf(pdf_file)
        all_documents.extend(documents)

    return all_documents