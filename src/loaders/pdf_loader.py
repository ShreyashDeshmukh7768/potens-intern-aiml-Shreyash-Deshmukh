from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF file and return LangChain Document objects.

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