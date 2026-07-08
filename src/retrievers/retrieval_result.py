"""Data structures for scored retrieval results."""

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(slots=True)
class RetrievalResult:
    """A single retrieval output containing the document and its similarity score.

    Attributes:
        document: The retrieved LangChain document.
        score: The similarity score returned by the vector store.
    """

    document: Document
    score: float
