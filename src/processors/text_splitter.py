"""Utilities for splitting LangChain documents into smaller chunks."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split LangChain documents into smaller chunks.

    Args:
        documents: Documents to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of chunked LangChain Document objects with enriched metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    chunk_indexes: dict[int, int] = {}

    for chunk in chunks:
        page = int(chunk.metadata.get("page", 0))
        chunk_index = chunk_indexes.get(page, 0)
        chunk_indexes[page] = chunk_index + 1

        source = chunk.metadata.get("source")
        filename = Path(str(source)).stem if source else "document"

        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_id"] = (
            f"{filename}_p{page}_c{chunk_index}"
        )

    return chunks