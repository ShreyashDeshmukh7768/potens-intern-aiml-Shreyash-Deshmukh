import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from src.retrievers.retrieval_result import RetrievalResult
from src.retrievers.semantic_retriever import retrieve_documents


class RetrieveDocumentsTests(unittest.TestCase):
    def test_returns_scored_retrieval_results(self) -> None:
        fake_document = Document(page_content="example", metadata={"source": "test.pdf"})

        class FakeVectorStore:
            def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]:
                self.query = query
                self.k = k
                return [(fake_document, 0.5)]

        with patch("src.retrievers.semantic_retriever.load_vector_store", return_value=FakeVectorStore()):
            results = retrieve_documents("What is RAG?", k=2)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RetrievalResult)
        self.assertEqual(results[0].document, fake_document)
        self.assertEqual(results[0].score, 0.5)

    def test_rejects_empty_query(self) -> None:
        with self.assertRaises(ValueError):
            retrieve_documents("   ", k=1)

    def test_returns_empty_list_for_irrelevant_query(self) -> None:
        fake_document = Document(page_content="example", metadata={"source": "test.pdf"})

        class FakeVectorStore:
            def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]:
                return [(fake_document, 0.95)]

        with patch("src.retrievers.semantic_retriever.load_vector_store", return_value=FakeVectorStore()):
            results = retrieve_documents("Who is Ronaldo?", k=2)

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
