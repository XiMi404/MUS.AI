"""Tests for vector search functionality."""

import pytest
import tempfile
import json
from pathlib import Path

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore, MuseumDocument
from src.ingestion import ingest_data


class TestVectorSearch:
    """Test vector search functionality."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            MuseumDocument(
                doc_id="test-001",
                museum_name="Тестовый музей 1",
                exhibition_title="Фотографическая выставка",
                description="Современная фотография и интерактивные инсталляции",
                start_date="2025-01-01",
                end_date="2025-12-31",
                tags=["фотография", "современное", "интерактив"],
                location="Тестовая улица, 1",
                accessibility=["лифт"],
                audience=["взрослые", "молодежь"]
            ),
            MuseumDocument(
                doc_id="test-002",
                museum_name="Тестовый музей 2",
                exhibition_title="Историческая выставка",
                description="Классическое искусство и исторические артефакты",
                start_date="2025-01-01",
                end_date="2025-12-31",
                tags=["история", "классика", "живопись"],
                location="Тестовая улица, 2",
                accessibility=["пандусы"],
                audience=["взрослые", "пожилые"]
            ),
            MuseumDocument(
                doc_id="test-003",
                museum_name="Тестовый музей 3",
                exhibition_title="Семейная выставка",
                description="Интерактивная выставка для всей семьи с детскими зонами",
                start_date="2025-01-01",
                end_date="2025-12-31",
                tags=["семья", "дети", "интерактив"],
                location="Тестовая улица, 3",
                accessibility=["лифт", "детские площадки"],
                audience=["семья", "дети"]
            )
        ]
    
    @pytest.fixture
    def vector_store(self, sample_documents):
        """Create vector store with sample documents."""
        embedding_model = EmbeddingModel()
        store = VectorStore(embedding_model)
        store.add_documents(sample_documents)
        return store
    
    def test_search_by_keywords(self, vector_store):
        """Test search by keywords."""
        results = vector_store.search("фотография", top_k=2)
        
        assert len(results) > 0
        assert any("фотография" in doc.tags for doc, _ in results)
    
    def test_search_by_multiple_keywords(self, vector_store):
        """Test search by multiple keywords."""
        results = vector_store.search("интерактив семья", top_k=3)
        
        assert len(results) > 0
        # Should find the family exhibition
        family_results = [doc for doc, score in results if "семья" in doc.tags]
        assert len(family_results) > 0
    
    def test_filter_by_tags(self, vector_store):
        """Test filtering by tags."""
        results = vector_store.search(
            "выставка", 
            filter_tags=["интерактив"], 
            top_k=3
        )
        
        assert len(results) > 0
        for doc, _ in results:
            assert "интерактив" in doc.tags
    
    def test_filter_by_audience(self, vector_store):
        """Test filtering by audience."""
        results = vector_store.search(
            "выставка",
            filter_audience=["семья"],
            top_k=3
        )
        
        assert len(results) > 0
        for doc, _ in results:
            assert "семья" in doc.audience
    
    def test_search_with_combined_filters(self, vector_store):
        """Test search with combined filters."""
        results = vector_store.search(
            "интерактив",
            filter_tags=["интерактив"],
            filter_audience=["взрослые"],
            top_k=3
        )
        
        assert len(results) > 0
        for doc, _ in results:
            assert "интерактив" in doc.tags
            assert "взрослые" in doc.audience
    
    def test_empty_search_results(self, vector_store):
        """Test handling of empty search results."""
        results = vector_store.search("несуществующий_термин", top_k=5)
        
        assert len(results) == 0
    
    def test_persistence(self, sample_documents):
        """Test vector store persistence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create store and add documents
            embedding_model = EmbeddingModel()
            store1 = VectorStore(embedding_model)
            store1.add_documents(sample_documents)
            
            # Search to verify it works
            results1 = store1.search("фотография", top_k=2)
            assert len(results1) > 0
            
            # Create new store instance (should load from disk)
            store2 = VectorStore(embedding_model)
            results2 = store2.search("фотография", top_k=2)
            
            assert len(results2) == len(results1)
            assert results1[0][0].doc_id == results2[0][0].doc_id


if __name__ == "__main__":
    pytest.main([__file__])