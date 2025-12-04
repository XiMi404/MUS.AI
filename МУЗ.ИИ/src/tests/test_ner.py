"""Tests for NER extraction."""

import pytest
from src.ner_extractor import NERExtractor


class TestNERExtractor:
    """Test NER extraction functionality."""
    
    def test_extract_age(self):
        """Test age extraction."""
        extractor = NERExtractor()
        
        test_cases = [
            ("Мне 25 лет", 25),
            ("Мне 30 лет", 30),
            ("Мне 18 лет", 18),
            ("Мне 45 лет", 45),
        ]
        
        for text, expected_age in test_cases:
            entities = extractor.extract_entities(text)
            assert entities["age"] == expected_age, f"Failed for: {text}"
    
    def test_extract_relationship(self):
        """Test relationship extraction."""
        extractor = NERExtractor()
        
        test_cases = [
            ("Хочу сходить с девушкой", "partner"),
            ("Куда можно пойти с бабушкой?", "grandparent"),
            ("С другом хотим посетить", "friend"),
            ("Семейный выход", "family"),
            ("Пойду один", "solo"),
        ]
        
        for text, expected_rel in test_cases:
            entities = extractor.extract_entities(text)
            assert entities["relationship"] == expected_rel, f"Failed for: {text}"
    
    def test_extract_mood(self):
        """Test mood extraction."""
        extractor = NERExtractor()
        
        test_cases = [
            ("Когда грустно, куда пойти?", "sad"),
            ("Хочу романтики", "romantic"),
            ("Весело провести время", "happy"),
            ("Спокойное место", "calm"),
        ]
        
        for text, expected_mood in test_cases:
            entities = extractor.extract_entities(text)
            assert entities["mood"] == expected_mood, f"Failed for: {text}"
    
    def test_extract_hobbies(self):
        """Test hobby extraction."""
        extractor = NERExtractor()
        
        text = "Люблю фотографию, живопись и историю"
        entities = extractor.extract_entities(text)
        
        assert "фотография" in entities["hobbies"]
        assert "живопись" in entities["hobbies"]
        assert "история" in entities["hobbies"]
    
    def test_comprehensive_extraction(self):
        """Test comprehensive entity extraction."""
        extractor = NERExtractor()
        
        text = "Мне 28 лет, хочу сходить с девушкой. Люблю современное искусство и фотографию. Когда грустно, предпочитаю спокойные места."
        entities = extractor.extract_entities(text)
        
        assert entities["age"] == 28
        assert entities["relationship"] == "partner"
        assert entities["mood"] == "sad"
        assert "современное искусство" in entities["hobbies"]
        assert "фотография" in entities["hobbies"]
        assert "спокойный" in entities["preferred_styles"]


if __name__ == "__main__":
    pytest.main([__file__])