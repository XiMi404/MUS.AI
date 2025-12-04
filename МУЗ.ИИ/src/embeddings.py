"""Embedding model wrapper for museum RAG pipeline."""

from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings


class EmbeddingModel:
    """Wrapper for sentence transformer embeddings."""
    
    def __init__(self, model_name: str = None):
        """Initialize embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or settings.embedding_model_name
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            
        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a search query with optional prefix.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding
        """
        # Add query prefix for better search performance
        prefixed_query = f"query: {query}"
        return self.encode(prefixed_query)[0]
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )