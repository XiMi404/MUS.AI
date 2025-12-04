"""Vector store implementation using FAISS for museum exhibitions."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from .config import settings
from .embeddings import EmbeddingModel


class MuseumDocument:
    """Document representation for museum exhibitions."""
    
    def __init__(
        self,
        doc_id: str,
        museum_name: str,
        exhibition_title: str,
        description: str,
        start_date: str,
        end_date: str,
        tags: List[str],
        location: str,
        accessibility: List[str],
        audience: List[str]
    ):
        self.doc_id = doc_id
        self.museum_name = museum_name
        self.exhibition_title = exhibition_title
        self.description = description
        self.start_date = start_date
        self.end_date = end_date
        self.tags = tags
        self.location = location
        self.accessibility = accessibility
        self.audience = audience
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'doc_id': self.doc_id,
            'museum_name': self.museum_name,
            'exhibition_title': self.exhibition_title,
            'description': self.description,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'tags': self.tags,
            'location': self.location,
            'accessibility': self.accessibility,
            'audience': self.audience
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MuseumDocument':
        """Create from dictionary."""
        return cls(**data)


class VectorStore:
    """FAISS-based vector store for museum exhibitions."""
    
    def __init__(self, embedding_model: EmbeddingModel):
        """Initialize vector store.
        
        Args:
            embedding_model: Embedding model instance
        """
        self.embedding_model = embedding_model
        self.index = None
        self.documents = {}
        self.doc_embeddings = {}
        
        self.index_path = Path(settings.vector_store_dir) / "faiss.index"
        self.docs_path = Path(settings.vector_store_dir) / "documents.pkl"
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index and documents."""
        if self.index_path.exists() and self.docs_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"Loaded {len(self.documents)} documents from existing index")
            except Exception as e:
                print(f"Failed to load existing index: {e}")
                self.index = None
                self.documents = {}
    
    def _save_index(self):
        """Save FAISS index and documents."""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
    
    def add_documents(self, documents: List[MuseumDocument]):
        """Add documents to vector store.
        
        Args:
            documents: List of museum documents
        """
        if not documents:
            return
        
        # Prepare texts for embedding
        texts = []
        for doc in documents:
            # Combine title and description for embedding
            text = f"{doc.exhibition_title}. {doc.description}"
            texts.append(text)
            self.documents[doc.doc_id] = doc
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Initialize or extend index
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_model.embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        
        # Save to disk
        self._save_index()
        
        print(f"Added {len(documents)} documents to vector store")
    
    def search(
        self, 
        query: str, 
        top_k: int = None,
        filter_tags: Optional[List[str]] = None,
        filter_audience: Optional[List[str]] = None
    ) -> List[Tuple[MuseumDocument, float]]:
        """Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_tags: Optional list of tags to filter by
            filter_audience: Optional list of audience types to filter by
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        top_k = top_k or settings.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode_query(query)
        
        # Search in index
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32), 
            min(top_k * 2, len(self.documents))  # Get more results for filtering
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                break
            
            # Get document ID from index position
            doc_id = list(self.documents.keys())[idx]
            doc = self.documents[doc_id]
            
            # Apply filters
            if filter_tags:
                if not any(tag in doc.tags for tag in filter_tags):
                    continue
            
            if filter_audience:
                if not any(aud in doc.audience for aud in filter_audience):
                    continue
            
            results.append((doc, float(score)))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_document_count(self) -> int:
        """Get total number of documents in store."""
        return len(self.documents)
    
    def clear(self):
        """Clear all documents and index."""
        self.index = None
        self.documents = {}
        self.doc_embeddings = {}
        
        # Remove files
        if self.index_path.exists():
            self.index_path.unlink()
        if self.docs_path.exists():
            self.docs_path.unlink()