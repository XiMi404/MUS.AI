"""Configuration module for museum RAG pipeline."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LM Studio configuration
    lmstudio_api_url: str = Field(default="http://127.0.0.1:8080", env="LMSTUDIO_API_URL")
    lmstudio_model_name: str = Field(default="llama-3.2-3b-instruct", env="LMSTUDIO_MODEL_NAME")
    
    # Vector store configuration
    vector_store_dir: str = Field(default="./data/vector_store", env="VECTOR_STORE_DIR")
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL_NAME")
    
    # Search configuration
    top_k: int = Field(default=5, env="TOP_K")
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    
    # Dialogue configuration
    max_clarification_rounds: int = Field(default=2, env="MAX_CLARIFICATION_ROUNDS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Ensure directories exist
Path(settings.vector_store_dir).mkdir(parents=True, exist_ok=True)
Path("./data").mkdir(parents=True, exist_ok=True)

# Model configurations
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048

# Weight coefficients for search ranking
HOBBY_WEIGHT = 0.4
MOOD_WEIGHT = 0.3
AGE_WEIGHT = 0.15
RELATIONSHIP_WEIGHT = 0.15

# Russian language patterns for NER
AGE_PATTERNS = [
    r'(\d+)\s*(?:лет|год|года)',
    r'мне\s*(\d+)',
    r'возраст\s*(\d+)',
]

RELATIONSHIP_PATTERNS = {
    'девушка': 'partner',
    'парень': 'partner', 
    'бабушка': 'grandparent',
    'дедушка': 'grandparent',
    'мама': 'parent',
    'папа': 'parent',
    'друг': 'friend',
    'подруга': 'friend',
    'ребенок': 'child',
    'один': 'solo',
    'соло': 'solo'
}

MOOD_PATTERNS = {
    'грустно': 'sad',
    'грусть': 'sad',
    'печально': 'sad',
    'весело': 'happy',
    'радостно': 'happy',
    'романтично': 'romantic',
    'романтика': 'romantic',
    'спокойно': 'calm',
    'умиротворенно': 'calm'
}

HOBBY_KEYWORDS = [
    'фотография', 'фото', 'живопись', 'искусство', 'история', 'наука', 
    'технологии', 'музыка', 'поэзия', 'литература', 'архитектура',
    'скульптура', 'современное', 'классика', 'интерактив', 'выставка'
]