"""Named Entity Recognition for user preferences."""

import re
from typing import Dict, List, Optional, Any

import spacy
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .config import (
    AGE_PATTERNS, RELATIONSHIP_PATTERNS, MOOD_PATTERNS, 
    HOBBY_KEYWORDS, settings
)


class NERExtractor:
    """Extracts named entities from user requests."""
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize NER extractor.
        
        Args:
            llm: Language model for fallback NER (optional)
        """
        self.llm = llm
        self.nlp = None
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("ru_core_news_sm")
        except OSError:
            try:
                self.nlp = spacy.load("ru_core_news_lg")
            except OSError:
                print("spaCy Russian model not found. Using regex-based extraction with LLM fallback.")
        
        # Initialize LLM-based NER if LLM provided
        if self.llm:
            self.ner_prompt = PromptTemplate(
                input_variables=["text"],
                template="""Извлеки следующие сущности из текста:
- возраст (число)
- отношение (девушка, парень, друг, семья, бабушка, дедушка и т.д.)
- настроение/повод (романтика, грусть, веселье и т.д.)
- интересы/хобби (фотография, живопись, история, технологии и т.д.)
- особые требования (физические ограничения, время и т.д.)

Текст: {text}

Ответ в формате JSON:
{{
  "age": число или null,
  "relationship": строка или null,
  "mood": строка или null,
  "hobbies": список строк,
  "accessibility": список строк,
  "preferred_styles": список строк
}}"""
            )
            
            self.ner_chain = LLMChain(
                llm=self.llm,
                prompt=self.ner_prompt
            )
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with extracted entities
        """
        entities = {
            "age": None,
            "relationship": None,
            "mood": None,
            "hobbies": [],
            "accessibility": [],
            "preferred_styles": []
        }
        
        # Extract using regex patterns first
        regex_entities = self._extract_with_regex(text)
        entities.update(regex_entities)
        
        # Extract using spaCy if available
        if self.nlp:
            spacy_entities = self._extract_with_spacy(text)
            # Merge spaCy results, preferring non-None values
            for key, value in spacy_entities.items():
                if value and (not entities.get(key) or entities[key] is None):
                    entities[key] = value
        
        # Use LLM fallback if still missing key information
        if self.llm and self._needs_llm_fallback(entities):
            try:
                llm_entities = self._extract_with_llm(text)
                # Merge LLM results
                for key, value in llm_entities.items():
                    if value and (not entities.get(key) or entities[key] is None):
                        entities[key] = value
            except Exception as e:
                print(f"LLM NER failed: {e}")
        
        # Post-process entities
        entities = self._post_process_entities(entities)
        
        return entities
    
    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract entities using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with extracted entities
        """
        entities = {}
        text_lower = text.lower()
        
        # Extract age
        for pattern in AGE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    age = int(match.group(1))
                    if 5 <= age <= 120:  # Reasonable age range
                        entities["age"] = age
                        break
                except ValueError:
                    continue
        
        # Extract relationship
        for keyword, relationship in RELATIONSHIP_PATTERNS.items():
            if keyword in text_lower:
                entities["relationship"] = relationship
                break
        
        # Extract mood
        for keyword, mood in MOOD_PATTERNS.items():
            if keyword in text_lower:
                entities["mood"] = mood
                break
        
        # Extract hobbies/interests
        hobbies = []
        for hobby in HOBBY_KEYWORDS:
            if hobby in text_lower:
                hobbies.append(hobby)
        
        if hobbies:
            entities["hobbies"] = hobbies
        
        # Extract accessibility needs
        accessibility = []
        if any(word in text_lower for word in ['инвалид', 'коляска', 'ограничен']):
            accessibility.append('wheelchair')
        
        if accessibility:
            entities["accessibility"] = accessibility
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with extracted entities
        """
        entities = {}
        doc = self.nlp(text)
        
        # Extract numbers (potential ages)
        ages = []
        for ent in doc.ents:
            if ent.label_ == "NUM" and ent.text.isdigit():
                age = int(ent.text)
                if 5 <= age <= 120:
                    ages.append(age)
        
        if ages:
            entities["age"] = min(ages)  # Take the smallest reasonable age
        
        # Extract people/relationships
        people = []
        for ent in doc.ents:
            if ent.label_ in ["PER", "PERSON"]:
                people.append(ent.text.lower())
        
        # Check for relationship keywords in context
        for token in doc:
            if token.text.lower() in RELATIONSHIP_PATTERNS:
                entities["relationship"] = RELATIONSHIP_PATTERNS[token.text.lower()]
                break
        
        return entities
    
    def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Extract entities using LLM.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with extracted entities
        """
        try:
            result = self.ner_chain.run(text=text)
            # Parse JSON response
            import json
            entities = json.loads(result)
            return entities
        except (json.JSONDecodeError, Exception):
            return {}
    
    def _needs_llm_fallback(self, entities: Dict[str, Any]) -> bool:
        """Check if LLM fallback is needed.
        
        Args:
            entities: Current extracted entities
            
        Returns:
            True if LLM fallback needed
        """
        # Need fallback if missing key information
        return (
            entities.get("age") is None or
            entities.get("relationship") is None or
            not entities.get("hobbies")
        )
    
    def _post_process_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process extracted entities.
        
        Args:
            entities: Raw extracted entities
            
        Returns:
            Post-processed entities
        """
        # Normalize hobbies
        if entities.get("hobbies"):
            normalized_hobbies = []
            for hobby in entities["hobbies"]:
                # Map common variations to standard terms
                hobby_map = {
                    "фото": "фотография",
                    "фотосъемка": "фотография",
                    "картины": "живопись",
                    "живописи": "живопись",
                    "искусстве": "искусство",
                    "истории": "история",
                    "технологиях": "технологии",
                    "музыке": "музыка",
                    "поэзии": "поэзия",
                    "литературе": "литература",
                    "архитектуре": "архитектура",
                    "скульптуре": "скульптура"
                }
                normalized = hobby_map.get(hobby, hobby)
                if normalized not in normalized_hobbies:
                    normalized_hobbies.append(normalized)
            entities["hobbies"] = normalized_hobbies
        
        # Map relationship to standardized values
        if entities.get("relationship"):
            rel_map = {
                "подруга": "partner",
                "молодой человек": "partner",
                "мама": "parent",
                "папа": "parent",
                "сын": "child",
                "дочь": "child"
            }
            entities["relationship"] = rel_map.get(
                entities["relationship"], 
                entities["relationship"]
            )
        
        # Add preferred styles based on hobbies and mood
        entities["preferred_styles"] = []
        
        if entities.get("mood") == "romantic":
            entities["preferred_styles"].extend(["романтический", "интимный"])
        
        if entities.get("mood") == "sad":
            entities["preferred_styles"].extend(["спокойный", "размышляющий"])
        
        if "фотография" in entities.get("hobbies", []):
            entities["preferred_styles"].append("визуальный")
        
        if any(word in entities.get("hobbies", []) for word in ["история", "архитектура"]):
            entities["preferred_styles"].append("классический")
        
        if any(word in entities.get("hobbies", []) for word in ["технологии", "современное"]):
            entities["preferred_styles"].append("современный")
        
        return entities