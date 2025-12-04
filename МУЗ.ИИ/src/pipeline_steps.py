"""Main pipeline steps implementation."""

from typing import Dict, List, Any, Optional, Tuple

from langchain.llms import BaseLLM

from .config import settings
from .dialogue_manager import DialogueManager
from .ner_extractor import NERExtractor
from .vector_store import VectorStore, MuseumDocument
from .final_response_builder import FinalResponseBuilder
from .embeddings import EmbeddingModel


class PipelineStep:
    """Base class for pipeline steps."""
    
    def __init__(self, name: str):
        """Initialize pipeline step.
        
        Args:
            name: Step name
        """
        self.name = name
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline step.
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context
        """
        raise NotImplementedError


class InitialRequestParser(PipelineStep):
    """Step 1: Parse initial user request."""
    
    def __init__(self):
        super().__init__("initial_request_parser")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse initial user request.
        
        Args:
            context: Pipeline context with 'user_request'
            
        Returns:
            Updated context with parsed request
        """
        user_request = context.get("user_request", "")
        
        # Basic parsing
        parsed_request = {
            "raw_text": user_request,
            "intent": "museum_recommendation",
            "explicit_age": None,
            "relationship": None,
            "mood_hint": None
        }
        
        # Simple regex-based extraction
        import re
        
        # Age extraction
        age_match = re.search(r'(\d+)\s*(?:лет|год|года)', user_request.lower())
        if age_match:
            parsed_request["explicit_age"] = int(age_match.group(1))
        
        # Relationship extraction
        relationship_keywords = {
            'девушка': 'partner',
            'парень': 'partner',
            'бабушка': 'grandparent',
            'дедушка': 'grandparent',
            'друг': 'friend',
            'семья': 'family'
        }
        
        for keyword, relationship in relationship_keywords.items():
            if keyword in user_request.lower():
                parsed_request["relationship"] = relationship
                break
        
        # Mood extraction
        mood_keywords = {
            'грустно': 'sad',
            'грусть': 'sad',
            'романтично': 'romantic',
            'романтика': 'romantic'
        }
        
        for keyword, mood in mood_keywords.items():
            if keyword in user_request.lower():
                parsed_request["mood_hint"] = mood
                break
        
        context["parsed_request"] = parsed_request
        context["enhanced_request"] = user_request  # Will be updated during dialogue
        
        return context


class DialogueStep(PipelineStep):
    """Step 2: Handle clarifying dialogue."""
    
    def __init__(self, llm: BaseLLM):
        super().__init__("dialogue")
        self.dialogue_manager = DialogueManager(llm)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clarifying dialogue.
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context with dialogue results
        """
        user_request = context.get("user_request", "")
        
        # Check if clarification is needed
        if self.dialogue_manager.needs_clarification(user_request):
            # Generate clarifying questions
            questions = self.dialogue_manager.generate_clarifying_questions(user_request)
            
            context["needs_clarification"] = True
            context["clarifying_questions"] = questions
            context["dialogue_manager"] = self.dialogue_manager
        else:
            context["needs_clarification"] = False
            context["clarifying_questions"] = []
        
        return context


class NERExtractionStep(PipelineStep):
    """Step 3: Extract named entities."""
    
    def __init__(self, llm: BaseLLM):
        super().__init__("ner_extraction")
        self.ner_extractor = NERExtractor(llm)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract named entities from enhanced request.
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context with extracted entities
        """
        enhanced_request = context.get("enhanced_request", "")
        
        if not enhanced_request:
            enhanced_request = context.get("user_request", "")
        
        entities = self.ner_extractor.extract_entities(enhanced_request)
        
        context["entities"] = entities
        return context


class VectorSearchStep(PipelineStep):
    """Step 4: Search in vector store."""
    
    def __init__(self, vector_store: VectorStore):
        super().__init__("vector_search")
        self.vector_store = vector_store
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Search for relevant exhibitions.
        
        Args:
            context: Pipeline context with entities
            
        Returns:
            Updated context with search results
        """
        entities = context.get("entities", {})
        
        # Build search query from entities
        search_query = self._build_search_query(entities)
        
        # Build filters from entities
        filter_tags = entities.get("hobbies", []) + entities.get("preferred_styles", [])
        filter_audience = self._get_audience_filter(entities.get("relationship"))
        
        # Search in vector store
        results = self.vector_store.search(
            query=search_query,
            top_k=settings.top_k,
            filter_tags=filter_tags if filter_tags else None,
            filter_audience=filter_audience if filter_audience else None
        )
        
        # Extract documents and scores
        documents = [doc for doc, _ in results]
        scores = [score for _, score in results]
        
        # Apply reranking if we have few results
        if len(documents) < settings.top_k:
            # Fallback search with broader query
            fallback_results = self.vector_store.search(
                query=search_query,
                top_k=settings.top_k * 2  # Get more results
            )
            
            # Merge results, avoiding duplicates
            existing_ids = {doc.doc_id for doc in documents}
            for doc, score in fallback_results:
                if doc.doc_id not in existing_ids:
                    documents.append(doc)
                    scores.append(score * 0.9)  # Slightly penalize fallback results
                    
                    if len(documents) >= settings.top_k:
                        break
        
        context["search_results"] = documents[:settings.top_k]
        context["similarity_scores"] = scores[:settings.top_k]
        context["search_query"] = search_query
        
        return context
    
    def _build_search_query(self, entities: Dict[str, Any]) -> str:
        """Build search query from entities.
        
        Args:
            entities: Extracted entities
            
        Returns:
            Search query string
        """
        query_parts = []
        
        # Add hobbies
        if entities.get("hobbies"):
            query_parts.extend(entities["hobbies"])
        
        # Add mood-related terms
        if entities.get("mood"):
            mood_terms = {
                "sad": ["спокойный", "размышляющий", "тихий"],
                "happy": ["веселый", "яркий", "позитивный"],
                "romantic": ["романтический", "интимный", "уютный"],
                "calm": ["спокойный", "медитативный", "гармоничный"]
            }
            query_parts.extend(mood_terms.get(entities["mood"], []))
        
        # Add preferred styles
        if entities.get("preferred_styles"):
            query_parts.extend(entities["preferred_styles"])
        
        # Add relationship-specific terms
        if entities.get("relationship"):
            rel_terms = {
                "partner": ["романтика", "для двоих", "интимный"],
                "grandparent": ["семейный", "доступный", "классический"],
                "parent": ["семейный", "для всех возрастов"],
                "friend": ["интересный", "обсуждаемый"],
                "child": ["детский", "интерактивный", "образовательный"],
                "solo": ["индивидуальный", "самостоятельный"]
            }
            query_parts.extend(rel_terms.get(entities["relationship"], []))
        
        # If no specific terms, use general museum terms
        if not query_parts:
            query_parts = ["выставка", "музей", "искусство"]
        
        return " ".join(query_parts)
    
    def _get_audience_filter(self, relationship: Optional[str]) -> List[str]:
        """Get audience filter based on relationship.
        
        Args:
            relationship: Relationship type
            
        Returns:
            List of audience types to filter by
        """
        if not relationship:
            return []
        
        audience_map = {
            "partner": ["взрослые", "молодежь"],
            "grandparent": ["пожилые", "взрослые", "семья"],
            "parent": ["семья", "взрослые"],
            "friend": ["молодежь", "взрослые"],
            "child": ["дети", "семья"],
            "solo": ["взрослые", "молодежь", "подростки"]
        }
        
        return audience_map.get(relationship, ["взрослые"])


class ResponseGenerationStep(PipelineStep):
    """Step 5: Generate final response."""
    
    def __init__(self, llm: BaseLLM):
        super().__init__("response_generation")
        self.response_builder = FinalResponseBuilder(llm)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final response with explanations.
        
        Args:
            context: Pipeline context with all previous results
            
        Returns:
            Final response
        """
        user_request = context.get("user_request", "")
        entities = context.get("entities", {})
        recommendations = context.get("search_results", [])
        similarity_scores = context.get("similarity_scores", [])
        
        # Generate final response
        final_response = self.response_builder.build_response(
            user_request=user_request,
            entities=entities,
            recommendations=recommendations,
            similarity_scores=similarity_scores
        )
        
        context["final_response"] = final_response
        return context


class MuseumRAGPipeline:
    """Main RAG pipeline orchestrator."""
    
    def __init__(self, llm: BaseLLM, vector_store: VectorStore):
        """Initialize pipeline.
        
        Args:
            llm: Language model instance
            vector_store: Vector store instance
        """
        self.llm = llm
        self.vector_store = vector_store
        
        # Initialize all pipeline steps
        self.steps = [
            InitialRequestParser(),
            DialogueStep(llm),
            NERExtractionStep(llm),
            VectorSearchStep(vector_store),
            ResponseGenerationStep(llm)
        ]
    
    def run(self, user_request: str) -> Dict[str, Any]:
        """Run the complete pipeline.
        
        Args:
            user_request: User's initial request
            
        Returns:
            Final response with recommendations
        """
        context = {
            "user_request": user_request,
            "timestamp": "",  # Could add timestamp
        }
        
        # Execute all pipeline steps
        for step in self.steps:
            context = step.execute(context)
        
        return context
    
    def continue_dialogue(self, user_response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Continue dialogue with user response.
        
        Args:
            user_response: User's response to clarifying questions
            context: Current pipeline context
            
        Returns:
            Updated context
        """
        dialogue_manager = context.get("dialogue_manager")
        
        if dialogue_manager:
            # Add to conversation history
            last_question = context.get("clarifying_questions", [""])[0]
            dialogue_manager.add_to_history(user_response, last_question)
            
            # Update enhanced request
            enhanced_request = dialogue_manager.get_enhanced_request()
            context["enhanced_request"] = enhanced_request
            
            # Re-run NER and search steps
            context = self.steps[2].execute(context)  # NER step
            context = self.steps[3].execute(context)  # Search step
            context = self.steps[4].execute(context)  # Response generation
        
        return context