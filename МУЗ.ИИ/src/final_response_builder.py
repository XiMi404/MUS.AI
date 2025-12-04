"""Final response generator with explanations."""

from typing import Dict, List, Any

from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .vector_store import MuseumDocument
from .config import settings


class FinalResponseBuilder:
    """Builds final response with explanations for recommendations."""
    
    def __init__(self, llm: BaseLLM):
        """Initialize response builder.
        
        Args:
            llm: Language model instance
        """
        self.llm = llm
        
        # Create prompt template for response generation
        self.response_prompt = PromptTemplate(
            input_variables=[
                "user_summary", "recommendations", "entities"
            ],
            template="""Ты - эксперт по музейным выставкам в Москве. Твоя задача - составить персонализированные рекомендации на основе запроса пользователя и найденных выставок.

ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{user_summary}

ИЗВЛЕЧЕННЫЕ СУЩНОСТИ:
{entities}

НАЙДЕННЫЕ ВЫСТАВКИ:
{recommendations}

ИНСТРУКЦИИ:
1. Начни с краткого резюме запроса пользователя (1-2 предложения)
2. Для каждой выставки:
   - Назови музей и название выставки
   - Дай краткое описание (2-3 предложения)
   - ОБЯЗАТЕЛЬНО объясни, почему эта выставка подходит конкретно этому пользователю, ссылаясь на его интересы, возраст, настроение и т.д.
   - Укажи даты проведения
   - Добавь практическую информацию (адрес, особенности доступности)
3. Заключи общими рекомендациями по посещению

ФОРМАТ ОТВЕТА:
Для каждой выставки используй структуру:
🔸 [Название музея] - "[Название выставки]"
📅 Период: [даты]
🎯 Почему подходит: [обоснование с привязкой к интересам пользователя]
📍 Адрес: [адрес]
📋 Описание: [описание]

Будь дружелюбным, информативным и конкретным в обосновании выбора."""
        )
        
        self.response_chain = LLMChain(
            llm=self.llm,
            prompt=self.response_prompt
        )
        
        # Create JSON response prompt
        self.json_prompt = PromptTemplate(
            input_variables=["user_summary", "recommendations", "entities"],
            template="""Создай структурированный JSON ответ на основе запроса пользователя и рекомендаций.

ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{user_summary}

ИЗВЛЕЧЕННЫЕ СУЩНОСТИ:
{entities}

НАЙДЕННЫЕ ВЫСТАВКИ:
{recommendations}

Создай JSON со следующей структурой:
{{
  "user_summary": "краткое описание запроса пользователя",
  "recommendations": [
    {{
      "id": "идентификатор выставки",
      "museum_name": "название музея",
      "title": "название выставки", 
      "short_description": "краткое описание",
      "why_fit": "подробное обоснование почему подходит именно этому пользователю",
      "dates": {{
        "start": "дата начала",
        "end": "дата окончания"
      }},
      "metadata": {{
        "tags": ["теги"],
        "accessibility": ["доступность"]
      }},
      "confidence": 0.95
    }}
  ],
  "explainers": "человекочитаемый текст объяснений"
}}

Убедись, что why_fit содержит конкретные ссылки на интересы и предпочтения пользователя."""
        )
        
        self.json_chain = LLMChain(
            llm=self.llm,
            prompt=self.json_prompt
        )
    
    def build_response(
        self,
        user_request: str,
        entities: Dict[str, Any],
        recommendations: List[MuseumDocument],
        similarity_scores: List[float]
    ) -> Dict[str, Any]:
        """Build final response with explanations.
        
        Args:
            user_request: Original user request
            entities: Extracted user entities
            recommendations: List of recommended exhibitions
            similarity_scores: Similarity scores for recommendations
            
        Returns:
            Complete response with explanations
        """
        # Create user summary
        user_summary = self._create_user_summary(user_request, entities)
        
        # Format recommendations for the prompt
        formatted_recs = self._format_recommendations(recommendations, similarity_scores)
        
        # Generate human-readable response
        try:
            human_response = self.response_chain.run(
                user_summary=user_summary,
                entities=str(entities),
                recommendations=formatted_recs
            )
        except Exception as e:
            print(f"Error generating human response: {e}")
            human_response = self._create_fallback_response(
                user_summary, entities, recommendations, similarity_scores
            )
        
        # Generate JSON response
        try:
            json_response = self.json_chain.run(
                user_summary=user_summary,
                entities=str(entities),
                recommendations=formatted_recs
            )
            # Parse JSON (handle potential formatting issues)
            import json
            try:
                structured_response = json.loads(json_response)
            except json.JSONDecodeError:
                structured_response = self._create_fallback_json(
                    user_summary, entities, recommendations, similarity_scores
                )
        except Exception as e:
            print(f"Error generating JSON response: {e}")
            structured_response = self._create_fallback_json(
                user_summary, entities, recommendations, similarity_scores
            )
        
        # Add human-readable text to structured response
        structured_response["explainers"] = human_response
        
        return structured_response
    
    def _create_user_summary(self, user_request: str, entities: Dict[str, Any]) -> str:
        """Create summary of user request and preferences.
        
        Args:
            user_request: Original user request
            entities: Extracted entities
            
        Returns:
            User summary string
        """
        summary_parts = [f"Пользователь запросил: '{user_request}'"]
        
        if entities.get("age"):
            summary_parts.append(f"Возраст: {entities['age']} лет")
        
        if entities.get("relationship"):
            relationship_map = {
                "partner": "с партнером",
                "grandparent": "с бабушкой/дедушкой",
                "parent": "с родителями",
                "friend": "с друзьями",
                "child": "с ребенком",
                "solo": "один/одна"
            }
            summary_parts.append(f"Состав: {relationship_map.get(entities['relationship'], entities['relationship'])}")
        
        if entities.get("mood"):
            mood_map = {
                "sad": "грустное настроение",
                "happy": "хорошее настроение",
                "romantic": "романтическое настроение",
                "calm": "спокойное настроение"
            }
            summary_parts.append(f"Настроение: {mood_map.get(entities['mood'], entities['mood'])}")
        
        if entities.get("hobbies"):
            summary_parts.append(f"Интересы: {', '.join(entities['hobbies'])}")
        
        if entities.get("preferred_styles"):
            summary_parts.append(f"Предпочтения: {', '.join(entities['preferred_styles'])}")
        
        return ". ".join(summary_parts) + "."
    
    def _format_recommendations(
        self, 
        recommendations: List[MuseumDocument], 
        similarity_scores: List[float]
    ) -> str:
        """Format recommendations for the prompt.
        
        Args:
            recommendations: List of museum documents
            similarity_scores: Similarity scores
            
        Returns:
            Formatted string
        """
        formatted = []
        
        for i, (doc, score) in enumerate(zip(recommendations, similarity_scores)):
            formatted.append(
                f"{i+1}. {doc.museum_name} - '{doc.exhibition_title}'\n"
                f"   Описание: {doc.description}\n"
                f"   Теги: {', '.join(doc.tags)}\n"
                f"   Аудитория: {', '.join(doc.audience)}\n"
                f"   Доступность: {', '.join(doc.accessibility)}\n"
                f"   Даты: {doc.start_date} - {doc.end_date}\n"
                f"   Релевантность: {score:.2f}"
            )
        
        return "\n\n".join(formatted)
    
    def _create_fallback_response(
        self,
        user_summary: str,
        entities: Dict[str, Any],
        recommendations: List[MuseumDocument],
        similarity_scores: List[float]
    ) -> str:
        """Create fallback response when LLM fails.
        
        Args:
            user_summary: User summary
            entities: Extracted entities
            recommendations: List of recommendations
            similarity_scores: Similarity scores
            
        Returns:
            Fallback response string
        """
        response_parts = [f"На основе вашего запроса подобраны следующие выставки:\n\n"]
        
        for i, (doc, score) in enumerate(zip(recommendations, similarity_scores)):
            response_parts.append(
                f"🔸 {doc.museum_name} - \"{doc.exhibition_title}\"\n"
                f"📅 Период: {doc.start_date} - {doc.end_date}\n"
                f"📍 Адрес: {doc.location}\n"
                f"📋 {doc.description[:200]}...\n"
                f"🎯 Подходит по тематике: {', '.join(doc.tags[:3])}\n\n"
            )
        
        return "".join(response_parts)
    
    def _create_fallback_json(
        self,
        user_summary: str,
        entities: Dict[str, Any],
        recommendations: List[MuseumDocument],
        similarity_scores: List[float]
    ) -> Dict[str, Any]:
        """Create fallback JSON response.
        
        Args:
            user_summary: User summary
            entities: Extracted entities
            recommendations: List of recommendations
            similarity_scores: Similarity scores
            
        Returns:
            Fallback JSON response
        """
        json_response = {
            "user_summary": user_summary,
            "recommendations": [],
            "explainers": ""
        }
        
        for i, (doc, score) in enumerate(zip(recommendations, similarity_scores)):
            # Create why_fit explanation
            why_fit_parts = []
            
            if entities.get("hobbies"):
                hobby_matches = [hobby for hobby in entities["hobbies"] if hobby in doc.tags]
                if hobby_matches:
                    why_fit_parts.append(f"Совпадение по интересам: {', '.join(hobby_matches)}")
            
            if entities.get("relationship"):
                audience_matches = [aud for aud in doc.audience if entities["relationship"] in aud.lower()]
                if audience_matches:
                    why_fit_parts.append(f"Подходит для вашей компании")
            
            if entities.get("preferred_styles"):
                style_matches = [style for style in entities["preferred_styles"] if any(tag in style for tag in doc.tags)]
                if style_matches:
                    why_fit_parts.append(f"Соответствует предпочитаемому стилю")
            
            why_fit = "; ".join(why_fit_parts) if why_fit_parts else "Подходит по общей тематике"
            
            recommendation = {
                "id": doc.doc_id,
                "museum_name": doc.museum_name,
                "title": doc.exhibition_title,
                "short_description": doc.description[:150] + "...",
                "why_fit": why_fit,
                "dates": {
                    "start": doc.start_date,
                    "end": doc.end_date
                },
                "metadata": {
                    "tags": doc.tags,
                    "accessibility": doc.accessibility
                },
                "confidence": float(score)
            }
            
            json_response["recommendations"].append(recommendation)
        
        return json_response