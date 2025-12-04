"""Dialogue manager for clarifying user preferences."""

from typing import Dict, List, Optional, Tuple

from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .config import settings


class DialogueManager:
    """Manages clarifying dialogue with users."""
    
    def __init__(self, llm: BaseLLM):
        """Initialize dialogue manager.
        
        Args:
            llm: Language model instance
        """
        self.llm = llm
        self.conversation_history = []
        self.clarification_rounds = 0
        
        # Define clarification prompt templates
        self.clarification_prompt = PromptTemplate(
            input_variables=["user_request", "missing_info", "conversation_history"],
            template="""Ты - вежливый помощник, который задает уточняющие вопросы для лучшего понимания предпочтений пользователя при выборе музея или выставки.

Текущий запрос пользователя: {user_request}

История диалога: {conversation_history}

На основе запроса определи, какой информации не хватает для качественного подбора выставки:
- Возраст посетителей
- С кем идет пользователь (девушка, друзья, семья, бабушка/дедушка)
- Интересы и предпочтения (живопись, фотография, история, технологии и т.д.)
- Настроение или повод (романтическое свидание, семейный выход, интеллектуальное развлечение)
- Физические ограничения или особые потребности
- Предпочтительный формат (интерактивные экспозиции или спокойные залы)

Сформулируй ОДИН краткий (1-2 предложения) уточняющий вопрос на русском языке, который поможет получить недостающую информацию. Будь дружелюбным и не навязчивым.

Вопрос:"""
        )
        
        self.clarification_chain = LLMChain(
            llm=self.llm,
            prompt=self.clarification_prompt
        )
        
        self.analysis_prompt = PromptTemplate(
            input_variables=["user_request"],
            template="""Проанализируй следующий запрос пользователя и определи, достаточно ли информации для подбора музея/выставки или нужны уточнения:

Запрос: {user_request}

Определи, указаны ли в запросе:
1. Возраст посетителей (или возрастная категория)
2. Состав компании (с кем идет человек)
3. Интересы и предпочтения
4. Повод или настроение
5. Особые требования

Ответь кратко: "нужны уточнения" или "информации достаточно", указав что именно не хватает."""
        )
        
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=self.analysis_prompt
        )
    
    def needs_clarification(self, user_request: str) -> bool:
        """Determine if the user request needs clarification.
        
        Args:
            user_request: Original user request
            
        Returns:
            True if clarification is needed
        """
        # Simple heuristic-based check first
        has_age = any(word in user_request.lower() for word in ['лет', 'год', 'возраст', 'мне'])
        has_relationship = any(word in user_request.lower() for word in ['с девушкой', 'с парнем', 'с другом', 'с семьей', 'с бабушкой', 'с дедушкой', 'один', 'одна'])
        has_interests = any(word in user_request.lower() for word in ['люблю', 'нравится', 'интересуюсь', 'увлекаюсь'])
        
        # If we have at least age and relationship, probably don't need clarification
        if has_age and has_relationship:
            return False
        
        # Use LLM for more sophisticated analysis
        try:
            analysis_result = self.analysis_chain.run(user_request=user_request)
            return "нужны уточнения" in analysis_result.lower()
        except Exception:
            # Fallback to heuristic
            return not (has_age or has_relationship or has_interests)
    
    def generate_clarifying_questions(self, user_request: str) -> List[str]:
        """Generate clarifying questions for the user.
        
        Args:
            user_request: Original user request
            
        Returns:
            List of clarifying questions (usually 1-2 questions)
        """
        if self.clarification_rounds >= settings.max_clarification_rounds:
            return []
        
        # Determine what information is missing
        missing_info = []
        
        if not any(word in user_request.lower() for word in ['лет', 'год', 'возраст', 'мне']):
            missing_info.append("возраст")
        
        if not any(word in user_request.lower() for word in ['с девушкой', 'с парнем', 'с другом', 'с семьей', 'с бабушкой', 'с дедушкой', 'один', 'одна']):
            missing_info.append("состав компании")
        
        if not any(word in user_request.lower() for word in ['люблю', 'нравится', 'интересуюсь', 'увлекаюсь']):
            missing_info.append("интересы")
        
        # Generate question using LLM
        try:
            question = self.clarification_chain.run(
                user_request=user_request,
                missing_info=", ".join(missing_info),
                conversation_history="\n".join(self.conversation_history[-3:])  # Last 3 exchanges
            )
            
            # Clean up the question
            question = question.strip()
            if question and not question.endswith('?'):
                question += '?'
            
            self.clarification_rounds += 1
            return [question] if question else []
            
        except Exception as e:
            print(f"Error generating clarification question: {e}")
            return self._get_fallback_questions(missing_info)
    
    def _get_fallback_questions(self, missing_info: List[str]) -> List[str]:
        """Get fallback questions when LLM fails.
        
        Args:
            missing_info: List of missing information types
            
        Returns:
            List of fallback questions
        """
        questions = []
        
        if "возраст" in missing_info:
            questions.append("Сколько вам лет или какой возрастной категории вы принадлежите?")
        
        if "состав компании" in missing_info:
            questions.append("С кем вы планируете посетить выставку? (девушка/парень, друзья, семья, бабушка/дедушка)")
        
        if "интересы" in missing_info:
            questions.append("Какие темы вас особенно интересуют? (живопись, фотография, история, технологии и т.д.)")
        
        return questions[:2]  # Return max 2 questions
    
    def add_to_history(self, user_input: str, assistant_response: str):
        """Add exchange to conversation history.
        
        Args:
            user_input: User's message
            assistant_response: Assistant's response
        """
        self.conversation_history.append(f"Пользователь: {user_input}")
        self.conversation_history.append(f"Ассистент: {assistant_response}")
    
    def get_enhanced_request(self) -> str:
        """Get the enhanced user request with clarification information.
        
        Returns:
            Enhanced request string
        """
        # Combine original request with conversation history
        if not self.conversation_history:
            return ""
        
        # Extract the most recent user intent
        history_text = "\n".join(self.conversation_history[-6:])  # Last 3 exchanges
        
        return history_text
    
    def reset(self):
        """Reset conversation history and counters."""
        self.conversation_history = []
        self.clarification_rounds = 0