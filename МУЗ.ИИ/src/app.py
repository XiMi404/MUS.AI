"""CLI interface for museum RAG pipeline."""

import json
import sys
from typing import Dict, Any, Optional

import click
from langchain.llms import BaseLLM

from .config import settings
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .ingestion import ingest_data, create_sample_data
from .pipeline_steps import MuseumRAGPipeline


class LMStudioLLM(BaseLLM):
    """Custom LLM wrapper for LM Studio API."""
    
    def __init__(self, api_url: str, model_name: str):
        """Initialize LM Studio LLM.
        
        Args:
            api_url: LM Studio API URL
            model_name: Model name
        """
        self.api_url = api_url
        self.model_name = model_name
        self._validate_connection()
    
    def _validate_connection(self):
        """Validate connection to LM Studio."""
        import requests
        try:
            response = requests.get(f"{self.api_url}/v1/models", timeout=5)
            if response.status_code != 200:
                print(f"Warning: Could not connect to LM Studio at {self.api_url}")
                print("Please ensure LM Studio is running with the API enabled.")
        except Exception as e:
            print(f"Warning: Connection to LM Studio failed: {e}")
            print("The pipeline will continue with reduced functionality.")
    
    def _call(self, prompt: str, stop: Optional[list] = None, **kwargs) -> str:
        """Call LM Studio API.
        
        Args:
            prompt: Input prompt
            stop: Stop sequences
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        import requests
        
        headers = {"Content-Type": "application/json"}
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/v1/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["text"]
            else:
                print(f"LM Studio API error: {response.status_code}")
                return "Извините, временно недоступен генеративный ИИ."
                
        except Exception as e:
            print(f"Error calling LM Studio API: {e}")
            return "Извините, произошла ошибка при обращении к ИИ."
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {"api_url": self.api_url, "model_name": self.model_name}


def initialize_pipeline(data_path: Optional[str] = None) -> MuseumRAGPipeline:
    """Initialize the RAG pipeline.
    
    Args:
        data_path: Path to museum data file (optional)
        
    Returns:
        Initialized pipeline
    """
    # Initialize embedding model
    embedding_model = EmbeddingModel()
    
    # Initialize vector store
    vector_store = VectorStore(embedding_model)
    
    # Initialize LLM
    llm = LMStudioLLM(
        api_url=settings.lmstudio_api_url,
        model_name=settings.lmstudio_model_name
    )
    
    # Ingest data if provided
    if data_path:
        try:
            count = ingest_data(data_path, vector_store)
            print(f"Successfully ingested {count} documents")
        except Exception as e:
            print(f"Error ingesting data: {e}")
            sys.exit(1)
    elif vector_store.get_document_count() == 0:
        # Create and ingest sample data
        print("No data found. Creating sample data...")
        create_sample_data()
        try:
            count = ingest_data("./data/sample_museums.json", vector_store, "json")
            print(f"Successfully ingested {count} sample documents")
        except Exception as e:
            print(f"Error ingesting sample data: {e}")
            sys.exit(1)
    
    # Initialize pipeline
    pipeline = MuseumRAGPipeline(llm, vector_store)
    
    return pipeline


def run_interactive_mode(pipeline: MuseumRAGPipeline):
    """Run pipeline in interactive mode.
    
    Args:
        pipeline: Initialized pipeline
    """
    print("\n🎨 Добро пожаловать в музейный помощник!")
    print("Расскажите, куда бы вы хотели сходить в Москве.")
    print("Примеры запросов:")
    print("- Куда сходить с девушкой?")
    print("- Мне 25, люблю фотографию, куда посоветуете?")
    print("- Куда можно пойти с бабушкой?")
    print("- Какие выставки подойдут для романтического свидания?")
    print("\nДля выхода введите 'выход' или 'exit'\n")
    
    while True:
        try:
            user_input = input("Ваш запрос: ").strip()
            
            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("Спасибо за использование! До встречи! 👋")
                break
            
            if not user_input:
                continue
            
            # Run pipeline
            print("\n🔍 Анализирую ваш запрос...")
            result = pipeline.run(user_input)
            
            # Handle clarifying questions
            if result.get("needs_clarification"):
                questions = result.get("clarifying_questions", [])
                for question in questions:
                    print(f"\n❓ {question}")
                    
                    clarification = input("Ваш ответ: ").strip()
                    if clarification.lower() in ['выход', 'exit', 'quit']:
                        print("Спасибо за использование! До встречи! 👋")
                        return
                    
                    # Continue dialogue
                    result = pipeline.continue_dialogue(clarification, result)
            
            # Display results
            final_response = result.get("final_response", {})
            
            if final_response:
                print("\n" + "="*60)
                print("🎯 ПЕРСОНАЛИЗИРОВАННЫЕ РЕКОМЕНДАЦИИ")
                print("="*60)
                
                explainers = final_response.get("explainers", "")
                if explainers:
                    print(explainers)
                else:
                    # Fallback display
                    recommendations = final_response.get("recommendations", [])
                    for i, rec in enumerate(recommendations, 1):
                        print(f"\n{i}. {rec.get('museum_name', '')} - \"{rec.get('title', '')}\"")
                        print(f"   📅 {rec.get('dates', {}).get('start', '')} - {rec.get('dates', {}).get('end', '')}")
                        print(f"   🎯 {rec.get('why_fit', '')}")
                        print(f"   📋 {rec.get('short_description', '')}")
            
            print("\n" + "-"*60)
            
        except KeyboardInterrupt:
            print("\n\nСпасибо за использование! До встречи! 👋")
            break
        except Exception as e:
            print(f"\n❌ Произошла ошибка: {e}")
            print("Попробуйте еще раз или обратитесь к администратору.")


@click.command()
@click.option('--data', '-d', help='Path to museum data file (CSV or JSON)')
@click.option('--interactive', '-i', is_flag=True, help='Run in interactive mode')
@click.option('--query', '-q', help='Single query to process')
@click.option('--output', '-o', help='Output file for results (JSON format)')
def main(data, interactive, query, output):
    """Museum RAG Pipeline CLI."""
    
    print("🎨 Запускаю музейный RAG-пайплайн...")
    
    try:
        # Initialize pipeline
        pipeline = initialize_pipeline(data)
        print(f"✅ Пайплайн инициализирован. В базе {pipeline.vector_store.get_document_count()} выставок.")
        
        if interactive:
            run_interactive_mode(pipeline)
        
        elif query:
            print(f"\n🔍 Обрабатываю запрос: '{query}'")
            result = pipeline.run(query)
            
            # Handle clarifying questions for single query
            if result.get("needs_clarification"):
                print("\n❓ Для лучшего подбора нужна дополнительная информация:")
                questions = result.get("clarifying_questions", [])
                for q in questions:
                    print(f"  - {q}")
                print("\nЗапустите в интерактивном режиме (-i) для диалога.")
                return
            
            # Display results
            final_response = result.get("final_response", {})
            
            if final_response:
                print("\n" + "="*60)
                print("🎯 РЕЗУЛЬТАТЫ ПОИСКА")
                print("="*60)
                
                explainers = final_response.get("explainers", "")
                if explainers:
                    print(explainers)
                
                # Save to file if requested
                if output:
                    with open(output, 'w', encoding='utf-8') as f:
                        json.dump(final_response, f, ensure_ascii=False, indent=2)
                    print(f"\n💾 Результаты сохранены в {output}")
        
        else:
            print("\nВыберите режим работы:")
            print("  -i, --interactive    Интерактивный режим")
            print("  -q, --query TEXT     Однократный запрос")
            print("  -h, --help          Помощь")
    
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()