#!/usr/bin/env python3
"""Demo script for museum RAG pipeline."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.embeddings import EmbeddingModel
from vector_store import VectorStore, MuseumDocument
from ingestion import create_sample_data, ingest_data
from pipeline_steps import MuseumRAGPipeline
from app import LMStudioLLM
from config import settings


def create_demo_llm():
    """Create a demo LLM that works without LM Studio."""
    class DemoLLM:
        def __init__(self):
            self.name = "DemoLLM"
        
        def __call__(self, prompt, **kwargs):
            return self.run(prompt)
        
        def run(self, prompt):
            """Generate demo responses."""
            if "уточняющие вопросы" in prompt or "clarifying questions" in prompt:
                return "С кем вы планируете посетить выставку? (девушка/парень, друзья, семья, бабушка/дедушка)"
            elif "почему подходит" in prompt or "why_fit" in prompt:
                return "Эта выставка идеально подходит для вас, потому что соответствует вашим интересам."
            else:
                return "Благодарю за обращение! Надеюсь, мои рекомендации будут полезны."
        
        def predict(self, prompt, **kwargs):
            return self.run(prompt)
    
    return DemoLLM()


def run_demo():
    """Run a complete demo of the pipeline."""
    print("🎨 Демонстрация музейного RAG-пайплайна")
    print("=" * 50)
    
    # Create sample data
    print("\n📊 Создание демонстрационных данных...")
    create_sample_data()
    
    # Initialize components
    print("\n🔧 Инициализация компонентов...")
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(embedding_model)
    
    # Ingest sample data
    print("\n📥 Загрузка данных...")
    try:
        count = ingest_data("./data/sample_museums.json", vector_store, "json")
        print(f"✅ Загружено {count} выставок")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return
    
    # Create pipeline
    print("\n🔄 Создание пайплайна...")
    
    # Try to use real LLM, fallback to demo
    try:
        llm = LMStudioLLM(
            api_url=settings.lmstudio_api_url,
            model_name=settings.lmstudio_model_name
        )
        print("✅ Используется LM Studio LLM")
    except Exception:
        print("⚠️  Используется демо-версия LLM (без LM Studio)")
        llm = create_demo_llm()
    
    pipeline = MuseumRAGPipeline(llm, vector_store)
    
    # Test queries
    test_queries = [
        "Куда сходить с девушкой?",
        "Мне 25 лет, люблю фотографию",
        "Куда можно пойти с бабушкой?",
        "Хочу посмотреть современное искусство",
        "Мне грустно, куда пойти одному?",
    ]
    
    print(f"\n🚀 Тестирование пайплайна на {len(test_queries)} запросах...")
    print("-" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Запрос: \"{query}\"")
        print("   Обработка...")
        
        try:
            result = pipeline.run(query)
            
            # Display basic results
            final_response = result.get("final_response", {})
            recommendations = final_response.get("recommendations", [])
            
            print(f"   ✅ Найдено {len(recommendations)} рекомендаций")
            
            for j, rec in enumerate(recommendations[:2], 1):  # Show top 2
                print(f"      {j}. {rec.get('museum_name', '')} - \"{rec.get('title', '')}\"")
                print(f"         📅 {rec.get('dates', {}).get('start', '')} - {rec.get('dates', {}).get('end', '')}")
                print(f"         🎯 {rec.get('why_fit', '')[:100]}...")
            
            if len(recommendations) > 2:
                print(f"         ... и еще {len(recommendations) - 2}")
                
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    
    print("\n" + "=" * 50)
    print("✨ Демонстрация завершена!")
    print("\nДля запуска интерактивного режима:")
    print("python -m src.app --interactive")
    print("\nДля однократного запроса:")
    print("python -m src.app --query \"Ваш запрос\"")


if __name__ == "__main__":
    run_demo()
