#!/usr/bin/env python3
"""Example usage of museum RAG pipeline."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from embeddings import EmbeddingModel
from vector_store import VectorStore
from ingestion import create_sample_data, ingest_data
from pipeline_steps import MuseumRAGPipeline
from app import LMStudioLLM
from config import settings


def main():
    """Example usage of the pipeline."""
    
    # Step 1: Prepare data
    print("1. Подготовка данных...")
    create_sample_data()
    
    # Step 2: Initialize components
    print("2. Инициализация компонентов...")
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(embedding_model)
    
    # Step 3: Ingest data
    print("3. Загрузка данных...")
    count = ingest_data("./data/sample_museums.json", vector_store, "json")
    print(f"   Загружено {count} выставок")
    
    # Step 4: Initialize LLM and pipeline
    print("4. Создание пайплайна...")
    try:
        llm = LMStudioLLM(
            api_url=settings.lmstudio_api_url,
            model_name=settings.lmstudio_model_name
        )
        print("   ✅ LM Studio LLM подключен")
    except Exception as e:
        print(f"   ⚠️  Используется fallback (демо) режим: {e}")
        # Create a simple fallback LLM
        class FallbackLLM:
            def __call__(self, prompt, **kwargs):
                return "[Fallback] Генерация ответа временно недоступна"
        llm = FallbackLLM()
    
    pipeline = MuseumRAGPipeline(llm, vector_store)
    
    # Step 5: Process queries
    print("\n5. Обработка запросов...")
    
    queries = [
        "Куда сходить с девушкой? Мне 25 лет",
        "Интересуюсь историей и архитектурой, куда посоветуете?",
        "Семья с ребенком 10 лет, что посмотреть?",
    ]
    
    results = {}
    
    for i, query in enumerate(queries, 1):
        print(f"\n   Запрос {i}: {query}")
        
        try:
            result = pipeline.run(query)
            final_response = result.get("final_response", {})
            
            if final_response:
                results[query] = final_response
                print(f"   ✅ Получен ответ с {len(final_response.get('recommendations', []))} рекомендациями")
            else:
                print(f"   ❌ Нет ответа")
                
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            results[query] = {"error": str(e)}
    
    # Step 6: Display results
    print("\n6. Результаты:")
    print("=" * 80)
    
    for query, response in results.items():
        print(f"\n📋 Запрос: {query}")
        print("-" * 40)
        
        if "error" in response:
            print(f"❌ Ошибка: {response['error']}")
            continue
        
        recommendations = response.get("recommendations", [])
        
        if not recommendations:
            print("❌ Рекомендации не найдены")
            continue
        
        print(f"🎯 Найдено {len(recommendations)} рекомендаций:\n")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec.get('museum_name', '')}")
            print(f"      "{rec.get('title', '')}")
            print(f"      📅 {rec.get('dates', {}).get('start', '')} - {rec.get('dates', {}).get('end', '')}")
            print(f"      🎯 {rec.get('why_fit', '')}")
            print(f"      📊 Уверенность: {rec.get('confidence', 0):.2f}")
            print()
    
    # Step 7: Save results to file
    print("7. Сохранение результатов...")
    with open("pipeline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("   ✅ Результаты сохранены в pipeline_results.json")
    
    print("\n✨ Пример использования завершен!")


if __name__ == "__main__":
    main()