# Прототип МУЗ.ИИ

ИИ-ассистент для подбора музеев и выставок в Москве на основе персонализированных предпочтений.

## Архитектура

Пайплайн состоит из 5 основных этапов:
1. **Парсинг запроса** - извлечение базовой информации из запроса пользователя
2. **Диалоговое уточнение** - задание уточняющих вопросов при необходимости, чтобы получить больше информации о предпочтениях пользователя
3. **Извлечение сущностей (NER)** - извлечение сущностей предпочтений
4. **Векторный поиск** - поиск релевантных музеев и выставок в базе данных
5. **Генерация ответа** - персонализированные рекомендации с обоснованием


### Структура проекта
```
museum_rag/
├── src/
│   ├── __init__.py
│   ├── config.py              # Конфигурация
│   ├── embeddings.py          # Модель эмбеддингов
│   ├── vector_store.py        # Векторное хранилище
│   ├── ingestion.py           # Ингестирование данных
│   ├── dialogue_manager.py    # Диалоговый менеджер
│   ├── ner_extractor.py       # Извлечение сущностей
│   ├── pipeline_steps.py      # Этапы пайплайна
│   ├── final_response_builder.py  # Генерация ответов
│   ├── app.py                 # CLI интерфейс
│   └── tests/                 # Тесты
├── data/                      # Данные
├── requirements.txt
├── .env.example
└── README.md
```

## Состояние проекта 

**Запускаются отдельные части на Linux Mint**. 
На других операционных системах тестов не проводилось.

## Установка на Linux Mint

### Подготовка
```
# Обновить репозитории
sudo apt update
# Установить нужные утилиты (git, pip, venv и пр.)
sudo apt install -y git build-essential curl
# Проверить версию Python
python3 --version || python3.10 --version
```

### Если у вас нет Python 3.10+, попробуйте установить стандартный пакет python3.10 (в зависимости от версии Mint это может быть доступно), либо подключить deadsnakes PPA. Пример (только если python3.10 не установлен):
```
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-distutils
```

### Создайте и активируйте venv в папке со скаченным МУЗ.ИИ
```
python3.10 -m venv .venv
source .venv/bin/activate
```

### установить pip (обновить) и зависимости
```
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install spacy
python -m spacy download ru_core_news_sm
```

### Установка и запуск LM Studio (локальный LLM API)
LM Studio — десктоп/локальный менеджер моделей. После установки можно стартовать локальный сервер либо через GUI (Local Server / Developer tab), либо через CLI lms server start. 


### Создайте .env и внесите информацию (ниже -- информация по умолчанию)):
```
LMSTUDIO_API_URL=
LMSTUDIO_MODEL_NAME=ai-sage_gigachat3-10b-a1.8b
VECTOR_STORE_DIR=./data/vector_store
EMBEDDING_MODEL_NAME=ai-forever/sbert_large_nlu_ru
TOP_K=5
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_CLARIFICATION_ROUNDS=2
```

### Ингестирование данных и запуск приложения
```
(предполагается, что вы в корне проекта museum_rag)
# создать тестовые данные (в прототипе нет API музейных агрегаторов)
python -c "from src.ingestion import create_sample_data; create_sample_data()"
# запустить интерактивный режим
python -m src.app --interactive
# либо единичный запрос
python -m src.app --query "Куда сходить с девушкой?"
# сохранить результат
python -m src.app --query "Мне 25, люблю фотографию" --output results.json
```

### Тесты
```
pytest src/tests/
# или отдельные тесты
pytest src/tests/test_ner.py
pytest src/tests/test_vector_search.py
```

## Данные

### Формат данных

Пайплайн поддерживает данные в форматах CSV и JSON со следующими полями:
```json
{
  "id": "уникальный_идентификатор",
  "museum_name": "название_музея",
  "exhibition_title": "название_выставки",
  "description": "описание_выставки",
  "start_date": "2025-01-01",
  "end_date": "2025-12-31",
  "tags": ["тег1", "тег2"],
  "location": "адрес_музея",
  "accessibility": ["лифт", "пандусы"],
  "audience": ["взрослые", "семья"]
}
```

### Примеры использования
```
Куда сходить с девушкой на романтическое свидание?
Мне 25, с девушкой любим фотографию и интерактивные выставки
Куда пойти одному, когда грустно?
Семья с детьми, нужно что-то развлекательное и образовательное
```

## Дальнейшее развитие

- [ ] Интеграция с внешними API музеев
- [ ] Веб-интерфейс
- [ ] Мобильное приложение
