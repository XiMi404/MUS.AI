"""Data ingestion module for museum exhibitions."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import settings
from .vector_store import MuseumDocument, VectorStore
from .embeddings import EmbeddingModel


def load_museum_data_from_csv(file_path: str) -> List[MuseumDocument]:
    """Load museum data from CSV file.
    
    Expected CSV columns:
    - id: unique identifier
    - museum_name: name of the museum
    - exhibition_title: title of the exhibition
    - description: detailed description
    - start_date: exhibition start date (YYYY-MM-DD)
    - end_date: exhibition end date (YYYY-MM-DD)
    - tags: comma-separated tags
    - location: museum location/address
    - accessibility: comma-separated accessibility features
    - audience: comma-separated target audience types
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        List of MuseumDocument objects
    """
    documents = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Parse tags and other list fields
            tags = [tag.strip() for tag in row.get('tags', '').split(',') if tag.strip()]
            accessibility = [acc.strip() for acc in row.get('accessibility', '').split(',') if acc.strip()]
            audience = [aud.strip() for aud in row.get('audience', '').split(',') if aud.strip()]
            
            doc = MuseumDocument(
                doc_id=row['id'],
                museum_name=row['museum_name'],
                exhibition_title=row['exhibition_title'],
                description=row['description'],
                start_date=row['start_date'],
                end_date=row['end_date'],
                tags=tags,
                location=row['location'],
                accessibility=accessibility,
                audience=audience
            )
            documents.append(doc)
    
    return documents


def load_museum_data_from_json(file_path: str) -> List[MuseumDocument]:
    """Load museum data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of MuseumDocument objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        doc = MuseumDocument(**item)
        documents.append(doc)
    
    return documents


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into chunks with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            sentence_end = text.rfind('.', start, end)
            if sentence_end == -1:
                sentence_end = text.rfind('!', start, end)
            if sentence_end == -1:
                sentence_end = text.rfind('?', start, end)
            
            if sentence_end != -1 and sentence_end > start:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def create_exhibition_chunks(doc: MuseumDocument) -> List[MuseumDocument]:
    """Create chunks from exhibition description.
    
    Args:
        doc: Museum document
        
    Returns:
        List of chunked documents
    """
    chunks = chunk_text(
        doc.description, 
        settings.chunk_size, 
        settings.chunk_overlap
    )
    
    chunked_docs = []
    for i, chunk in enumerate(chunks):
        chunk_doc = MuseumDocument(
            doc_id=f"{doc.doc_id}_chunk_{i}",
            museum_name=doc.museum_name,
            exhibition_title=doc.exhibition_title,
            description=chunk,
            start_date=doc.start_date,
            end_date=doc.end_date,
            tags=doc.tags,
            location=doc.location,
            accessibility=doc.accessibility,
            audience=doc.audience
        )
        chunked_docs.append(chunk_doc)
    
    return chunked_docs


def ingest_data(
    data_path: str,
    vector_store: VectorStore,
    file_format: str = 'auto'
) -> int:
    """Ingest museum data into vector store.
    
    Args:
        data_path: Path to data file
        vector_store: Vector store instance
        file_format: 'csv', 'json', or 'auto' (detect from extension)
        
    Returns:
        Number of documents ingested
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Detect file format
    if file_format == 'auto':
        if data_path.suffix.lower() == '.csv':
            file_format = 'csv'
        elif data_path.suffix.lower() == '.json':
            file_format = 'json'
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    # Load data
    if file_format == 'csv':
        documents = load_museum_data_from_csv(str(data_path))
    elif file_format == 'json':
        documents = load_museum_data_from_json(str(data_path))
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    # Filter out expired exhibitions
    current_date = datetime.now().strftime('%Y-%m-%d')
    active_documents = []
    
    for doc in documents:
        if doc.end_date >= current_date:
            active_documents.append(doc)
    
    print(f"Found {len(active_documents)} active exhibitions")
    
    # Create chunks for long descriptions
    all_chunks = []
    for doc in active_documents:
        if len(doc.description) > settings.chunk_size:
            chunks = create_exhibition_chunks(doc)
            all_chunks.extend(chunks)
        else:
            all_chunks.append(doc)
    
    # Add to vector store
    vector_store.add_documents(all_chunks)
    
    return len(all_chunks)


def create_sample_data(output_path: str = "./data/sample_museums.json"):
    """Create sample museum data for testing.
    
    Args:
        output_path: Path to save sample data
    """
    sample_data = [
        {
            "doc_id": "tretyakov-001",
            "museum_name": "Третьяковская галерея",
            "exhibition_title": "Импрессионисты и постимпрессионисты",
            "description": "Уникальная выставка работ великих импрессионистов и постимпрессионистов XIX-XX веков. В экспозиции представлены полотна Моне, Ренуара, Дега, Сезанна, Ван Гога, Гогена. Выставка позволяет проследить эволюцию живописного искусства от классического импрессионизма к радикальным экспериментам постимпрессионистов. Особое внимание уделено влиянию фотографии на развитие живописи того периода. Интерактивные зоны позволяют посетителям узнать больше о техниках художников и историческом контексте создания произведений.",
            "start_date": "2025-09-15",
            "end_date": "2026-02-28",
            "tags": ["импрессионизм", "живопись", "фотография", "история", "интерактив"],
            "location": "Лаврушинский переулок, 10",
            "accessibility": ["лифт", "пандусы", "аудиогид"],
            "audience": ["взрослые", "подростки", "семья"]
        },
        {
            "doc_id": "moma-002", 
            "museum_name": "Московский музей современного искусства",
            "exhibition_title": "Цифровые горизонты: современное искусство и технологии",
            "description": "Выставка представляет работы современных художников, использующих цифровые технологии в своем творчестве. Инсталляции с дополненной реальностью, генеративное искусство, NFT-коллекции и интерактивные медиа-скульптуры. Особое внимание уделено взаимодействию искусства и искусственного интеллекта. Посетители могут не только наблюдать, но и участвовать в создании цифровых арт-объектов. Мастер-классы по цифровой фотографии и визуальным эффектам проходят каждые выходные.",
            "start_date": "2025-10-01",
            "end_date": "2026-01-15",
            "tags": ["современное искусство", "технологии", "интерактив", "фотография", "VR"],
            "location": "Гоголевский бульвар, 10",
            "accessibility": ["лифт", "пандусы", "тактильные экспонаты"],
            "audience": ["молодежь", "взрослые", "подростки"]
        },
        {
            "doc_id": "pushkin-003",
            "museum_name": "Музей изобразительных искусств им. Пушкина",
            "exhibition_title": "Фотографический портрет: от дагерротипа до цифры",
            "description": "Масштабная выставка, прослеживающая развитие фотографического портрета с 1840-х годов до наших дней. В экспозиции представлены редкие дагерротипы, классические портреты начала XX века, работы знаменитых фотографов-авангардистов, а также современные цифровые проекты. Особый раздел посвящен портретной фотографии в контексте семейной истории. Интерактивные стенды позволяют ознакомиться с техникой создания фотопортретов разных эпох. Подходит для романтических свиданий и семейных посещений.",
            "start_date": "2025-11-01",
            "end_date": "2026-03-31",
            "tags": ["фотография", "история", "портрет", "семья", "романтика"],
            "location": "Волхонка, 12",
            "accessibility": ["лифт", "пандусы", "аудиогид"],
            "audience": ["семья", "взрослые", "подростки"]
        },
        {
            "doc_id": "cosmos-004",
            "museum_name": "Музей космонавтики",
            "exhibition_title": "Космос глазами художников",
            "description": "Уникальная выставка, объединяющая науку и искусство в освещении темы космоса. Представлены работы художников разных эпох, вдохновленных космическими открытиями: от ранних фантастических иллюстраций до современных инсталляций с использованием данных с космических телескопов. Интерактивные зоны позволяют посетителям создать собственные космические пейзажи с помощью генеративных алгоритмов. Особенно интересно для молодежи и семей с детьми. Есть возможность сделать уникальные фотографии на фоне космических инсталляций.",
            "start_date": "2025-08-20",
            "end_date": "2026-04-30",
            "tags": ["космос", "наука", "искусство", "интерактив", "фотография", "семья"],
            "location": "Профсоюзная улица, 123",
            "accessibility": ["лифт", "пандусы", "детские площадки"],
            "audience": ["семья", "дети", "подростки", "взрослые"]
        },
        {
            "doc_id": "gms-005",
            "museum_name": "Государственный музей архитектуры им. Щусева",
            "exhibition_title": "Поэзия пространства: архитектура и литература",
            "description": "Выставка исследует тесную связь между архитектурой и литературой через века. Представлены архитектурные чертежи, вдохновленные литературными произведениями, а также книги, повлиявшие на формирование архитектурных стилей. Интерактивные инсталляции позволяют посетителям 'пройтись' по знаменитым литературным пространствам. Особый раздел посвящен московской архитектуре в русской поэзии. Подходит для интеллектуальных свиданий и посещений с пожилыми родственниками, интересующимися культурой и историей.",
            "start_date": "2025-09-01",
            "end_date": "2026-02-14",
            "tags": ["архитектура", "литература", "поэзия", "история", "интеллектуальное"],
            "location": "улица Воздвиженка, 5/25",
            "accessibility": ["лифт", "пандусы", "места отдыха"],
            "audience": ["взрослые", "пожилые", "интеллектуалы"]
        }
    ]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"Sample data created at {output_path}")


if __name__ == "__main__":
    # Create sample data
    create_sample_data()
    
    # Initialize embedding model and vector store
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(embedding_model)
    
    # Ingest sample data
    data_path = "./data/sample_museums.json"
    if Path(data_path).exists():
        count = ingest_data(data_path, vector_store, 'json')
        print(f"Successfully ingested {count} documents")
    else:
        print(f"Sample data file not found: {data_path}")