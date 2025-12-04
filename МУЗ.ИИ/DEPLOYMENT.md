# Deployment Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Environment

```bash
cp .env.example .env
# Edit .env file with your settings
```

### 3. Run Demo

```bash
python demo.py
```

### 4. Interactive Mode

```bash
python -m src.app --interactive
```

## Production Deployment

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/vector_store
RUN python -c "from src.ingestion import create_sample_data; create_sample_data()"
RUN python -c "from src.embeddings import EmbeddingModel; from src.vector_store import VectorStore; from src.ingestion import ingest_data; emb = EmbeddingModel(); vs = VectorStore(emb); ingest_data('./data/sample_museums.json', vs, 'json')"

CMD ["python", "-m", "src.app", "--interactive"]
```

Build and run:

```bash
docker build -t museum-rag .
docker run -p 8080:8080 museum-rag
```

### Systemd Service

Create `/etc/systemd/system/museum-rag.service`:

```ini
[Unit]
Description=Museum RAG Pipeline
After=network.target

[Service]
Type=simple
User=museum-rag
WorkingDirectory=/opt/museum-rag
Environment=PATH=/opt/museum-rag/venv/bin
ExecStart=/opt/museum-rag/venv/bin/python -m src.app --interactive
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable service:

```bash
sudo systemctl enable museum-rag
sudo systemctl start museum-rag
```

## Configuration

### Environment Variables

- `LMSTUDIO_API_URL`: LM Studio API endpoint (default: http://127.0.0.1:8080)
- `LMSTUDIO_MODEL_NAME`: Model name (default: llama-3.2-3b-instruct)
- `VECTOR_STORE_DIR`: Vector store directory (default: ./data/vector_store)
- `TOP_K`: Number of recommendations (default: 5)
- `CHUNK_SIZE`: Text chunk size (default: 512)
- `MAX_CLARIFICATION_ROUNDS`: Max dialogue rounds (default: 2)

### Performance Tuning

1. **Vector Store Optimization**:
   - Increase `CHUNK_SIZE` for better context (1024-2048)
   - Adjust `CHUNK_OVERLAP` (50-200)

2. **Search Optimization**:
   - Increase `TOP_K` for more results (5-10)
   - Enable reranking with custom weights

3. **LLM Optimization**:
   - Adjust temperature (0.1-0.9)
   - Set max_tokens appropriately

## Monitoring

### Health Checks

Add health check endpoint:

```python
# health_check.py
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore

def health_check():
    try:
        emb = EmbeddingModel()
        vs = VectorStore(emb)
        count = vs.get_document_count()
        return {"status": "healthy", "documents": count}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Logging

Configure logging in `src/config.py`:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('museum_rag.log'),
        logging.StreamHandler()
    ]
)
```

## Security Considerations

1. **API Security**:
   - Use API keys if exposing endpoints
   - Implement rate limiting
   - Validate input data

2. **Data Security**:
   - Encrypt sensitive data
   - Regular backups
   - Access control

3. **Network Security**:
   - Use HTTPS
   - Firewall configuration
   - VPN for remote access

## Scaling

### Horizontal Scaling

1. **Load Balancing**:
   - Use nginx/haproxy
   - Round-robin distribution
   - Health checks

2. **Caching**:
   - Redis for session storage
   - CDN for static assets
   - Query result caching

3. **Database Scaling**:
   - Read replicas
   - Sharding
   - Connection pooling

### Vertical Scaling

1. **Resources**:
   - 8GB+ RAM recommended
   - SSD storage
   - Multi-core CPU

2. **Optimization**:
   - GPU acceleration (if available)
   - Model quantization
   - Batch processing

## Troubleshooting

### Common Issues

1. **LM Studio Connection**:
   ```bash
   # Check if LM Studio is running
   curl http://127.0.0.1:8080/v1/models
   ```

2. **Memory Issues**:
   ```bash
   # Monitor memory usage
   htop
   # Or use systemd
   systemctl status museum-rag
   ```

3. **Vector Store Corruption**:
   ```bash
   # Clear and rebuild
   rm -rf data/vector_store/*
   python -m src.ingestion
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Backup and Recovery

### Data Backup

```bash
# Backup vector store
tar -czf vector_store_backup.tar.gz data/vector_store/

# Backup source data
cp data/sample_museums.json museums_backup.json
```

### Recovery

```bash
# Restore vector store
tar -xzf vector_store_backup.tar.gz

# Rebuild if needed
python -m src.ingestion
```

## Updates

### Updating Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Code Updates

```bash
# Pull latest changes
git pull origin main

# Restart service
sudo systemctl restart museum-rag
```

## Support

For issues and questions:
1. Check logs: `tail -f museum_rag.log`
2. Create GitHub issue
3. Contact support: support@museum-rag.com