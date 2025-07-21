# YouAreBot - Advanced Conversational AI Project

## Overview
This is an advanced GPT-bot for the "You are Bot" project that attempts to deceive users into thinking it is a human being. The project includes multiple LLM backends, a web interface, and training capabilities for bot detection.

## Architecture

### Core Components
1. **FastAPI Backend** (`main.py`) - Main API server handling chat requests
2. **Streamlit Web Interface** (`web/`) - User-friendly chat interface  
3. **LLM Backends** (`llm/`) - Multiple language model options:
   - OpenAI GPT models
   - DeepSeek models  
   - Local LLM server (llama.cpp or simple server)
4. **Database** (PostgreSQL) - Conversation history storage
5. **Training System** (`lora_trainer/`) - Bot detection and model training

### Services
- **gptbot**: Main FastAPI backend (port 8672)
- **streamlit**: Web chat interface (port 8501)  
- **llama-server**: Local LLM server (port 8080)
- **postgres**: Database (port 5432)

## Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key (optional)
- DeepSeek API key (optional)

### 1. Environment Setup
Create a `.env` file in the root directory:

```env
# Model Configuration
MODEL_OPERATOR=openai  # Options: openai, deepseek, llama
OPEN_AI_API_KEY=your_openai_api_key_here
PROXY_URL=your_proxy_url_here
USE_PROXY=false

# Server Configuration  
PORT=8672

# Database Configuration
DB_USER=student
DB_PASSWORD=student_pass
DB_HOST=postgres
DB_PORT=5432
DB_NAME=chat_db
```

### 2. Start All Services
```bash
docker-compose up -d --build
```

### 3. Access the Application
- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8672/docs
- **Direct API**: http://localhost:8672

## Configuration

### Model Operators
Change the `MODEL_OPERATOR` in your `.env` file:

- `openai`: Uses OpenAI GPT models (requires API key)
- `deepseek`: Uses DeepSeek models (requires API key) 
- `llama`: Uses local LLM server (no API key needed)

### API Keys
- **OpenAI**: Get from https://platform.openai.com/api-keys
- **DeepSeek**: Get from https://platform.deepseek.com/

## API Usage

### Send a Message
```bash
curl -X POST "http://localhost:8672/get_message" \
  -H "Content-Type: application/json" \
  -d '{
    "dialog_id": "550e8400-e29b-41d4-a716-446655440000",
    "last_msg_text": "Hello, how are you?",
    "last_message_id": "550e8400-e29b-41d4-a716-446655440001"
  }'
```

### Response Format
```json
{
  "new_msg_text": "I'm doing well, thank you! How are you today?",
  "dialog_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## Development

### Local Development (Without Docker)
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up database (PostgreSQL required)

3. Run the application:
```bash
python main.py
```

### Web Interface Development
Navigate to the `web/` directory:
```bash
cd web
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Training & Bot Detection

### LoRA Training
The project includes advanced training capabilities in the `lora_trainer/` folder:

- **Experiment Tracking**: Track training experiments
- **Bot Detection**: Train models to detect bot vs human conversations
- **Model Fine-tuning**: Fine-tune models for better human-like responses

See `lora_trainer/USAGE.md` for detailed training instructions.

### Running Experiments
```bash
cd lora_trainer
python run_experiment.py
```

### View Experiment Results
```bash
python view_experiments.py
```

## Deployment

### Production Deployment
1. Update the `.env` file with production values
2. Use production-ready database settings
3. Configure proper proxy settings if needed
4. Run with docker-compose:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Scaling
- The FastAPI backend can be scaled horizontally
- Use a load balancer for multiple backend instances
- Consider using Redis for session storage in scaled deployments

## Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure your API keys are correctly set in `.env`
   - Check API key permissions and quota

2. **Docker Build Failures**
   - Run `docker-compose down -v` to clean up
   - Rebuild with `docker-compose up --build`

3. **Database Connection Issues**
   - Ensure PostgreSQL is healthy: `docker-compose ps`
   - Check database logs: `docker-compose logs postgres`

4. **Model Download Issues**
   - For local models, ensure sufficient disk space
   - Check internet connectivity for model downloads

### Logs
View logs for specific services:
```bash
docker-compose logs gptbot
docker-compose logs streamlit  
docker-compose logs llama-server
docker-compose logs postgres
```

## Project Structure

```
youarebot-bot/
├── main.py                 # FastAPI backend
├── create_answer.py        # LLM query handling
├── config.py              # Configuration management
├── database.py            # Database operations
├── models_fast_api.py     # Pydantic models
├── preprompt.txt          # System prompt
├── web/                   # Streamlit web interface
│   ├── streamlit_app.py
│   ├── app/models.py
│   └── requirements.txt
├── llm/                   # Local LLM server
│   ├── simple_server.py
│   └── Dockerfile.simple
├── lora_trainer/          # Training and experiments
│   ├── experiment_tracker.py
│   ├── run_experiment.py
│   ├── test_bot_detector.py
│   └── USAGE.md
└── docker-compose.yaml   # Service orchestration
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the "You are Bot" research project.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing issues in the repository
3. Create a new issue with detailed information

## Authors
- @mak.a.novikov
- @m.semchinov 