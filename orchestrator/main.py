from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import logging
from typing import List, Optional, Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Orchestrator Gateway", version="1.0.0")

# Request/Response models for /predict endpoint
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    probability: float
    prediction: str
    confidence: str

# Request/Response models for /get_message endpoint
class ChatMessage(BaseModel):
    role: str
    content: str

class GetMessageRequest(BaseModel):
    dialog_id: str
    last_msg_text: str
    last_message_id: Optional[str] = None

class GetMessageResponse(BaseModel):
    new_msg_text: str
    dialog_id: str

# Service configuration
CLASSIFIER_URL = "http://classifier:8000"
LLM_URL = "http://llm:11434"

async def forward_to_classifier(text: str) -> PredictResponse:
    """Forward prediction request to classifier service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CLASSIFIER_URL}/predict",
                json={"text": text},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return PredictResponse(**data)
    except httpx.RequestError as e:
        logger.error(f"Request error to classifier: {e}")
        raise HTTPException(status_code=503, detail="Classifier service unavailable")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from classifier: {e}")
        raise HTTPException(status_code=e.response.status_code, detail="Classifier service error")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def forward_to_llm(message: str) -> str:
    """Forward chat request to LLM service"""
    try:
        async with httpx.AsyncClient() as client:
            # Prepare the chat completion request
            chat_request = {
                "messages": [
                    {"role": "user", "content": message}
                ],
                "model": "llama-2-7b-chat",
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = await client.post(
                f"{LLM_URL}/v1/chat/completions",
                json=chat_request,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract the assistant's message from the response
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                raise HTTPException(status_code=500, detail="Invalid LLM response format")
                
    except httpx.RequestError as e:
        logger.error(f"Request error to LLM: {e}")
        raise HTTPException(status_code=503, detail="LLM service unavailable")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from LLM: {e}")
        raise HTTPException(status_code=e.response.status_code, detail="LLM service error")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Forward prediction request to classifier service
    Returns probability of text being from a bot
    """
    logger.info(f"Received prediction request for text: {request.text[:50]}...")
    
    try:
        result = await forward_to_classifier(request.text)
        logger.info(f"Prediction result: {result.prediction} (probability: {result.probability})")
        return result
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        raise

@app.post("/get_message", response_model=GetMessageResponse)
async def get_message(request: GetMessageRequest):
    """
    Forward chat request to LLM service
    Returns LLM response
    """
    logger.info(f"Received chat request: {request.last_msg_text[:50]}...")
    
    try:
        llm_response = await forward_to_llm(request.last_msg_text)
        
        result = GetMessageResponse(
            new_msg_text=llm_response,
            dialog_id=request.dialog_id
        )
        
        logger.info(f"LLM response: {llm_response[:50]}...")
        return result
        
    except Exception as e:
        logger.error(f"Error in get_message endpoint: {e}")
        raise

@app.get("/health")
async def health_check():
    """Check health of orchestrator and downstream services"""
    health_status = {
        "orchestrator": "healthy",
        "services": {}
    }
    
    # Check classifier service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{CLASSIFIER_URL}/health", timeout=5.0)
            if response.status_code == 200:
                health_status["services"]["classifier"] = "healthy"
            else:
                health_status["services"]["classifier"] = "unhealthy"
    except Exception:
        health_status["services"]["classifier"] = "unreachable"
    
    # Check LLM service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{LLM_URL}/health", timeout=5.0)
            if response.status_code == 200:
                health_status["services"]["llm"] = "healthy"
            else:
                health_status["services"]["llm"] = "unhealthy"
    except Exception:
        health_status["services"]["llm"] = "unreachable"
    
    return health_status

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Orchestrator Gateway",
        "version": "1.0.0",
        "description": "API gateway for bot detection and chat services",
        "endpoints": {
            "predict": "POST /predict - Bot detection (forwards to classifier)",
            "get_message": "POST /get_message - Chat completion (forwards to LLM)",
            "health": "GET /health - Health check"
        },
        "downstream_services": {
            "classifier": CLASSIFIER_URL,
            "llm": LLM_URL
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 