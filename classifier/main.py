from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pytorch
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import os
from typing import List
from pathlib import Path
import json
from peft import PeftModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bot Detection Classifier", version="1.0.0")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    probability: float
    prediction: str
    confidence: str

class ClassifierService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def load_champion_model(self):
        """Search the mounted experiments for a LoRA adapter and load it. If nothing usable is found fall back to a heuristic mock model."""
        try:
            experiments_root = Path("/app/experiments")
            preferred_exp = os.getenv("PREFERRED_EXPERIMENT", "high_performance_model")
            preferred_path = experiments_root / preferred_exp
            search_paths = []
            if preferred_path.exists():
                search_paths.append(preferred_path)
            # add remaining experiments
            search_paths.extend(p for p in experiments_root.iterdir() if p.is_dir() and p not in search_paths)

            adapter_weights = []
            for base in search_paths:
                adapter_weights.extend(list(base.glob("**/adapter_model.safetensors")))

            # Use first found
            if not adapter_weights:
                raise FileNotFoundError("No adapter_model.safetensors found in any experiment")

            adapter_dir = adapter_weights[0].parent
            logger.info(f"Found LoRA adapter in {adapter_dir}")

            # Read adapter_config.json to learn the base model name (fallback to DistilBERT)
            base_model_name = "distilbert-base-uncased"
            cfg_path = adapter_dir / "adapter_config.json"
            if cfg_path.exists():
                try:
                    with cfg_path.open("r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    base_model_name = cfg.get("base_model_name_or_path", base_model_name)
                except Exception as cfg_err:
                    logger.warning(f"Could not parse adapter_config.json: {cfg_err}; using default base model")

            logger.info(f"Loading base model: {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)

            # Attach LoRA adapter
            logger.info("Applying LoRA adapter…")
            self.model = PeftModel.from_pretrained(base_model, str(adapter_dir)).to(self.device)
            self.model.eval()
            logger.info("LoRA model loaded successfully.")

        except Exception as load_err:
            logger.error(f"Failed to load LoRA model: {load_err}. Falling back to heuristic model.")
            self.load_mock_model()
    
    def load_mock_model(self):
        """Load a simple heuristic model as fallback."""
        logger.info("Loading simple heuristic fallback model")
        
        class MockModel:
            def predict(self, text):
                # Simple heuristic: longer texts and certain keywords suggest bot
                text_lower = text.lower()
                score = 0.3  # base probability
                
                # Bot indicators
                if len(text) > 100:
                    score += 0.2
                if any(word in text_lower for word in ['artificial', 'ai', 'model', 'algorithm']):
                    score += 0.3
                if any(word in text_lower for word in ['I am', 'I\'m a', 'as an ai']):
                    score += 0.4
                    
                # Human indicators
                if any(word in text_lower for word in ['uh', 'um', 'like', 'you know']):
                    score -= 0.2
                if '?' in text:
                    score -= 0.1
                    
                return min(max(score, 0.0), 1.0)
        
        self.model = MockModel()
        self.tokenizer = None
        logger.info("Mock model loaded successfully")
    
    def predict(self, text: str) -> dict:
        """Make prediction on input text"""
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            if hasattr(self.model, 'predict'):  # Mock model
                probability = self.model.predict(text)
            else:  # Real transformers model
                if self.tokenizer is None:
                    raise HTTPException(status_code=500, detail="Tokenizer not loaded")
                
                # Tokenize input
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Make prediction
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    # Assuming binary classification: [human, bot]
                    probability = probabilities[0][1].item()  # Bot probability
            
            # Determine prediction and confidence
            prediction = "bot" if probability > 0.5 else "human"
            
            if probability > 0.8 or probability < 0.2:
                confidence = "high"
            elif probability > 0.6 or probability < 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            
            return {
                "probability": round(probability, 4),
                "prediction": prediction,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Global classifier instance
classifier = ClassifierService()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting classifier service...")
    classifier.load_champion_model()
    logger.info("Classifier service ready!")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict if text is from bot or human"""
    try:
        result = classifier.predict(request.text)
        return PredictResponse(**result)
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": classifier.model is not None,
        "service": "bot-detection-classifier"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Bot Detection Classifier",
        "version": "1.0.0",
        "endpoints": ["/predict", "/health"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 