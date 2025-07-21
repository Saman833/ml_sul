#!/usr/bin/env python3

import argparse
import subprocess
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading
import os

app = FastAPI()

class ChatRequest(BaseModel):
    messages: list
    model: str = "llama-2-7b-chat"
    temperature: float = 0.7
    max_tokens: int = 1000

class ChatResponse(BaseModel):
    response: str
    model: str

def start_llama_server(model_path: str, host: str = "0.0.0.0", port: int = 8080):
    """Start the llama.cpp server in a subprocess"""
    # List available files for debugging
    print("Available files in /app/llama.cpp:")
    import subprocess
    try:
        result = subprocess.run(["ls", "-la"], cwd="/app/llama.cpp", capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Error listing files: {e}")
    
    # Try different possible server locations
    possible_paths = [
        "./build/bin/server",
        "./build/server",
        "./server",
        "./bin/server"
    ]
    
    server_path = None
    for path in possible_paths:
        full_path = os.path.join("/app/llama.cpp", path)
        print(f"Checking path: {full_path}")
        if os.path.exists(full_path):
            server_path = path
            print(f"Found server at: {path}")
            break
    
    if not server_path:
        print("Could not find llama.cpp server binary. Available paths checked:")
        for path in possible_paths:
            print(f"  - {path}")
        raise FileNotFoundError("Could not find llama.cpp server binary")
    
    cmd = [
        server_path,
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--ctx-size", "2048"
    ]
    
    print(f"Starting llama.cpp server with command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, cwd="/app/llama.cpp")
    return process

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Handle chat completion requests"""
    try:
        # Format messages for llama.cpp
        prompt = ""
        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
            elif role == "system":
                prompt += f"System: {content}\n"
        
        prompt += "Assistant: "
        
        # Call llama.cpp server
        llama_response = requests.post(
            "http://localhost:8080/completion",
            json={
                "prompt": prompt,
                "n_predict": request.max_tokens,
                "temperature": request.temperature,
                "stop": ["User:", "System:"]
            },
            timeout=30
        )
        
        if llama_response.status_code != 200:
            raise HTTPException(status_code=500, detail="LLM server error")
        
        response_data = llama_response.json()
        response_text = response_data.get("content", "")
        
        return ChatResponse(
            response=response_text,
            model=request.model
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy"}
        else:
            return {"status": "unhealthy"}
    except:
        return {"status": "unhealthy"}

def download_model_if_needed(model_path: str):
    """Download the model if it doesn't exist"""
    import os
    import subprocess
    
    # Check if we have the wrong model file (old large model)
    old_model_path = "models/llama-2-7b-chat.ggmlv3.q4_0.bin"
    if os.path.exists(old_model_path) and not os.path.exists(model_path):
        print(f"Found old model at {old_model_path}, removing it...")
        os.remove(old_model_path)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, downloading...")
        # Using a verified tiny model (~25MB)
        model_url = "https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-q4_0.gguf"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Download the model with error handling
        print(f"Downloading model from: {model_url}")
        print(f"Downloading model... (this may take a few minutes)")
        try:
            result = subprocess.run(["wget", "-v", model_url, "-O", model_path], 
                                   check=True, capture_output=True, text=True)
            print(f"Model downloaded successfully to {model_path}")
        except subprocess.CalledProcessError as e:
            print(f"Download failed with exit code {e.returncode}")
            print(f"Error output: {e.stderr}")
            print(f"Standard output: {e.stdout}")
            raise
    else:
        print(f"Model found at {model_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the model file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--llama-port", type=int, default=8080, help="Port for llama.cpp server")
    
    args = parser.parse_args()
    
    print(f"Starting with model path: {args.model}")
    
    # Download model if needed
    download_model_if_needed(args.model)
    
    # Check if the model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found after download attempt")
        exit(1)
    
    print(f"Using model: {args.model}")
    
    # Start llama.cpp server in background
    llama_process = start_llama_server(args.model, args.host, args.llama_port)
    
    # Wait a bit for the server to start
    time.sleep(10)
    
    try:
        # Start FastAPI server
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        # Clean up
        llama_process.terminate()
        llama_process.wait()

if __name__ == "__main__":
    main() 