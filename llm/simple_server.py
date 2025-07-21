#!/usr/bin/env python3

from fastapi import FastAPI
from pydantic import BaseModel
import re
import random
from typing import List, Optional

app = FastAPI(title="LLM Chat Server", version="1.0.0")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "llama-2-7b-chat"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]

# Enhanced response patterns
def generate_response(messages: List[ChatMessage]) -> str:
    # Get the last user message
    user_message = ""
    for msg in messages:
        if msg.role == "user":
            user_message = msg.content
    
    message_lower = user_message.lower()
    
    # Weather-related responses
    if any(word in message_lower for word in ['weather', 'rain', 'sun', 'cloud', 'temperature', 'hot', 'cold']):
        weather_responses = [
            "The weather today is quite pleasant! It's partly cloudy with a gentle breeze.",
            "I'd say it's a perfect day for a walk - not too hot, not too cold, with some nice sunshine.",
            "The forecast shows scattered clouds with temperatures in the mid-70s. Great weather for outdoor activities!",
            "Weather can be quite unpredictable, but today feels like one of those perfect spring days.",
            "The current conditions are ideal - clear skies and comfortable temperatures around 72°F."
        ]
        return random.choice(weather_responses)
    
    # Greeting responses
    elif any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        greeting_responses = [
            "Hello there! Great to meet you. How can I help you today?",
            "Hi! I'm excited to chat with you. What's on your mind?",
            "Hey! Welcome to our conversation. What would you like to talk about?",
            "Greetings! I'm here and ready to chat. How are you doing?",
            "Hello! Nice to see you here. What brings you to chat today?"
        ]
        return random.choice(greeting_responses)
    
    # How are you responses
    elif any(phrase in message_lower for phrase in ['how are you', 'how do you do', 'what\'s up']):
        status_responses = [
            "I'm doing fantastic, thank you for asking! How about yourself?",
            "I'm great! Always excited to have interesting conversations. How are you?",
            "Doing wonderful! I love chatting with people. What's new with you?",
            "I'm excellent! Ready for whatever topic you'd like to discuss. How's your day going?",
            "Feeling great! Thanks for checking in. How are things on your end?"
        ]
        return random.choice(status_responses)
    
    # Good responses
    elif any(phrase in message_lower for phrase in ['good', 'fine', 'great', 'excellent', 'awesome']):
        positive_responses = [
            "That's wonderful to hear! I'm glad you're doing well.",
            "Excellent! It's always great when things are going good.",
            "That's fantastic! What's making your day so good?",
            "I'm so happy to hear that! Positive energy is contagious.",
            "That's amazing! It sounds like you're in a great mood."
        ]
        return random.choice(positive_responses)
    
    # Help/question responses
    elif any(word in message_lower for word in ['help', 'question', 'what', 'how', 'why', 'when', 'where']):
        help_responses = [
            "I'd be happy to help! Could you tell me more about what you're looking for?",
            "Great question! I'll do my best to provide a helpful answer.",
            "I'm here to assist! What specific information can I help you with?",
            "Let me see what I can do to help you with that!",
            "That's an interesting question! I'd love to explore that topic with you."
        ]
        return random.choice(help_responses)
    
    # Random characters or unclear input
    elif len(user_message.strip()) < 4 or not re.search(r'[a-zA-Z]', user_message):
        unclear_responses = [
            "I'm not quite sure what you mean. Could you elaborate a bit more?",
            "That's an interesting input! Could you tell me more about what you're thinking?",
            "I'd love to understand better. Could you rephrase that?",
            "Hmm, I'm not following. Could you give me a bit more context?",
            "That's intriguing! What did you have in mind?"
        ]
        return random.choice(unclear_responses)
    
    # Default conversational responses
    else:
        default_responses = [
            "That's interesting! Tell me more about that.",
            "I see what you mean. What are your thoughts on that?",
            "That's a great point! I'd love to hear more of your perspective.",
            "Fascinating! How do you feel about that topic?",
            "That's worth discussing! What's your experience with that?",
            "I appreciate you sharing that. What else is on your mind?",
            "That's thought-provoking! What led you to think about that?"
        ]
        return random.choice(default_responses)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests compatible with OpenAI API"""
    import time
    import uuid
    
    response_text = generate_response(request.messages)
    
    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model or "llama-2-7b-chat",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop"
            )
        ]
    )
    
    return response

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "llm-chat-server"}

@app.get("/")
async def root():
    return {
        "service": "LLM Chat Server",
        "version": "1.0.0",
        "endpoints": ["/v1/chat/completions", "/health"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434) 