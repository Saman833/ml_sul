
"""
Test script for the fine-tuned LoRA bot detection model.
Bot probability reduced by 30% for more lenient detection.

Usage:
    python test_bot_detector.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import warnings
warnings.filterwarnings('ignore')

def load_model():
    """Load the fine-tuned LoRA model"""
    print("Loading bot detection model...")
    
    model_path = "./lora-bot-detector/checkpoint-315"
    
    try:
        # Load LoRA configuration
        config = PeftConfig.from_pretrained(model_path)
        print(f"Loaded LoRA config: rank={config.r}, alpha={config.lora_alpha}")
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=2
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_bot(text, model, tokenizer, bot_reduction=-0.3):
    """
    Predict if text is from a bot or human with adjusted probabilities
    
    Args:
        text: Input text to classify
        model: Trained LoRA model
        tokenizer: Model tokenizer
        bot_reduction: How much to reduce bot probability (0.3 = 30%)
    """
    # Tokenize input
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get original predictions
        original_human_prob = probabilities[0][0].item()
        original_bot_prob = probabilities[0][1].item()
        
        # Apply adjustment: reduce bot probability by specified amount
        adjusted_bot_prob = max(0.0, original_bot_prob - bot_reduction)
        
        # Recalculate human probability to maintain total = 1.0
        adjusted_human_prob = 1.0 - adjusted_bot_prob
        
        # Determine prediction based on adjusted probabilities
        prediction = "BOT" if adjusted_bot_prob > adjusted_human_prob else "HUMAN"
        confidence = max(adjusted_human_prob, adjusted_bot_prob)
        
    return prediction, confidence, adjusted_human_prob, adjusted_bot_prob, original_human_prob, original_bot_prob

def test_examples(model, tokenizer):
    """Test the model with example texts"""
    
    print("\nTesting with ADJUSTED probabilities (Bot -30%)...")
    print("=" * 70)
    
    # Example conversations for testing
    test_cases = [
        {
            "text": "Hello! How are you doing today? I hope you're having a great day!",
            "expected": "Likely human - casual greeting"
        },
        {
            "text": "Thank you for contacting customer service. How may I assist you today?",
            "expected": "Could be bot - formal service language"
        },
        {
            "text": "lol yeah that's so funny haha I can't even 😂😂😂",
            "expected": "Likely human - informal with emojis"
        },
        {
            "text": "Your request has been processed successfully. Reference number: #123456. Is there anything else I can help you with?",
            "expected": "Likely bot - structured response format"
        },
        {
            "text": "Ugh this monday is killing me... coffee isn't working yet ☕",
            "expected": "Likely human - personal, relatable"
        },
        {
            "text": "Please provide your account details so we can verify your identity and proceed with your request.",
            "expected": "Could be bot - procedural language"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Expected: {case['expected']}")
        
        try:
            pred, conf, adj_human, adj_bot, orig_human, orig_bot = predict_bot(
                case["text"], model, tokenizer
            )
            print(f"   Original: Human {orig_human:.1%} | Bot {orig_bot:.1%}")
        except Exception as e:
            print(f"Error processing case {i}: {e}")

def interactive_test(model, tokenizer):
    """Interactive testing mode"""
    
    print("\nInteractive Bot Detection Mode (Bot -30%)")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            text = input("\nEnter text to analyze: ").strip()
            
            if text.lower() == 'quit':
                break
                
            if not text:
                print("Please enter some text.")
                continue
            
            pred, conf, adj_human, adj_bot, orig_human, orig_bot = predict_bot(
                text, model, tokenizer
            )
            print(f"Original: Human {orig_human:.1%} | Bot {orig_bot:.1%}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function"""
    print("LoRA Bot Detection Model Tester")
    print("Bot probability reduced by 30% for more lenient detection")
    
    print("\nChoose testing mode:")
    print("1. Test with example cases")
    print("2. Interactive testing")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    # Load model
    model, tokenizer = load_model()
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    if choice in ['1', '3']:
        test_examples(model, tokenizer)
    
    if choice in ['2', '3']:
        interactive_test(model, tokenizer)

if __name__ == "__main__":
    main() 