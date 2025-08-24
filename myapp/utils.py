# myapp/utils.py
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)

# Global model instances
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info("Loading DialoGPT model...")
        try:
            model_name = "microsoft/DialoGPT-medium"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    return tokenizer, model

def generate_response(history, mood=""):
    try:
        tokenizer, model = load_model()
        if tokenizer is None or model is None:
            raise Exception("Model not loaded")
        
        # Format conversation history for DialoGPT.
        # Each turn is separated by the EOS token.
        chat_history_str = ""
        # Use last 5 turns for context to avoid overly long inputs
        for turn in history[-5:]:
            chat_history_str += turn['content'] + tokenizer.eos_token

        # Add mood context to the last utterance if available
        if mood and mood != 'neutral':
             chat_history_str += f" [User's current mood is {mood}]"

        # Tokenize input
        inputs = tokenizer(
            chat_history_str,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        input_ids = inputs.input_ids
        
        # Generate response using sampling
        outputs = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 128,  # Generate up to 128 new tokens
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
        )
        
        # Decode only the newly generated tokens
        response = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        fallbacks = [
            "I'm having some trouble understanding that. Could you rephrase?",
            "Let me think about that differently...",
            "Could you tell me more about that?",
            "I'm still learning about human emotions. Could you explain differently?"
        ]
        import random
        return random.choice(fallbacks)