def send_to_n8n_chat(user_id, user_message):
    """Send chat message to n8n and get AI response"""
    try:
        n8n_url = os.getenv('N8N_CHAT_URL')
        if not n8n_url:
            return {"success": False, "response": "Configuration missing"}
        payload = {
            "user_id": str(user_id),
            "user_message": str(user_message),
            "timestamp": datetime.now().isoformat()
        }
        response = requests.post(n8n_url, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "response": "Service temporarily unavailable"}
    except Exception as e:
        print(f"n8n chat error: {e}")
        return {"success": False, "response": "I'm having trouble responding. Please try again."}

import json
import os
import random
import re
import pickle
import logging
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
import time

# --- n8n chat integration function ---
def send_to_n8n_chat(user_id, user_message):
    """Send chat message to n8n and get AI response"""
    try:
        n8n_url = os.getenv('N8N_CHAT_URL')
        if not n8n_url:
            return {"success": False, "response": "Configuration missing"}
        payload = {
            "user_id": str(user_id),
            "user_message": str(user_message),
            "timestamp": datetime.now().isoformat()
        }
        response = requests.post(n8n_url, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "response": "Service temporarily unavailable"}
    except Exception as e:
        print(f"n8n chat error: {e}")
        return {"success": False, "response": "I'm having trouble responding. Please try again."}

# Load environment variables
load_dotenv()
# Get Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not found in environment variables. Please add GEMINI_API_KEY to your .env file.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationMemory:
    """Manages conversation history and user preferences"""
    
    def __init__(self, max_memory_size=100):
        self.max_memory_size = max_memory_size
        self.conversations = {}  # user_id -> conversation history
        self.user_preferences = {}  # user_id -> preferences
        self.conversation_topics = {}  # user_id -> current topics
        
    def add_message(self, user_id: str, message: str, is_user: bool = True, emotion: Optional[str] = None):
        """Add a message to user's conversation history"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
            
        conversation_entry = {
            'message': message,
            'is_user': is_user,
            'timestamp': datetime.now(),
            'emotion': emotion
        }
        
        self.conversations[user_id].append(conversation_entry)
        
        # Keep only recent conversations
        if len(self.conversations[user_id]) > self.max_memory_size:
            self.conversations[user_id] = self.conversations[user_id][-self.max_memory_size:]
    
    def get_recent_context(self, user_id: str, num_messages: int = 6) -> List[Dict]:
        """Get recent conversation context"""
        if user_id not in self.conversations:
            return []
        return self.conversations[user_id][-num_messages:]
    
    def extract_topics(self, user_id: str) -> List[str]:
        """Extract current conversation topics"""
        if user_id not in self.conversations:
            return []
            
        recent_messages = self.conversations[user_id][-10:]  # Last 10 messages
        topics = []
        
        # Enhanced topic extraction based on keywords
        topic_keywords = {
            'work': ['work', 'job', 'office', 'boss', 'colleague', 'meeting', 'deadline', 'fired', 'unemployed', 'career', 'workplace', 'workload'],
            'family': ['family', 'mom', 'dad', 'sister', 'brother', 'parents', 'kids', 'children'],
            'health': ['health', 'sick', 'doctor', 'hospital', 'pain', 'medicine', 'exercise'],
            'relationships': ['boyfriend', 'girlfriend', 'partner', 'relationship', 'love', 'dating'],
            'school': ['school', 'college', 'university', 'study', 'exam', 'homework', 'class', 'test', 'failed', 'presentation', 'college event'],
            'hobbies': ['hobby', 'music', 'sports', 'reading', 'gaming', 'cooking', 'travel'],
            'emotions': ['happy', 'sad', 'angry', 'anxious', 'stressed', 'excited', 'worried', 'nervous'],
            'financial': ['money', 'rent', 'bills', 'pay', 'financial', 'debt', 'expenses', 'cost'],
            'transportation': ['bike', 'car', 'transport', 'late', 'broken', 'start', 'wont start', 'commute'],
            'social': ['social', 'party', 'crowd', 'meeting people', 'social event', 'social anxiety'],
            'preparation': ['prep', 'preparation', 'prepare', 'planning', 'organize', 'practice']
        }
        
        for message_data in recent_messages:
            if message_data['is_user']:
                message_lower = message_data['message'].lower()
                for topic, keywords in topic_keywords.items():
                    if any(keyword in message_lower for keyword in keywords):
                        if topic not in topics:
                            topics.append(topic)
        
        return topics
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences based on conversation history"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        return self.user_preferences[user_id]
    
    def update_preferences(self, user_id: str, preferences: Dict):
        """Update user preferences"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        self.user_preferences[user_id].update(preferences)

class EmoticonChatbot:
    def __init__(self, intents_path=None):
        # Initialize conversation memory
        self.memory = ConversationMemory()
        
        # Generation parameters
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 40
        
        # Configuration flags
        self.use_internet_search = False  # Flag to control internet search feature
        
        self._initialize_chat()
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Gemini model using env-configurable model names"""
        try:
            # model names come from env (set in settings.py)
            self.model_name = os.getenv('DEFAULT_GEMINI_MODEL', 'gemini-1.0-pro-latest-001')
            self.fallback_model = os.getenv('GEMINI_FALLBACK_MODEL', 'gemini-1.0-mini')
            self.model = genai.GenerativeModel(self.model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )
            )
            print(f"✅ EmoticonChatbot model initialized with Gemini API ({self.model_name})")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            print("⚠️ Warning: Could not initialize Gemini model. Please check your API key and internet connection.")

    def _call_gemini_with_fallback(self, prompt: str, max_attempts: int = 2):
        """Call Gemini model with fallback model on failure."""
        last_exc = None
        attempts = 0
        while attempts < max_attempts:
            model_to_use = self.model_name if attempts == 0 else self.fallback_model
            try:
                # instantiate model for each attempt
                model = genai.GenerativeModel(model_to_use,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                    )
                )
                # Use chat or direct generate depending on availability
                try:
                    chat = model.start_chat(history=[])
                    resp = chat.send_message(prompt)
                    if hasattr(resp, 'text'):
                        return resp.text
                    return str(resp)
                except Exception:
                    # fallback to direct generation API
                    resp = model.generate_content(prompt)
                    if hasattr(resp, 'text'):
                        return resp.text
                    return str(resp)
            except Exception as e:
                logger.warning(f"Gemini attempt {attempts+1} failed for {model_to_use}: {e}")
                last_exc = e
                attempts += 1
                time.sleep(1 * attempts)
        raise last_exc
            
    def _initialize_chat(self):
        """Initialize a new chat session"""
        try:
            self.chat = self.model.start_chat(history=[])
            print("✅ New chat session initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing chat session: {str(e)}")
            return False
            
    def create_new_chat(self):
        """Create a new chat session and clear conversation history"""
        try:
            # Clear conversation memory for the user
            self.memory = ConversationMemory()
            
            # Initialize new chat session
            success = self._initialize_chat()
            
            return success
        except Exception as e:
            logger.error(f"Error creating new chat: {str(e)}")
            return False
            
        # Load intents for fallback responses
        try:
            if intents_path and os.path.exists(intents_path):
                with open(intents_path, "r", encoding="utf-8") as f:
                    self.intents = json.load(f)["intents"]
                    print("✅ Intents loaded successfully")
            else:
                self.intents = []
                logger.info("No intents file provided or found. Using empty intents list.")
        except Exception as e:
            logger.warning(f"Could not load intents: {e}")
            self.intents = []
        
        # No need for duplicate intents loading code

    def _format_conversation_for_model(self, messages: List[Dict]) -> str:
        """Format conversation history using the model's chat template"""
        formatted_messages = []
        
        for msg in messages:
            if msg['is_user']:
                formatted_messages.append(f"User: {msg['message']}")
            else:
                formatted_messages.append(f"Assistant: {msg['message']}")
        
        # Join with EOS token as per the chat template
        conversation_text = self.tokenizer.eos_token.join(formatted_messages)
        return conversation_text

    def _generate_contextual_prompt(self, user_message: str, user_id: str, detected_emotion: Optional[str] = None) -> str:
        """Generate a contextual prompt incorporating conversation history and user context"""
        
        # Get recent conversation context
        recent_context = self.memory.get_recent_context(user_id, num_messages=8)
        
        # Extract current topics
        current_topics = self.memory.extract_topics(user_id)
        
        # Get user preferences
        preferences = self.memory.get_user_preferences(user_id)
        
        # Build context-aware prompt
        context_parts = []
        
        # Add emotional context if available
        if detected_emotion and detected_emotion != 'neutral':
            emotion_contexts = {
                'happy': "The user appears to be in a positive mood.",
                'sad': "The user seems to be feeling down or sad.",
                'angry': "The user appears to be frustrated or angry.",
                'fear': "The user seems to be experiencing anxiety or fear.",
                'surprise': "The user appears to be surprised or shocked.",
                'disgust': "The user seems to be feeling disgusted or upset."
            }
            context_parts.append(emotion_contexts.get(detected_emotion.lower(), ""))
        
        # Add topic context if available
        if current_topics:
            context_parts.append(f"Current conversation topics: {', '.join(current_topics)}.")
        
        # Add conversation history with better context
        if recent_context:
            conversation_text = self._format_conversation_for_model(recent_context)
            context_parts.append(conversation_text)
        
        # Add specific context instructions for better responses
        context_parts.append("Remember to respond appropriately to the user's specific situation and emotions.")
        
        # Add current user message
        context_parts.append(f"User: {user_message}")
        context_parts.append("Assistant:")
        
        return " ".join(context_parts)

    def _generate_response(self, prompt: str) -> str:
        """Generate response using Gemini API"""
        try:
            # Format the prompt with context
            formatted_prompt = f"""You are EmoticonAI, an empathetic and supportive AI assistant. Your responses should be understanding, helpful, and tailored to the user's emotional state.

Context: {prompt}

Respond in a natural, conversational way while being empathetic and supportive."""
            
            # Ensure we have a valid chat instance (if required by the genai client)
            if not hasattr(self, 'chat') or self.chat is None:
                try:
                    self.chat = self.model.start_chat(history=[])
                except Exception as e:
                    logger.error(f"Failed to initialize chat: {e}")
                    # proceed; some genai flows may not require an active chat object

            # Prefer n8n if enabled
            n8n_enabled = os.getenv('ENABLE_N8N_CHAT', 'false').lower() in ('1', 'true', 'yes')
            n8n_url = os.getenv('N8N_CHAT_URL', '')
            if n8n_enabled and n8n_url:
                try:
                    user_message = prompt.split("User: ")[-1].split("Assistant:")[0].strip()
                    n8n_resp = send_to_n8n_chat("anonymous", user_message)
                    if n8n_resp.get('success'):
                        return n8n_resp.get('response')
                except Exception:
                    logger.exception('n8n call failed; falling back to Gemini')

            # If n8n not used or failed, call Gemini with fallback wrapper
            try:
                generated_text = self._call_gemini_with_fallback(formatted_prompt)
                cleaned_response = self._clean_response(generated_text)
                if self._is_quality_response(cleaned_response) and len(cleaned_response.split()) > 5:
                    return cleaned_response
                else:
                    return self._get_contextual_fallback_response(prompt.split("User: ")[-1].split("Assistant:")[0].strip())
            except Exception as e:
                logger.exception('Gemini calls failed')
                return self._get_contextual_fallback_response(prompt.split("User: ")[-1].split("Assistant:")[0].strip())
                
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error with Gemini API: {str(e)}")
                return "I'm experiencing connection issues right now. Please check your internet connection and try again in a moment."
                    
            except Exception as e:
                logger.error(f"Unexpected error with Gemini API: {str(e)}")
                return "I apologize for the technical difficulties. I'm having trouble processing your request right now. Please try again in a few minutes."
            
            # If we get here, use fallback with the original prompt
            user_message = prompt.split("User: ")[-1].split("Assistant:")[0].strip()
            return self._get_contextual_fallback_response(user_message)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Extract user message from prompt for fallback
            try:
                user_message = prompt.split("User: ")[-1].split("Assistant:")[0].strip()
                return self._get_contextual_fallback_response(user_message)
            except:
                return "I apologize, but I'm having trouble understanding. Could you please rephrase that?"

    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        if not response:
            return ""
        
        # Remove any remaining prompt artifacts
        response = re.sub(r'User:.*?Assistant:', '', response, flags=re.DOTALL)
        response = re.sub(r'Assistant:', '', response)
        
        # Remove repetitive patterns
        response = re.sub(r'\b(agent|Agent)\s+(agent|Agent)\b', '', response)
        response = re.sub(r'\b(agent|Agent)[:\-\?\!]*\s*', '', response)
        
        # Remove extra whitespace and newlines
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Ensure response ends properly
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response

    def _is_quality_response(self, response: str) -> bool:
        """Check if the generated response is of good quality"""
        if not response or len(response.strip()) < 15:
            return False
        
        # Check for repetitive patterns
        words = response.split()
        if len(words) < 5:
            return False
        
        # Check for too many repeated words
        word_counts = {}
        for word in words:
            word_counts[word.lower()] = word_counts.get(word.lower(), 0) + 1
        
        # If any word appears more than 2 times in a short response, it's likely incoherent
        for count in word_counts.values():
            if count > 2 and len(words) < 15:
                return False
        
        # Check for common incoherent patterns
        incoherent_patterns = [
            'agent agent', 'agent:', 'agent-', 'agent?', 'agent!',
            'agentagent', 'agent agent agent', 'agent agent agent agent',
            'that that', 'the the', 'is is', 'are are', 'you you', 'i i',
            '::', ':::', '::::', '? ?', '! !', '...', '....'
        ]
        
        response_lower = response.lower()
        for pattern in incoherent_patterns:
            if pattern in response_lower:
                return False
        
        # Check for excessive punctuation
        if response.count(':') > 2 or response.count('?') > 3 or response.count('!') > 3:
            return False
        
        # Check for proper sentence structure (should start with capital letter)
        if response and not response[0].isupper():
            return False
        
        return True

    def _get_contextual_fallback_response(self, user_message: str) -> str:
        """Get a contextual fallback response based on user message and conversation history"""
        
        message_lower = user_message.lower()
        
        # Specific context-based responses for common situations
        context_responses = {
            'meeting_preparation': {
                'keywords': ['meeting', 'presentation', 'event', 'host', 'college event', 'conference', 'speech'],
                'responses': [
                    "Meeting preparation can be nerve-wracking! Here are some quick stress-reducers: Take deep breaths, practice your key points, arrive early, and remember - you're prepared. What specific part of the meeting is worrying you most?",
                    "College events can feel overwhelming. Try this: visualize success, practice your material, get a good night's sleep, and have a backup plan. What's the main goal of your meeting?",
                    "Meeting stress is totally normal! Try these: organize your thoughts beforehand, dress comfortably, bring water, and remember everyone gets nervous. What would help you feel more confident?",
                    "For event hosting stress: prepare your materials early, do a practice run, focus on your audience's needs, and remember you're there to help them. What's the event about?",
                    "Meeting anxiety is common! Try: deep breathing exercises, positive self-talk, preparing talking points, and arriving 10 minutes early. What's your biggest concern about tomorrow?"
                ]
            },
            'interview_preparation': {
                'keywords': ['interview', 'job interview', 'interview prep', 'interview stress', 'interview anxiety'],
                'responses': [
                    "Interview preparation can be stressful! Here's what helps: research the company, practice common questions, prepare your 'tell me about yourself' story, and get plenty of sleep. What type of interview is it?",
                    "Interview anxiety is normal! Try: practicing with a friend, preparing questions to ask them, dressing professionally, and arriving early. What's the position you're interviewing for?",
                    "For interview stress: research the role thoroughly, prepare specific examples of your achievements, practice your handshake, and remember - they want to hire someone great like you! What's your biggest worry?",
                    "Interview prep tips: review your resume, prepare STAR method answers, plan your outfit the night before, and practice your elevator pitch. What would help you feel more prepared?",
                    "Interview nerves are common! Try: visualization techniques, positive affirmations, preparing thoughtful questions, and remembering your worth. What's the company or role?"
                ]
            },
            'exam_preparation': {
                'keywords': [ 'final exam', 'study stress', 'exam anxiety', 'test prep', 'exam prep'],
                'responses': [
                    "Exam preparation stress is real! Try: breaking study sessions into chunks, using active recall techniques, getting enough sleep, and taking regular breaks. What subject are you studying?",
                    "Exam anxiety is common! Here's what helps: create a study schedule, use practice tests, teach the material to someone else, and remember - you've got this! What's your biggest study challenge?",
                    "For exam stress: organize your notes, use the Pomodoro technique, stay hydrated, and avoid cramming. What would help you feel more confident about the exam?",
                    "Exam prep tips: review past assignments, form study groups, use flashcards, and get plenty of rest. What subject is causing you the most stress?",
                    "Study stress management: take breaks every 45 minutes, exercise to reduce anxiety, eat brain-boosting foods, and visualize success. What's your study plan?"
                ]
            },
            'social_anxiety': {
                'keywords': ['social anxiety', 'social stress', 'party anxiety', 'crowd anxiety', 'social event', 'meeting people'],
                'responses': [
                    "Social anxiety can be really challenging! Try: arriving early to get comfortable, having conversation starters ready, focusing on others rather than yourself, and remembering most people feel the same way. What's the social situation?",
                    "Social stress is common! Here's what helps: practice deep breathing, set realistic expectations, have an exit plan, and remember - you don't have to be perfect. What's making you anxious?",
                    "For social anxiety: prepare some topics to discuss, arrive with a friend if possible, focus on listening, and remember everyone has awkward moments. What would help you feel more comfortable?",
                    "Social event tips: dress comfortably, arrive early to get acclimated, have some questions ready, and remember - people are usually friendly. What's the event you're worried about?",
                    "Social anxiety management: practice relaxation techniques, set small goals, remember your strengths, and know it's okay to take breaks. What's your biggest social concern?"
                ]
            },
            'work_stress': {
                'keywords': ['work stress', 'work pressure', 'deadline', 'workload', 'boss stress', 'workplace anxiety'],
                'responses': [
                    "Work stress can be overwhelming! Try: prioritizing tasks, taking short breaks, communicating with your manager, and setting realistic expectations. What's the main source of stress at work?",
                    "Workplace pressure is tough! Here's what helps: break big tasks into smaller ones, use time management techniques, practice saying 'no' when needed, and remember your worth. What's the most stressful part?",
                    "For work stress: create a to-do list, take regular breaks, practice stress-relief techniques, and maintain work-life boundaries. What would help you feel more in control?",
                    "Work stress management: organize your workspace, communicate clearly with colleagues, take lunch breaks away from your desk, and remember - you're doing your best. What's your biggest work challenge?",
                    "Workplace anxiety tips: set clear boundaries, practice time management, seek support from colleagues, and remember your achievements. What's causing the most stress?"
                ]
            },
            'test_failure': {
                'keywords': ['failed', 'test', 'failed in', 'didn\'t pass', 'flunked'],
                'responses': [
                    "I'm sorry to hear about your test. That can be really disappointing. What subject was it?",
                    "Test failures can be tough to handle. How are you feeling about it?",
                    "That's a setback, but it doesn't define your abilities. What do you think went wrong?",
                    "I understand how frustrating that must be. Would you like to talk about what happened?",
                    "Tests can be stressful. What would help you feel better about this?"
                ]
            },
            'job_loss': {
                'keywords': ['fired', 'lost job', 'got fired', 'boss fired', 'unemployed', 'jobless'],
                'responses': [
                    "I'm so sorry to hear about losing your job. That's a really difficult situation to go through.",
                    "Losing your job can be incredibly stressful and scary. How are you coping with this?",
                    "That's a major life change. What are your biggest concerns right now?",
                    "I can only imagine how overwhelming this must feel. What's your plan going forward?",
                    "Job loss affects so many aspects of life. What support do you have right now?"
                ]
            },
            'financial_stress': {
                'keywords': ['rent', 'money', 'bills', 'pay', 'financial', 'debt', 'expenses'],
                'responses': [
                    "Financial stress can be incredibly overwhelming. What's your biggest concern right now?",
                    "Money worries affect so many people. What would help you feel more secure?",
                    "That's a real source of anxiety. Do you have any support or resources you can turn to?",
                    "Financial stress is valid and serious. What's one small step you could take?",
                    "I hear how worried you are about money. What's your most pressing need right now?"
                ]
            },
            'transportation_issues': {
                'keywords': ['bike', 'car', 'transport', 'late', 'broken', 'start', 'wont start'],
                'responses': [
                    "Transportation problems can really throw off your whole day. That must have been frustrating.",
                    "Being late because of transportation issues is so stressful. How did you handle it?",
                    "That's a tough situation - transportation problems can have serious consequences.",
                    "I can see how that would be really upsetting. What happened after that?",
                    "Transportation issues can feel so unfair. How are you feeling about it now?"
                ]
            },
            'joke_request': {
                'keywords': ['joke', 'funny', 'make me laugh', 'tell me a joke'],
                'responses': [
                    "Why don't scientists trust atoms? Because they make up everything! 😄",
                    "What did the ocean say to the beach? Nothing, it just waved! 🌊",
                    "Why couldn't the bicycle stand up by itself? It was two tired! 🚲",
                    "How do you organize a space party? You planet! 🪐",
                    "What do you call a fake noodle? An impasta! 🍝"
                ]
            },
            'greeting': {
                'keywords': ['hi', 'hello', 'hey', 'howdy', 'hi there', 'hey there'],
                'responses': [
                    "Hello! 😊 How are you feeling today?",
                    "Hey there! It's always nice to hear from you. What's on your mind?",
                    "Hi! I'm here to listen and support you. How can I help?",
                    "Hey! I hope your day is going well. Want to share anything?",
                    "Hello! I'm glad you reached out. How are you doing?"
                ]
            }
        }
        
        # Check for specific contexts first
        for context, data in context_responses.items():
            if any(keyword in message_lower for keyword in data['keywords']):
                return random.choice(data['responses'])
        
        # Enhanced emotional keyword matching
        emotional_keywords = {
            'sad': ['sad', 'depressed', 'lonely', 'hopeless', 'crying', 'tears', 'grief', 'loss', 'down', 'blue'],
            'angry': ['angry', 'furious', 'mad', 'rage', 'hate', 'frustrated', 'annoyed', 'upset', 'irritated'],
            'anxious': ['anxious', 'worried', 'scared', 'afraid', 'nervous', 'panic', 'stress', 'overwhelmed', 'tense'],
            'happy': ['happy', 'joy', 'excited', 'thrilled', 'elated', 'great', 'wonderful', 'grateful', 'blessed'],
            'tired': ['tired', 'exhausted', 'sleepy', 'worn out', 'fatigued', 'drained', 'burned out']
        }
        
        # Check for emotional keywords
        for emotion, keywords in emotional_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return self.get_emotional_support_response(emotion)
        
        # Check intents for contextual responses
        if self.intents:
            for intent in self.intents:
                for pattern in intent['patterns']:
                    pattern_words = pattern.lower().split()
                    if any(word in message_lower for word in pattern_words if len(word) > 3):
                        return random.choice(intent['responses'])
        
        # Default empathetic responses
        fallback_responses = [
            "I understand what you're saying. Could you tell me more about that?",
            "That's interesting. How does that make you feel?",
            "I'm here to listen. What's on your mind?",
            "Thank you for sharing that with me. How can I support you right now?",
            "I hear you. Would you like to explore this further?",
            "That sounds important. Can you elaborate on that?",
            "I'm here for you. What would be most helpful right now?",
            "Thank you for trusting me with that. How are you feeling about it?",
            "I appreciate you opening up to me. What's your biggest concern?",
            "That's a valid feeling. What do you think would help?",
            "I can sense this is important to you. What would you like to focus on?",
            "Your feelings matter. What's been on your mind lately?",
            "I'm here to support you. What's been challenging for you?",
            "Thank you for sharing with me. How are you coping with this?",
            "I hear the emotion in your words. What would help you feel better?"
        ]
        
        return random.choice(fallback_responses)

    def get_response(self, input_text: str, user_id: str = "default", 
                    detected_emotion: Optional[str] = None, 
                    conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Generate a response to user input with conversation memory and context learning
        
        Args:
            input_text: User's message
            user_id: Unique identifier for the user
            detected_emotion: Emotion detected from facial analysis
            conversation_history: Optional conversation history from frontend
        
        Returns:
            Generated response string
        """
        if not input_text or not isinstance(input_text, str):
            return "Please enter a valid message."

        # --- n8n chat integration ---
        n8n_result = send_to_n8n_chat(user_id, input_text)
        if n8n_result.get("success"):
            final_response = n8n_result.get("response")
            return final_response
        # --- fallback to existing chatbot logic ---
        # Check for positive event with happy emotion
        if detected_emotion and detected_emotion.lower() == 'happy':
            positive_keywords = [
                'hired', 'internship', 'job', 'offer', 'congratulations', 'congrats', 'promotion', 'selected',
                'accepted', 'passed', 'win', 'won', 'success', 'achievement', 'award', 'prize', 'good news',
                'excited', 'yay', 'awesome', 'great news', 'dream', 'goal', 'milestone', 'celebrate', 'happy to share',
                'got the', 'new role', 'new position', 'new job', 'new internship', 'interview cleared', 'joining', 'onboarded'
            ]
            message_lower = input_text.lower()
            if any(kw in message_lower for kw in positive_keywords):
                celebration_responses = [
                    "🎉 Wow, congratulations! That's amazing news! How do you feel about your new opportunity?",
                    "🥳 That's fantastic! I'm so happy for you! What are you most excited about?",
                    "👏 Congratulations on your achievement! You deserve it. How will you celebrate?",
                    "🌟 That's wonderful news! I'm thrilled for you. Tell me more about your new role!",
                    "🙌 Yay! This is such a big milestone. How are you feeling right now?"
                ]
                return random.choice(celebration_responses)
        
        # Check if this is a greeting and we have emotion data
        if self._is_greeting(input_text) and detected_emotion and detected_emotion != 'neutral':
            emotion_greeting = self._get_emotion_based_greeting(detected_emotion)
            if emotion_greeting:
                return emotion_greeting
        
        # Update memory with user message
        self.memory.add_message(user_id, input_text, is_user=True, emotion=detected_emotion)
        
        # Update memory with conversation history if provided
        if conversation_history:
            for entry in conversation_history:
                if isinstance(entry, dict) and 'content' in entry and 'role' in entry:
                    self.memory.add_message(
                        user_id, 
                        entry['content'], 
                        is_user=(entry['role'] == 'user')
                    )
        
        # Generate contextual prompt
        prompt = self._generate_contextual_prompt(input_text, user_id, detected_emotion)
        
        # Generate response
        response = self._generate_response(prompt)
        
        # Update memory with AI response
        self.memory.add_message(user_id, response, is_user=False)
        
        # Update user preferences based on conversation
        self._update_user_preferences(user_id, input_text, response)
        
        return response

    def _update_user_preferences(self, user_id: str, user_message: str, ai_response: str):
        """Update user preferences based on conversation patterns"""
        message_lower = user_message.lower()
        
        # Detect communication style preferences
        preferences = self.memory.get_user_preferences(user_id)
        
        # Update based on message length
        if len(user_message) > 100:
            preferences['prefers_detailed_responses'] = True
        elif len(user_message) < 20:
            preferences['prefers_concise_responses'] = True
        
        # Update based on emotional content
        if any(word in message_lower for word in ['sad', 'depressed', 'lonely']):
            preferences['needs_emotional_support'] = True
        
        if any(word in message_lower for word in ['work', 'job', 'career']):
            preferences['discusses_work'] = True
        
        if any(word in message_lower for word in ['family', 'relationship', 'friend']):
            preferences['discusses_relationships'] = True
        
        # Update preferences
        self.memory.update_preferences(user_id, preferences)

    def get_emotional_support_response(self, emotion: str, intensity: float = 0.5) -> str:
        """Generate emotional support responses based on detected emotion"""
        support_responses = {
            'sad': [
                "I can see you're going through a difficult time. It's okay to feel sad, and I'm here to listen.",
                "Your feelings are valid, and it's completely normal to have moments of sadness. What's on your mind?",
                "I'm sorry you're feeling down. Remember that difficult emotions are temporary, and you're not alone.",
                "It sounds like you're carrying a heavy heart. Would you like to talk about what's troubling you?"
            ],
            'angry': [
                "I can sense your frustration. It's okay to feel angry - it's a natural emotion. What triggered these feelings?",
                "Anger can be overwhelming. Let's take a moment to breathe together. What happened that made you feel this way?",
                "Your anger is valid. Sometimes we need to feel it to process what's happening. Want to explore this together?",
                "I hear the intensity in your words. What would help you feel more at peace right now?"
            ],
            'anxious': [
                "Anxiety can feel overwhelming. Let's take this one breath at a time. What's causing you to feel anxious?",
                "I understand anxiety can be really challenging. You're safe here, and we can work through this together.",
                "It's okay to feel anxious. Sometimes naming our fears helps reduce their power. What are you most worried about?",
                "Anxiety often makes everything feel bigger than it is. Let's break this down into smaller, manageable pieces."
            ],
            'happy': [
                "It's wonderful to see you in such good spirits! What's bringing you joy today?",
                "Your positive energy is contagious! I'm glad you're feeling happy. What made your day special?",
                "It's great to hear you're feeling good! Positive moments like these are worth celebrating.",
                "Your happiness shines through! What's been going well for you lately?"
            ],
            'tired': [
                "It sounds like you're feeling exhausted. Remember to be kind to yourself and take the rest you need.",
                "Being tired can affect everything else. What would help you feel more rested and energized?",
                "It's okay to feel tired. Sometimes we need to slow down and recharge. What's been draining your energy?",
                "Rest is important for your well-being. What would help you feel more refreshed?"
            ]
        }
        
        responses = support_responses.get(emotion.lower(), support_responses.get('sad', []))
        return random.choice(responses)

    def get_conversation_summary(self, user_id: str) -> Dict:
        """Get a summary of the user's conversation history"""
        if user_id not in self.memory.conversations:
            return {"message": "No conversation history available"}
        
        conversations = self.memory.conversations[user_id]
        user_messages = [msg for msg in conversations if msg['is_user']]
        assistant_messages = [msg for msg in conversations if not msg['is_user']]
        
        # Extract topics
        topics = self.memory.extract_topics(user_id)
        
        # Get preferences
        preferences = self.memory.get_user_preferences(user_id)
        
        return {
            "total_exchanges": len(conversations),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "current_topics": topics,
            "user_preferences": preferences,
            "recent_messages": [msg['message'] for msg in conversations[-5:]]
        }

    def reset_conversation(self, user_id: str):
        """Reset conversation history for a specific user"""
        if user_id in self.memory.conversations:
            self.memory.conversations[user_id] = []
            self.memory.user_preferences[user_id] = {}
            # Reset Gemini chat history
            self.chat = self.model.start_chat(history=[])
        print(f"Conversation history reset for user {user_id}")

    def save_memory(self, filepath: str):
        """Save conversation memory to file"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.memory, f)
            print(f"Memory saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def _is_greeting(self, message: str) -> bool:
        """Check if the message is a greeting"""
        greeting_words = ['hi', 'hello', 'hey', 'howdy', 'hi there', 'hey there', 'what\'s up', 'sup', 'yo']
        message_lower = message.lower().strip()
        return any(greeting in message_lower for greeting in greeting_words)
    
    def _get_emotion_based_greeting(self, emotion: str) -> str:
        """Get an emotion-based greeting based on detected emotion"""
        emotion_greetings = {
            'happy': [
                "Hi there! 😊 I can see you're smiling! What's making you so happy today?",
                "Hello! Your smile is contagious! 😄 What's bringing you joy right now?",
                "Hey! I love seeing that happy expression! What's got you in such a great mood?",
                "Hi! Your happiness is radiating! 🌟 What wonderful thing happened today?",
                "Hello there! 😊 That smile of yours is beautiful! What's making your day special?"
            ],
            'sad': [
                "Hi... I can see you're feeling down. 😔 What's on your mind? I'm here to listen.",
                "Hello. I notice you seem sad. 💙 Would you like to talk about what's troubling you?",
                "Hey there. I can see you're having a tough time. What's weighing on your heart?",
                "Hi. I sense you're feeling low. 😔 Remember, it's okay to not be okay. What's going on?",
                "Hello. I can see you're sad. 💙 You don't have to face this alone. What's on your mind?"
            ],
            'angry': [
                "Hi. I can see you're frustrated. 😤 What happened? I'm here to listen.",
                "Hello. I notice you seem angry. What's got you upset? Let's talk about it.",
                "Hey. I can see you're mad about something. What's going on?",
                "Hi. I sense you're frustrated. 😤 It's okay to feel angry. What happened?",
                "Hello. I can see you're upset. What's bothering you? I'm here to help."
            ],
            'fear': [
                "Hi. I can see you're feeling anxious. 😰 What's worrying you? You're safe here.",
                "Hello. I notice you seem scared. What's making you feel afraid?",
                "Hey. I can see you're anxious. 😰 What's on your mind? I'm here to support you.",
                "Hi. I sense you're worried about something. What's causing your anxiety?",
                "Hello. I can see you're feeling fearful. What's troubling you? You're not alone."
            ],
            'surprise': [
                "Hi! 😲 I can see you're surprised! What just happened?",
                "Hello! Wow, you look shocked! What's the news?",
                "Hey! 😲 I can see that surprised expression! What caught you off guard?",
                "Hi! Something unexpected just happened, didn't it? What was it?",
                "Hello! 😲 I can see you're taken aback! What's the surprise?"
            ],
            'disgust': [
                "Hi. I can see you're feeling disgusted. 😣 What happened?",
                "Hello. I notice you seem repulsed by something. What's going on?",
                "Hey. I can see you're feeling disgusted. 😣 What's bothering you?",
                "Hi. I sense you're upset about something. What happened?",
                "Hello. I can see you're feeling disgusted. What's the issue?"
            ],
            'neutral': [
                "Hi! How are you feeling today?",
                "Hello! What's on your mind?",
                "Hey there! How can I help you today?",
                "Hi! How's your day going?",
                "Hello! What would you like to talk about?"
            ]
        }
        
        greetings = emotion_greetings.get(emotion.lower(), emotion_greetings['neutral'])
        return random.choice(greetings)
    
    def load_memory(self, filepath: str):
        """Load conversation memory from file"""
        try:
            with open(filepath, 'rb') as f:
                self.memory = pickle.load(f)
            print(f"Memory loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading memory: {e}")