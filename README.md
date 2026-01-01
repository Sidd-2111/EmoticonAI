# EmoticonAI - AI-Powered Emotional Companion

> **AI Best Friend for Humans** - An intelligent emotional support system combining real-time emotion detection with empathetic AI conversations.

## 🌟 Overview

EmoticonAI is a Django-based web application that provides emotional support through:
- **Real-time webcam emotion detection** using DeepFace and OpenCV
- **Context-aware conversational AI** powered by Google Gemini
- **Conversation memory** for personalized interactions
- **n8n webhook integration** for advanced workflow automation
- **User profiles & mood journaling** for tracking emotional wellness

## 📁 Project Structure

```
EmoticonAI/
├── Emoticon/                 # Django project configuration
│   ├── settings.py          # Project settings
│   ├── urls.py              # Root URL configuration
│   ├── wsgi.py & asgi.py    # Server entry points
├── myapp/                    # Main application
│   ├── backend/             # Core AI modules
│   │   ├── facedetection.py # Emotion detection engine
│   │   └── chatbot.py       # Gemini chatbot integration
│   ├── migrations/          # Database migrations
│   ├── templates/           # HTML templates
│   ├── models.py            # Database models
│   ├── views.py             # Request handlers
│   ├── urls.py              # URL routing
│   ├── forms.py             # Django forms
│   ├── utils.py             # DialoGPT utilities
│   └── signals.py           # Django signals
├── scripts/                  # Helper scripts
├── manage.py                 # Django CLI
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Features

### 1. Emotion Detection System
- **Real-time face detection** using Haar Cascade
- **Emotion analysis** via DeepFace (7 emotions: happy, sad, angry, fear, surprise, disgust, neutral)
- **Webcam streaming** with emotion overlays
- **Global mood tracking** across sessions

### 2. AI Chatbot (Gemini-Powered)
- **Emotion-aware responses** tailored to user's detected mood
- **Conversation memory** (tracks last 20 exchanges per user)
- **Topic extraction** (work, family, health, relationships, school, etc.)
- **Pattern detection** for recurring emotions
- **Fallback to DialoGPT** for offline mode

### 3. User Management
- User registration & authentication
- Profile management with photo upload
- Password change functionality
- Session-based conversation history

### 4. Mood Journaling
- Create and manage journal entries
- Associate moods with journal entries
- Track emotional wellness over time

### 5. n8n Webhook Integration
- Emotion event webhooks with HMAC signature verification
- Chat message webhooks for external processing
- Support for automated workflows and analytics

## 🛠️ Technology Stack

- **Backend**: Django 4.x, Python 3.11+
- **AI/ML**: 
  - Google Gemini API (gemini-1.5-flash)
  - DeepFace (emotion analysis)
  - DialoGPT-medium (fallback chatbot)
- **Computer Vision**: OpenCV, cv2
- **Database**: SQLite (default) / PostgreSQL
- **Authentication**: Django Auth

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- Webcam (for emotion detection)
- Google Gemini API key

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/Sidd-2111/EmoticonAI.git
cd EmoticonAI
```

2. **Create virtual environment**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set environment variables**

Create a `.env` file in the project root:
```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
N8N_WEBHOOK_SECRET=your_n8n_secret
N8N_CHAT_URL=https://your-n8n-instance.com/webhook/chat
N8N_EMOTION_URL=https://your-n8n-instance.com/webhook/emotion
ENABLE_N8N_CHAT=false
ENABLE_N8N_EMOTION=false

# Model Configuration
DEFAULT_GEMINI_MODEL=gemini-1.5-flash
GEMINI_FALLBACK_MODEL=gemini-1.0-pro
```

5. **Run database migrations**
```bash
python manage.py makemigrations
python manage.py migrate
```

6. **Create superuser (optional)**
```bash
python manage.py createsuperuser
```

7. **Run development server**
```bash
python manage.py runserver
```

8. **Access the application**
- Main app: http://localhost:8000/
- Admin panel: http://localhost:8000/admin/

## 📚 API Documentation

### Core Functions

#### Emotion Detection (`myapp/backend/facedetection.py`)

```python
class EmotionDetector:
    def __init__(self):
        # Initializes Haar Cascade classifier
    
    def detect_emotions(self, frame):
        # Detects faces and analyzes emotions
        # Returns: List of dicts with 'box', 'emotions', 'dominant_emotion'
```

**Utility Functions:**
- `get_current_mood()` - Returns current global mood
- `start_detection()` - Activates emotion detection
- `stop_detection()` - Deactivates emotion detection
- `send_to_n8n_init(user_id, emotion, confidence)` - Sends emotion data to n8n
- `draw_results(frame, results)` - Draws bounding boxes and labels

#### Chatbot (`myapp/backend/chatbot.py`)

```python
class ConversationMemory:
    def add_message(user_id, message, is_user, emotion)
    def get_recent_context(user_id, num_messages=6)
    def extract_topics(user_id)
    def get_user_preferences(user_id)

class EmoticonChatbot:
    def get_response(input_text, user_id, detected_emotion, conversation_history):
        # Generates contextual, emotion-aware responses
    
    def get_emotional_support_response(emotion, intensity=0.5):
        # Returns emotion-specific support messages
    
    def reset_conversation(user_id):
        # Clears conversation history for user
```

### URL Endpoints (`myapp/urls.py`)

#### Public Pages
- `/` - Main landing page
- `/home/` - Dashboard/home
- `/signup/` - User registration
- `/login/` - User login

#### Protected Pages
- `/profile/` - User profile view
- `/edit-profile/` - Edit profile
- `/dashboard/` - Main dashboard
- `/journal/` - Mood journal
- `/change-password/` - Password change

#### Emotion & Camera APIs
- `/mood_status/` - GET current mood (JSON)
- `/video_feed/` - Webcam stream with emotion detection
- `/start_camera/` - Start camera (JSON)
- `/stop_camera/` - Stop camera (JSON)

#### Chatbot APIs
- `/chat/api/` - POST chat message
- `/chat/feedback/` - POST feedback
- `/chat/update-context/` - POST emotion context
- `/chat/stats/` - GET conversation statistics
- `/new_chat/` - POST to clear conversation history

#### n8n Webhook Endpoints
- `/webhook/n8n/chat/` - Receives chat events from n8n
- `/webhook/n8n/emotion/` - Receives emotion events from n8n

### Database Models (`myapp/models.py`)

- **Profile** - Extended user profile with personal info
- **ChatMessage** - Stores chat messages with emotions
- **JournalEntry** - User mood journal entries
- **N8nChatEvent** - n8n chat webhook events
- **N8nEmotionEvent** - n8n emotion webhook events

## 🔐 Security Features

- **HMAC signature verification** for n8n webhooks
- **CSRF protection** on all forms
- **Session-based authentication**
- **Password hashing** via Django's built-in system
- **Environment variable management** for sensitive data

## 🎯 Key Functions Summary

### Emotion Detection
| Function | Purpose | Returns |
|----------|---------|----------|
| `EmotionDetector.detect_emotions()` | Analyzes frame for emotions | List of emotion results |
| `get_current_mood()` | Gets global mood state | String (emotion) |
| `send_to_n8n_init()` | Sends emotion to n8n | Dict (response) |

### Chatbot
| Function | Purpose | Returns |
|----------|---------|----------|
| `EmoticonChatbot.get_response()` | Generates AI response | String |
| `ConversationMemory.add_message()` | Stores message | None |
| `ConversationMemory.extract_topics()` | Identifies topics | List of strings |

## 🧪 Testing

```bash
python manage.py test myapp
```

## 📝 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key |
| `N8N_WEBHOOK_SECRET` | ❌ No | Shared secret for n8n webhooks |
| `ENABLE_N8N_CHAT` | ❌ No | Enable n8n chat integration |
| `ENABLE_N8N_EMOTION` | ❌ No | Enable n8n emotion integration |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- **Sidd-2111** 
- **YASH-DHADGE** 

## 🙏 Acknowledgments

- Google Gemini API for conversational AI
- DeepFace library for emotion recognition
- OpenCV for computer vision capabilities
- Microsoft DialoGPT for fallback chatbot
- Django framework

---

**Made with ❤️ for emotional wellness and AI companionship**
