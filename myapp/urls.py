from django.urls import path
from django.contrib.auth.decorators import login_required
from . import views

app_name = 'myapp'  # Define an application namespace

urlpatterns = [
    # Public pages
    path('', views.main_view, name='main'),
    path('home/', views.home_view, name='home'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),  # Custom login view
    
    # Protected pages (require login)
    path('profile/', (views.profile_page_view), name='profile'),
    path('edit-profile/', (views.edit_profile), name='edit_profile'),
    path('dashboard/', (views.dashboard_view), name='dashboard'),
    path('mood_status/', (views.mood_status_view), name='mood_status'),
    path('mood-status/', (views.mood_status_view), name='mood-status'),  # Keep old URL for backward compatibility
    path('journal/', (views.journal_view), name='journal'),
    path('change-password/', (views.change_password_view), name='change_password'),
    path('moodchange/', (views.moodchange), name='moodchange'),
    
    # Chatbot APIs (all protected)
    path('chat/api/', (views.chatbot_api), name='chatbot_api'),
    path('chat/feedback/', (views.chatbot_feedback_api), name='chatbot_feedback_api'),
    path('chat/update-context/', (views.update_chatbot_context), name='update_chatbot_context'),
    path('new_chat/', (views.new_chat), name='new_chat'),
    path('chat/stats/', (views.chatbot_stats_api), name='chatbot_stats_api'),
    
    # Video streaming (protected)
    path('video_feed/', (views.video_feed), name='video_feed'),
    path('start_camera/', (views.start_camera), name='start_camera'),
    path('stop_camera/', (views.stop_camera), name='stop_camera'),
    
    # Authentication URLs
    path('logout/', (views.logout_view), name='logout'),

    path('api/emotion-init/', views.emotion_init_api, name='emotion_init_api'),
    path('api/chat/', views.chat_api, name='chat_api'),
    # n8n webhook endpoints
    path('webhook/n8n/chat/', views.n8n_chat_webhook, name='n8n_chat_webhook'),
    path('webhook/n8n/emotion/', views.n8n_emotion_webhook, name='n8n_emotion_webhook'),
]
