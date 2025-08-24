from django.db import models
from django.contrib.auth.models import User
from django.conf import settings

class ChatMessage(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='chat_messages')
    message_text = models.TextField(blank=True, null=True)
    emotion = models.CharField(max_length=100, blank=True, null=True)  # Store detected emotion
    timestamp = models.DateTimeField(auto_now_add=True)
    # You might add other fields like 'is_ai_response' if needed

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.user.username}: {self.message_text if self.message_text else 'No message'}"

class JournalEntry(models.Model):
    """Represents a user's journal entry."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='journal_entries')
    content = models.TextField()
    # You might want to add a field for the mood associated with the entry
    # mood = models.CharField(max_length=50, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at'] # Order by newest first
        verbose_name_plural = "Journal Entries"

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    avatar = models.ImageField(upload_to='avatars/', default='avatars/default.png')
    bio = models.TextField(max_length=500, blank=True)
    location = models.CharField(max_length=100, blank=True)


class N8nChatEvent(models.Model):
    user_id = models.CharField(max_length=128, blank=True, null=True)
    payload = models.JSONField()
    received_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"N8nChatEvent {self.user_id} @ {self.received_at}"


class N8nEmotionEvent(models.Model):
    user_id = models.CharField(max_length=128, blank=True, null=True)
    emotion = models.CharField(max_length=64, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    payload = models.JSONField()
    received_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"N8nEmotionEvent {self.user_id} {self.emotion} @ {self.received_at}"