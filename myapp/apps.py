from django.apps import AppConfig

class MyAppConfig(AppConfig):
    name = 'myapp'
    default_auto_field = 'django.db.models.BigAutoField'
    
    def ready(self):
        from . import signals  # Import signals to register them