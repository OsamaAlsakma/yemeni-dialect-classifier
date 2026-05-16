"""
apps.py — Конфигурация Django-приложения "classifier".
"""
from django.apps import AppConfig


class ClassifierConfig(AppConfig):
    """Приложение страниц веб-интерфейса классификатора."""
    default_auto_field = "django.db.models.BigAutoField"
    name = "classifier"
    verbose_name = "Классификатор диалектов"
