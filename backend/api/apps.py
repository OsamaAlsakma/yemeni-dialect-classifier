"""
apps.py — Конфигурация Django-приложения REST API.
"""
from django.apps import AppConfig


class ApiConfig(AppConfig):
    """Приложение REST API для классификации диалектов."""
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"
    verbose_name = "REST API классификатора"
