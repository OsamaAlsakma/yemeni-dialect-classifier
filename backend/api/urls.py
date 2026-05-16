"""
urls.py — URL-маршруты REST API.
"""
from django.urls import path
from . import views

app_name = "api"

urlpatterns = [
    path("classify/", views.classify_audio, name="classify"),
    path("health/",   views.health_check,   name="health"),
    path("dialects/", views.list_dialects,  name="dialects"),
    path("stats/",    views.dataset_stats,  name="stats"),
]
