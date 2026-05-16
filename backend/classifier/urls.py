"""
urls.py — URL-маршруты приложения classifier.
"""
from django.urls import path
from . import views

app_name = "classifier"

urlpatterns = [
    path("",       views.home,  name="home"),
    path("about/", views.about, name="about"),
]
