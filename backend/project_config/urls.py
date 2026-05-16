"""
urls.py — Корневая конфигурация URL-маршрутов Django.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.conf.urls.i18n import i18n_patterns


# Маршруты без префикса языка
urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include("api.urls")),       # REST API
    path("i18n/", include("django.conf.urls.i18n")),  # переключение языка
]

# Маршруты с префиксом языка (/ru/, /ar/)
urlpatterns += i18n_patterns(
    path("", include("classifier.urls")),
    prefix_default_language=True,
)

# Статика и медиа в режиме разработки
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
