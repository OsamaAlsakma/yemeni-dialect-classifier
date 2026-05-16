"""
settings.py — Конфигурация Django-проекта.

Yemeni Dialect Classifier — веб-приложение для автоматической
классификации йеменских диалектов.
"""
import sys
from pathlib import Path

# =============================================================
# ПУТИ ПРОЕКТА
# =============================================================
BASE_DIR     = Path(__file__).resolve().parent.parent     # backend/
PROJECT_ROOT = BASE_DIR.parent                            # корень проекта

# Добавляем ML-модуль в PYTHONPATH, чтобы импортировать predictor
ML_SRC = PROJECT_ROOT / "ml" / "src"
sys.path.insert(0, str(ML_SRC))

# =============================================================
# БАЗОВЫЕ НАСТРОЙКИ
# =============================================================
SECRET_KEY = "django-insecure-CHANGE-ME-IN-PRODUCTION-2026"
DEBUG      = True
ALLOWED_HOSTS = ["*"]                # для разработки

# =============================================================
# ПРИЛОЖЕНИЯ
# =============================================================
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # Сторонние библиотеки
    "rest_framework",                 # REST API
    "corsheaders",                    # CORS для AJAX-запросов

    # Локальные приложения
    "classifier",                     # модуль классификации
    "api",                            # модуль REST API
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.locale.LocaleMiddleware",         # для i18n
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "project_config.urls"

# =============================================================
# ШАБЛОНЫ
# =============================================================
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [PROJECT_ROOT / "frontend" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.template.context_processors.i18n",   # для перевода
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "project_config.wsgi.application"

# =============================================================
# БАЗА ДАННЫХ
# =============================================================
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# =============================================================
# ВАЛИДАЦИЯ ПАРОЛЕЙ
# =============================================================
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# =============================================================
# ИНТЕРНАЦИОНАЛИЗАЦИЯ — РУССКИЙ И АРАБСКИЙ
# =============================================================
LANGUAGE_CODE = "ru"                  # язык по умолчанию

LANGUAGES = [
    ("ru", "Русский"),
    ("ar", "العربية"),
]

LOCALE_PATHS = [
    PROJECT_ROOT / "frontend" / "locale",
]

TIME_ZONE = "Europe/Moscow"
USE_I18N  = True
USE_TZ    = True

# =============================================================
# СТАТИКА И МЕДИА
# =============================================================
STATIC_URL = "/static/"
STATICFILES_DIRS = [
    PROJECT_ROOT / "frontend" / "static",
]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL  = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# Максимальный размер загружаемого файла: 25 МБ
DATA_UPLOAD_MAX_MEMORY_SIZE = 25 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 25 * 1024 * 1024

# =============================================================
# CORS — РАЗРЕШЕНИЕ КРОСС-ДОМЕННЫХ ЗАПРОСОВ
# =============================================================
CORS_ALLOW_ALL_ORIGINS = DEBUG        # в production указать конкретные домены

# =============================================================
# DJANGO REST FRAMEWORK
# =============================================================
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.MultiPartParser",
        "rest_framework.parsers.FormParser",
    ],
}

# =============================================================
# ML-НАСТРОЙКИ
# =============================================================
ML_MODELS_DIR = PROJECT_ROOT / "ml" / "saved_models"
ML_DEFAULT_MODEL = "cnn_lstm"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
