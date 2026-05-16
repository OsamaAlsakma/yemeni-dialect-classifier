"""
views.py — Представления REST API для классификации диалектов.

Содержит endpoints:
    POST /api/v1/classify/   — классификация загруженного аудио
    GET  /api/v1/health/     — проверка работоспособности API
    GET  /api/v1/dialects/   — список поддерживаемых диалектов
"""
import os
import tempfile
import uuid
from pathlib import Path

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response

# Импорт ML-модуля (путь добавлен в settings.py)
from predict import get_predictor
from config import (
    DIALECTS, DIALECT_RU, DIALECT_REGION_RU,
    NUM_CLASSES, SAMPLE_RATE,
    TOTAL_RECORDINGS, TRAIN_RECORDINGS, VAL_RECORDINGS, TEST_RECORDINGS,
)


# Допустимые расширения аудиофайлов
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".webm", ".ogg", ".m4a"}
MAX_FILE_SIZE_MB   = 25


# =============================================================
# ENDPOINT: КЛАССИФИКАЦИЯ ЗАГРУЖЕННОГО АУДИО
# =============================================================
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def classify_audio(request):
    """
    POST /api/v1/classify/

    Принимает аудиофайл и возвращает результат классификации диалекта.

    Параметры запроса (multipart/form-data):
        audio   : file       — аудиофайл (WAV/MP3/WEBM/OGG/FLAC/M4A)
        top_n   : int (опц.) — число лучших классов (по умолчанию 3)

    Возвращает JSON:
        {
          "success": true,
          "result": {
            "top_dialect_ar":    "...",
            "top_dialect_ru":    "...",
            "top_region_ru":     "...",
            "top_confidence":    0.87,
            "top_confidence_pct": 87.32,
            "top_n":             [...],
            "all_probabilities": {...},
            "model_used":        "cnn_lstm"
          }
        }
    """
    # === Шаг 1. Валидация наличия файла ===
    if "audio" not in request.FILES:
        return Response(
            {
                "success": False,
                "error": "Аудиофайл не предоставлен. "
                         "Используйте поле 'audio' в multipart-форме.",
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    audio_file = request.FILES["audio"]

    # === Шаг 2. Проверка расширения файла ===
    ext = Path(audio_file.name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return Response(
            {
                "success": False,
                "error": f"Неподдерживаемый формат: {ext}. "
                         f"Допустимые форматы: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    # === Шаг 3. Проверка размера файла ===
    if audio_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return Response(
            {
                "success": False,
                "error": f"Файл слишком большой. "
                         f"Максимум: {MAX_FILE_SIZE_MB} МБ.",
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    # === Шаг 4. Извлечение параметра top_n ===
    try:
        top_n = int(request.data.get("top_n", 3))
        top_n = max(1, min(top_n, NUM_CLASSES))
    except (ValueError, TypeError):
        top_n = 3

    # === Шаг 5. Сохранение временного файла на диск ===
    # Librosa требует путь к файлу, поэтому сохраняем во временный каталог
    temp_dir = Path(tempfile.gettempdir()) / "ydc_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = temp_dir / temp_filename

    try:
        # Запись содержимого загруженного файла
        with open(temp_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        # === Шаг 6. Инференс модели ===
        predictor = get_predictor(settings.ML_DEFAULT_MODEL)
        result = predictor.predict(str(temp_path), top_n=top_n)

        return Response(
            {
                "success": True,
                "result":  result,
                "filename": audio_file.name,
                "file_size_kb": round(audio_file.size / 1024, 2),
            },
            status=status.HTTP_200_OK,
        )

    except FileNotFoundError as e:
        # Модель не загружена
        return Response(
            {
                "success": False,
                "error": "Модель не найдена. Сначала обучите модель: "
                         "python ml/src/train.py --model cnn_lstm",
                "details": str(e),
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    except Exception as e:
        # Любая другая ошибка обработки
        return Response(
            {
                "success": False,
                "error": "Ошибка обработки аудио",
                "details": str(e),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    finally:
        # === Шаг 7. Очистка временного файла ===
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError:
                pass


# =============================================================
# ENDPOINT: ПРОВЕРКА РАБОТОСПОСОБНОСТИ
# =============================================================
@api_view(["GET"])
def health_check(request):
    """
    GET /api/v1/health/

    Возвращает статус сервиса.
    """
    # Проверяем, можем ли загрузить модель
    try:
        predictor = get_predictor(settings.ML_DEFAULT_MODEL)
        model_loaded = True
        model_name = predictor.model_name
    except FileNotFoundError:
        model_loaded = False
        model_name = None
    except Exception:
        model_loaded = False
        model_name = None

    return Response(
        {
            "status": "ok" if model_loaded else "degraded",
            "service": "Yemeni Dialect Classifier API",
            "version": "1.0",
            "model_loaded": model_loaded,
            "current_model": model_name,
            "supported_classes": NUM_CLASSES,
        },
        status=status.HTTP_200_OK,
    )


# =============================================================
# ENDPOINT: СПИСОК ПОДДЕРЖИВАЕМЫХ ДИАЛЕКТОВ
# =============================================================
@api_view(["GET"])
def list_dialects(request):
    """
    GET /api/v1/dialects/

    Возвращает список всех поддерживаемых диалектов с описанием.
    """
    dialects_list = [
        {
            "index":      i,
            "name_ar":    name,
            "name_ru":    DIALECT_RU[name],
            "region_ru":  DIALECT_REGION_RU[name],
        }
        for i, name in enumerate(DIALECTS)
    ]
    return Response(
        {
            "count":    NUM_CLASSES,
            "dialects": dialects_list,
        },
        status=status.HTTP_200_OK,
    )


# =============================================================
# ENDPOINT: СТАТИСТИКА ДАТАСЕТА
# =============================================================
@api_view(["GET"])
def dataset_stats(request):
    """
    GET /api/v1/stats/

    Возвращает статистику обучающего датасета.
    """
    return Response(
        {
            "total_recordings": TOTAL_RECORDINGS,
            "train":            TRAIN_RECORDINGS,
            "validate":         VAL_RECORDINGS,
            "test":             TEST_RECORDINGS,
            "num_classes":      NUM_CLASSES,
            "sample_rate":      SAMPLE_RATE,
            "split_ratio":      "60 / 20 / 20",
        },
        status=status.HTTP_200_OK,
    )
