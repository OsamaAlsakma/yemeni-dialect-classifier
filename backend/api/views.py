import os
import tempfile
import uuid
from pathlib import Path

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response

from ml_inference.predictor import get_predictor, DIALECTS, DIALECT_RU, DIALECT_REGION, NUM_CLASSES, SAMPLE_RATE

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".webm", ".ogg", ".m4a"}
MAX_FILE_SIZE_MB = 25

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def classify_audio(request):
    if "audio" not in request.FILES:
        return Response({"success": False, "error": "No audio file provided."}, status=status.HTTP_400_BAD_REQUEST)
    audio_file = request.FILES["audio"]
    ext = Path(audio_file.name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return Response({"success": False, "error": f"Unsupported format: {ext}"}, status=status.HTTP_400_BAD_REQUEST)
    if audio_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return Response({"success": False, "error": f"File too large. Max: {MAX_FILE_SIZE_MB} MB."}, status=status.HTTP_400_BAD_REQUEST)
    try:
        top_n = int(request.data.get("top_n", 3))
        top_n = max(1, min(top_n, NUM_CLASSES))
    except (ValueError, TypeError):
        top_n = 3
    temp_dir = Path(tempfile.gettempdir()) / "ydc_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4().hex}{ext}"
    try:
        with open(temp_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
        predictor = get_predictor(settings.ML_DEFAULT_MODEL)
        result = predictor.predict(str(temp_path), top_n=top_n)
        return Response({"success": True, "result": result, "filename": audio_file.name, "file_size_kb": round(audio_file.size/1024, 2)}, status=status.HTTP_200_OK)
    except FileNotFoundError as e:
        return Response({"success": False, "error": "Model not found.", "details": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    except Exception as e:
        return Response({"success": False, "error": "Audio processing error", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError:
                pass

@api_view(["GET"])
def health_check(request):
    try:
        predictor = get_predictor(settings.ML_DEFAULT_MODEL)
        model_loaded = True
        model_name = predictor.model_name
    except Exception:
        model_loaded = False
        model_name = None
    return Response({"status": "ok" if model_loaded else "degraded", "service": "Yemeni Dialect Classifier API", "version": "1.0", "model_loaded": model_loaded, "current_model": model_name, "supported_classes": NUM_CLASSES}, status=status.HTTP_200_OK)

@api_view(["GET"])
def list_dialects(request):
    dialects_list = [{"index": i, "name_ar": name, "name_ru": DIALECT_RU[name], "region_ru": DIALECT_REGION[name]["ru"]} for i, name in enumerate(DIALECTS)]
    return Response({"count": NUM_CLASSES, "dialects": dialects_list}, status=status.HTTP_200_OK)

@api_view(["GET"])
def dataset_stats(request):
    return Response({"num_classes": NUM_CLASSES, "sample_rate": SAMPLE_RATE}, status=status.HTTP_200_OK)
