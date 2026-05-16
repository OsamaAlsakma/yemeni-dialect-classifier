"""
predict.py — Модуль инференса для использования обученной модели.

Загружает сохранённые веса лучшей модели и предсказывает диалект
для произвольного аудиофайла. Используется веб-приложением Django.

Класс DialectPredictor реализован как singleton, что обеспечивает
загрузку модели в память только один раз.
"""
import numpy as np
import tensorflow as tf
from pathlib import Path

from config import (
    MODELS_DIR, BEST_MODEL,
    DIALECTS, IDX2DIALECT, DIALECT_RU, DIALECT_REGION_RU,
    NUM_CLASSES, N_MELS,
)
from preprocess import load_audio, extract_mel


class DialectPredictor:
    """
    Класс для предсказания диалекта по аудиофайлу.

    Атрибуты:
        model      : tf.keras.Model — загруженная обученная модель
        model_name : str            — имя текущей модели
    """

    def __init__(self, model_name=BEST_MODEL):
        """
        Загружает указанную модель из каталога saved_models.

        Параметры:
            model_name : str — имя модели: "cnn", "lstm" или "cnn_lstm"
        """
        self.model_name = model_name
        model_path = MODELS_DIR / f"{model_name}_best.h5"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Модель {model_path} не найдена. "
                f"Сначала запустите python train.py --model {model_name}"
            )

        print(f"Загрузка модели: {model_path}")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✓ Модель {model_name.upper()} загружена")

    def predict(self, audio_path, top_n=3):
        """
        Выполняет классификацию аудиофайла.

        Параметры:
            audio_path : str | Path — путь к аудиофайлу
            top_n      : int        — сколько лучших классов возвращать

        Возвращает:
            dict со следующими полями:
                top_dialect_ar    : str  — арабское название лучшего диалекта
                top_dialect_ru    : str  — русское название
                top_region_ru     : str  — описание региона
                top_confidence    : float — уверенность лучшего класса (0..1)
                top_n             : list[dict] — список Top-N предсказаний
                all_probabilities : dict — вероятности всех 11 классов
        """
        # Полный конвейер: загрузка → предобработка → инференс
        y_audio = load_audio(audio_path)
        mel = extract_mel(y_audio)

        # Добавление batch и channel размерностей для модели
        # Для CNN и CNN+LSTM: (1, N_MELS, T, 1)
        if self.model_name in ("cnn", "cnn_lstm"):
            x_input = mel[np.newaxis, ..., np.newaxis]
        else:
            # Для LSTM нужны MFCC, но в API используем только CNN+LSTM
            from preprocess import extract_mfcc
            mfcc = extract_mfcc(y_audio)
            x_input = mfcc[np.newaxis, ...]

        # Получение вероятностей
        probs = self.model.predict(x_input, verbose=0)[0]   # (11,)

        # Топ-N классов по убыванию вероятности
        top_indices = probs.argsort()[::-1][:top_n]

        top_n_list = []
        for idx in top_indices:
            dialect_ar = IDX2DIALECT[int(idx)]
            top_n_list.append({
                "index":        int(idx),
                "dialect_ar":   dialect_ar,
                "dialect_ru":   DIALECT_RU[dialect_ar],
                "region_ru":    DIALECT_REGION_RU[dialect_ar],
                "probability":  float(probs[idx]),
                "confidence_pct": round(float(probs[idx]) * 100, 2),
            })

        # Полное распределение по всем классам
        all_probs = {}
        for idx in range(NUM_CLASSES):
            dialect_ar = IDX2DIALECT[idx]
            all_probs[dialect_ar] = {
                "dialect_ru":  DIALECT_RU[dialect_ar],
                "probability": float(probs[idx]),
            }

        # Лучший класс
        best = top_n_list[0]

        return {
            "model_used":        self.model_name,
            "top_dialect_ar":    best["dialect_ar"],
            "top_dialect_ru":    best["dialect_ru"],
            "top_region_ru":     best["region_ru"],
            "top_confidence":    best["probability"],
            "top_confidence_pct": best["confidence_pct"],
            "top_n":             top_n_list,
            "all_probabilities": all_probs,
        }


# Глобальный экземпляр (singleton), создаётся при первом импорте
_predictor_instance = None


def get_predictor(model_name=BEST_MODEL):
    """
    Возвращает singleton-экземпляр DialectPredictor.

    Это позволяет загрузить модель в память только один раз и
    переиспользовать её для всех запросов веб-приложения.

    Параметры:
        model_name : str — имя модели для загрузки

    Возвращает:
        DialectPredictor — экземпляр класса.
    """
    global _predictor_instance
    if _predictor_instance is None or _predictor_instance.model_name != model_name:
        _predictor_instance = DialectPredictor(model_name)
    return _predictor_instance


# Тестовый запуск из командной строки
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Использование: python predict.py <path_to_audio.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]
    predictor = get_predictor()
    result = predictor.predict(audio_path)

    print(f"\n{'='*60}")
    print(f"Результат классификации")
    print(f"{'='*60}")
    print(f"Лучший диалект (ar): {result['top_dialect_ar']}")
    print(f"Лучший диалект (ru): {result['top_dialect_ru']}")
    print(f"Регион:              {result['top_region_ru']}")
    print(f"Уверенность:         {result['top_confidence_pct']}%")
    print(f"\nTop-3:")
    for i, entry in enumerate(result["top_n"], 1):
        print(f"  {i}. {entry['dialect_ru']:25s} {entry['confidence_pct']:6.2f}%")
