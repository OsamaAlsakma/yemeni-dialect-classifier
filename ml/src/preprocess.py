"""
preprocess.py — Предобработка аудио и извлечение акустических признаков.

Модуль реализует:
  • загрузку аудиофайлов в стандартизированном формате (16 kHz, mono);
  • извлечение мел-спектрограмм (для свёрточных моделей);
  • извлечение MFCC + Δ + ΔΔ (для рекуррентных моделей);
  • сборку всех записей выборки в единый NPZ-файл с кэшированием.

Запуск:
  python preprocess.py
"""
import json
import sys
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

from config import (
    SAMPLE_RATE, N_SAMPLES, N_FFT, HOP_LENGTH, WIN_LENGTH,
    N_MELS, N_MFCC, FMIN, FMAX,
    DIALECT2IDX, DATA_DIR, FEATURES_DIR,
    META_TRAIN, META_VAL, META_TEST,
)


# =============================================================
# ЗАГРУЗКА И НОРМАЛИЗАЦИЯ АУДИОСИГНАЛА
# =============================================================
def load_audio(path):
    """
    Загружает аудиофайл и приводит его к стандартизированному виду.

    Этапы:
        1. Ресэмплирование к 16 кГц и преобразование к моно.
        2. Удаление тишины в начале и в конце (silence trimming).
        3. Нормализация амплитуды по пиковому значению.
        4. Приведение к фиксированной длине 8 секунд:
           - короткие записи дополняются нулями (zero-padding);
           - длинные записи обрезаются.

    Параметры:
        path : str | Path — путь к аудиофайлу (WAV, MP3, FLAC и др.)

    Возвращает:
        np.ndarray формы (N_SAMPLES,) типа float32.
    """
    # Загрузка с автоматическим ресэмплированием и сведением каналов
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    # Удаление пауз с порогом -60 дБ относительно пика
    y, _ = librosa.effects.trim(y, top_db=60)

    # Нормализация амплитуды в диапазон [-1, +1]
    y = librosa.util.normalize(y)

    # Приведение к фиксированной длине
    if len(y) < N_SAMPLES:
        # Дополнение нулями справа (zero-padding)
        y = np.pad(y, (0, N_SAMPLES - len(y)), mode="constant")
    else:
        # Обрезка лишних отсчётов
        y = y[:N_SAMPLES]

    return y.astype(np.float32)


# =============================================================
# ИЗВЛЕЧЕНИЕ МЕЛ-СПЕКТРОГРАММЫ
# =============================================================
def extract_mel(y):
    """
    Извлекает логарифмическую мел-спектрограмму из аудиосигнала.

    Используется как 2D-изображение для подачи в свёрточные сети
    (CNN и гибридная CNN+LSTM).

    Параметры:
        y : np.ndarray — аудиосигнал длины N_SAMPLES.

    Возвращает:
        np.ndarray формы (N_MELS, T) типа float32.
        Значения выражены в децибелах относительно пиковой мощности.
    """
    # Кратковременное преобразование Фурье + мел-фильтрбанк
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )

    # Логарифмическая шкала (дБ) для нормализации динамического диапазона
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db.astype(np.float32)


# =============================================================
# ИЗВЛЕЧЕНИЕ MFCC С ПРОИЗВОДНЫМИ
# =============================================================
def extract_mfcc(y):
    """
    Извлекает MFCC с производными первого и второго порядков.

    Используется как 1D-последовательность для рекуррентной сети LSTM.

    Параметры:
        y : np.ndarray — аудиосигнал длины N_SAMPLES.

    Возвращает:
        np.ndarray формы (T, 120) типа float32.
        120 = 40 MFCC + 40 Δ + 40 ΔΔ.
    """
    # Основные MFCC-коэффициенты
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    )

    # Производные: дельта (скорость изменения) и дельта-дельта (ускорение)
    delta  = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Конкатенация: (40, T) + (40, T) + (40, T) = (120, T)
    stacked = np.concatenate([mfcc, delta, delta2], axis=0)

    # Транспонирование к виду (T, 120) — последовательность векторов
    return stacked.T.astype(np.float32)


# =============================================================
# ЗАГРУЗКА МЕТАДАННЫХ ВЫБОРКИ
# =============================================================
def load_metadata(meta_path):
    """
    Читает JSON-файл со списком метаданных аудиозаписей.

    Параметры:
        meta_path : Path — путь к metadata.json.

    Возвращает:
        list[dict] — список словарей с метаданными записей.
    """
    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


# =============================================================
# СБОРКА ПРИЗНАКОВ ВЫБОРКИ
# =============================================================
def build_dataset(split_name, meta_path, save_npz=True):
    """
    Собирает все записи указанной выборки и извлекает признаки.

    Для каждой аудиозаписи извлекаются одновременно мел-спектрограмма
    и MFCC. Результат сохраняется в файле features_<split>.npz для
    быстрого повторного использования при обучении.

    Параметры:
        split_name : str        — "train" / "validate" / "test"
        meta_path  : Path       — путь к metadata.json данной выборки
        save_npz   : bool       — флаг сохранения результата на диск

    Возвращает:
        tuple (X_mel, X_mfcc, y_lab):
            X_mel  : np.ndarray (N, N_MELS, T)
            X_mfcc : np.ndarray (N, T, 120)
            y_lab  : np.ndarray (N,) — индексы классов
    """
    print(f"\n{'='*60}")
    print(f"Обработка выборки: {split_name.upper()}")
    print(f"{'='*60}")

    # Загрузка метаданных всей выборки
    records = load_metadata(meta_path)
    print(f"Найдено записей: {len(records)}")

    X_mel, X_mfcc, y_lab = [], [], []
    skipped = 0

    # Поэлементная обработка с индикатором прогресса
    for rec in tqdm(records, desc=f"[{split_name}]"):
        # Полный путь к аудиофайлу
        audio_path = DATA_DIR / rec["audio_path"]

        if not audio_path.exists():
            skipped += 1
            continue

        # Проверка наличия диалекта в словаре классов
        gov = rec["governorate"]
        if gov not in DIALECT2IDX:
            skipped += 1
            continue

        # Загрузка и обработка аудиосигнала
        try:
            y_audio = load_audio(audio_path)
        except Exception as e:
            print(f"  ! Ошибка загрузки {audio_path.name}: {e}")
            skipped += 1
            continue

        # Извлечение обоих типов признаков
        X_mel.append(extract_mel(y_audio))
        X_mfcc.append(extract_mfcc(y_audio))
        y_lab.append(DIALECT2IDX[gov])

    # Преобразование в numpy-массивы
    X_mel  = np.stack(X_mel)
    X_mfcc = np.stack(X_mfcc)
    y_lab  = np.array(y_lab, dtype=np.int32)

    print(f"Успешно обработано: {len(y_lab)}")
    print(f"Пропущено:          {skipped}")
    print(f"Размер X_mel:       {X_mel.shape}")
    print(f"Размер X_mfcc:      {X_mfcc.shape}")
    print(f"Размер меток:       {y_lab.shape}")

    # Сохранение результата на диск
    if save_npz:
        out_path = FEATURES_DIR / f"features_{split_name}.npz"
        np.savez_compressed(
            out_path,
            X_mel=X_mel,
            X_mfcc=X_mfcc,
            y=y_lab,
        )
        print(f"✓ Сохранено: {out_path}")

    return X_mel, X_mfcc, y_lab


# =============================================================
# ТОЧКА ВХОДА — ОБРАБОТКА ВСЕХ ТРЁХ ВЫБОРОК
# =============================================================
if __name__ == "__main__":
    splits = [
        ("train",    META_TRAIN),
        ("validate", META_VAL),
        ("test",     META_TEST),
    ]

    for split_name, meta_path in splits:
        if not meta_path.exists():
            print(f"⚠ Файл не найден: {meta_path}")
            print(f"  Убедитесь, что датасет распакован в {DATA_DIR}")
            sys.exit(1)
        build_dataset(split_name, meta_path)

    print("\n✓ Предобработка всех выборок завершена.")
