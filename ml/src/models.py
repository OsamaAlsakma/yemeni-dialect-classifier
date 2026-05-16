"""
models.py — Три архитектуры нейронных сетей для классификации диалектов.

Реализованы средствами Keras Functional API:
  • build_cnn       — свёрточная сеть (CNN) для мел-спектрограмм;
  • build_lstm      — рекуррентная сеть (BiLSTM + Attention) для MFCC;
  • build_cnn_lstm  — гибридная архитектура CNN + LSTM (рекомендуемая).

Все модели принимают входы переменной длины (None по временной оси)
и выдают softmax-распределение по 11 классам диалектов.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

from config import NUM_CLASSES, N_MELS


# =============================================================
# 1. АРХИТЕКТУРА CNN — для обработки мел-спектрограмм
# =============================================================
def build_cnn(input_shape=(N_MELS, None, 1), num_classes=NUM_CLASSES):
    """
    Свёрточная нейронная сеть.

    Обрабатывает мел-спектрограмму как 2D-изображение через
    четыре свёрточных блока возрастающей глубины. Каждый блок:
        Conv2D → BatchNorm → ReLU → MaxPool

    Параметры:
        input_shape : tuple — форма входа (mel-bins, T, channels)
        num_classes : int   — число классов на выходе

    Возвращает:
        tf.keras.Model — скомпилированная модель с именем "CNN"
    """
    inp = Input(shape=input_shape, name="mel_input")
    x = inp

    # Свёрточные блоки: число фильтров удваивается в каждом
    for n_filters in (32, 64, 128, 256):
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        # MaxPool применяется во всех блоках кроме последнего
        if n_filters != 256:
            x = layers.MaxPool2D(pool_size=(2, 2))(x)

    # Глобальное усреднение → переход к векторному представлению
    x = layers.GlobalAveragePooling2D()(x)

    # Классификатор: полносвязный слой + регуляризация Dropout
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # Выходной слой с активацией softmax
    out = layers.Dense(num_classes, activation="softmax", name="dialect_probs")(x)

    return Model(inputs=inp, outputs=out, name="CNN")


# =============================================================
# 2. АРХИТЕКТУРА LSTM — для обработки MFCC-последовательностей
# =============================================================
def build_lstm(input_shape=(None, 120), num_classes=NUM_CLASSES):
    """
    Двунаправленная рекуррентная сеть с механизмом внимания.

    Обрабатывает MFCC-последовательность как временной ряд через
    два слоя BiLSTM. Затем применяется attention pooling: каждый
    временной шаг получает обучаемый вес важности.

    Параметры:
        input_shape : tuple — форма входа (T, 120)
        num_classes : int   — число классов

    Возвращает:
        tf.keras.Model — модель с именем "LSTM"
    """
    inp = Input(shape=input_shape, name="mfcc_input")

    # Первый BiLSTM-слой: учитывает прошлый и будущий контекст
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True), name="bilstm_1"
    )(inp)
    x = layers.Dropout(0.4)(x)

    # Второй BiLSTM-слой
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True), name="bilstm_2"
    )(x)
    x = layers.Dropout(0.4)(x)

    # Attention Pooling: обучаемые веса важности для каждого шага
    attn_weights = layers.Dense(1, activation="tanh", name="attn_score")(x)
    attn_weights = layers.Softmax(axis=1, name="attn_softmax")(attn_weights)
    x = layers.Multiply()([x, attn_weights])
    x = layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1), name="weighted_sum"
    )(x)

    # Классификатор
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax", name="dialect_probs")(x)

    return Model(inputs=inp, outputs=out, name="LSTM")


# =============================================================
# 3. ГИБРИДНАЯ АРХИТЕКТУРА CNN + LSTM — основная модель
# =============================================================
def build_cnn_lstm(input_shape=(N_MELS, None, 1), num_classes=NUM_CLASSES):
    """
    Гибридная архитектура: CNN-экстрактор + LSTM-кодировщик.

    Свёрточные слои сжимают только частотную ось мел-спектрограммы,
    сохраняя временную для последующего рекуррентного анализа.
    Это позволяет одновременно использовать:
      • локальные частотно-временные паттерны (через CNN);
      • долговременные временные зависимости (через BiLSTM).

    Параметры:
        input_shape : tuple — форма входа (mel-bins, T, channels)
        num_classes : int   — число классов

    Возвращает:
        tf.keras.Model — модель с именем "CNN_LSTM"
    """
    inp = Input(shape=input_shape, name="mel_input")
    x = inp

    # === CNN-экстрактор (3 блока) ===
    # MaxPool (2,1) сжимает только частоту, временная ось сохраняется
    for n_filters in (32, 64, 128):
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D(pool_size=(2, 1))(x)

    # === Переход CNN → LSTM ===
    # Преобразование (B, F', T, C) → (B, T, F'×C) — последовательность во времени
    cnn_shape = tf.keras.backend.int_shape(x)
    x = layers.Permute((2, 1, 3))(x)              # (B, T, F', C)
    x = layers.Reshape(
        (-1, cnn_shape[1] * cnn_shape[3])
    )(x)                                          # (B, T, F'·C)

    # === LSTM-кодировщик (2 слоя BiLSTM) ===
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True), name="bilstm_1"
    )(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True), name="bilstm_2"
    )(x)
    x = layers.Dropout(0.4)(x)

    # === Attention Pooling ===
    attn_weights = layers.Dense(1, activation="tanh", name="attn_score")(x)
    attn_weights = layers.Softmax(axis=1, name="attn_softmax")(attn_weights)
    x = layers.Multiply()([x, attn_weights])
    x = layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1), name="weighted_sum"
    )(x)

    # === Классификатор ===
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax", name="dialect_probs")(x)

    return Model(inputs=inp, outputs=out, name="CNN_LSTM")


# =============================================================
# СЛОВАРЬ ФАБРИК — для удобного выбора модели по имени
# =============================================================
MODEL_FACTORY = {
    "cnn":      build_cnn,
    "lstm":     build_lstm,
    "cnn_lstm": build_cnn_lstm,
}


# =============================================================
# ТОЧКА ВХОДА — вывод сводки по всем моделям
# =============================================================
if __name__ == "__main__":
    for model_name, factory_fn in MODEL_FACTORY.items():
        print(f"\n{'='*65}")
        print(f"Архитектура: {model_name.upper()}")
        print(f"{'='*65}")
        model = factory_fn()
        model.summary()
