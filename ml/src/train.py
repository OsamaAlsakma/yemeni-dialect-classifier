"""
train.py — Обучение моделей и оценка качества на тестовой выборке.

Модуль реализует полный цикл обучения:
    1. Загрузка предвычисленных признаков.
    2. Построение и компиляция модели.
    3. Обучение с применением callback-функций.
    4. Оценка на тестовой выборке.
    5. Сохранение результатов: веса, журнал, метрики, графики.

Использование:
    python train.py --model cnn          # обучить только CNN
    python train.py --model lstm         # обучить только LSTM
    python train.py --model cnn_lstm     # обучить только CNN+LSTM
    python train.py --model all          # обучить все три модели
"""
import argparse
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from config import (
    NUM_CLASSES, BATCH_SIZE, EPOCHS, LR,
    PATIENCE_ES, PATIENCE_LR, LR_FACTOR, RANDOM_SEED,
    MODELS_DIR, LOGS_DIR, RESULTS_DIR, FEATURES_DIR,
    DIALECTS, DIALECT_RU, MODEL_NAMES,
)
from models import MODEL_FACTORY


# =============================================================
# ЗАГРУЗКА ПРИЗНАКОВ ИЗ NPZ-ФАЙЛОВ
# =============================================================
def load_features(split_name, model_name):
    """
    Загружает предвычисленные признаки в зависимости от модели.

    Для CNN и CNN+LSTM используется мел-спектрограмма (4D-тензор).
    Для LSTM используется MFCC (3D-тензор последовательностей).

    Параметры:
        split_name : str — "train" / "validate" / "test"
        model_name : str — "cnn" / "lstm" / "cnn_lstm"

    Возвращает:
        tuple (X, y_onehot):
            X        : np.ndarray признаков нужного типа
            y_onehot : np.ndarray (N, NUM_CLASSES) меток в one-hot формате
    """
    npz_path = FEATURES_DIR / f"features_{split_name}.npz"
    data = np.load(npz_path)

    # Преобразование меток в one-hot кодирование
    y_onehot = tf.keras.utils.to_categorical(data["y"], NUM_CLASSES)

    # Выбор типа признаков по архитектуре модели
    if model_name == "lstm":
        return data["X_mfcc"], y_onehot
    # Для CNN и CNN+LSTM добавляем канальную размерность
    X = data["X_mel"][..., np.newaxis]   # (N, N_MELS, T, 1)
    return X, y_onehot


# =============================================================
# АУГМЕНТАЦИЯ ДАННЫХ — SpecAugment
# =============================================================
def add_specaugment(spec, freq_mask=20, time_mask=30, n_masks=2):
    """
    Применяет аугментацию SpecAugment к мел-спектрограмме.

    Случайно обнуляет прямоугольные полосы по частоте и по времени,
    что повышает устойчивость модели к локальным искажениям сигнала.

    Параметры:
        spec      : np.ndarray — мел-спектрограмма
        freq_mask : int        — максимальная ширина частотной маски
        time_mask : int        — максимальная ширина временной маски
        n_masks   : int        — число пар масок

    Возвращает:
        np.ndarray — модифицированная мел-спектрограмма.
    """
    spec = spec.copy()
    for _ in range(n_masks):
        # Частотная маска
        f  = np.random.randint(0, freq_mask)
        f0 = np.random.randint(0, max(1, spec.shape[0] - f))
        spec[f0:f0 + f, :] = 0
        # Временная маска
        t  = np.random.randint(0, time_mask)
        t0 = np.random.randint(0, max(1, spec.shape[1] - t))
        spec[:, t0:t0 + t] = 0
    return spec


# =============================================================
# ПОСТРОЕНИЕ И СОХРАНЕНИЕ ГРАФИКОВ
# =============================================================
def plot_training_curves(history, model_name):
    """
    Строит и сохраняет графики кривых обучения.

    Создаёт два графика:
      • функция потерь (train + validation);
      • точность (train + validation).

    Параметры:
        history    : dict — словарь history.history из model.fit()
        model_name : str  — имя модели для подписи и имени файла
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # График потерь
    axes[0].plot(history["loss"],     "k-",  label="train", linewidth=2)
    axes[0].plot(history["val_loss"], "k--", label="validation", linewidth=2)
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Функция потерь ({model_name.upper()})")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # График точности
    axes[1].plot(history["accuracy"],     "k-",  label="train", linewidth=2)
    axes[1].plot(history["val_accuracy"], "k--", label="validation", linewidth=2)
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Точность классификации ({model_name.upper()})")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    out_path = RESULTS_DIR / f"{model_name}_training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ График сохранён: {out_path}")


def plot_confusion_matrix(cm, model_name):
    """
    Строит и сохраняет матрицу ошибок (confusion matrix).

    Параметры:
        cm         : np.ndarray (11, 11) — матрица ошибок
        model_name : str                 — имя модели
    """
    # Названия диалектов на русском для подписей осей
    labels_ru = [DIALECT_RU[d] for d in DIALECTS]

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greys",                # чёрно-белая палитра для ГОСТ
        xticklabels=labels_ru,
        yticklabels=labels_ru,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(f"Матрица ошибок ({model_name.upper()})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    out_path = RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ График сохранён: {out_path}")


# =============================================================
# ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ ОДНОЙ МОДЕЛИ
# =============================================================
def train_one(model_name):
    """
    Полный цикл обучения одной модели: построение → обучение → оценка.

    Параметры:
        model_name : str — имя модели из MODEL_FACTORY

    Возвращает:
        float — точность на тестовой выборке.
    """
    print(f"\n{'='*65}")
    print(f"ОБУЧЕНИЕ МОДЕЛИ: {model_name.upper()}")
    print(f"{'='*65}")

    # Фиксация случайности для воспроизводимости экспериментов
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Загрузка трёх выборок
    print("Загрузка признаков...")
    X_tr, y_tr = load_features("train",    model_name)
    X_va, y_va = load_features("validate", model_name)
    X_te, y_te = load_features("test",     model_name)
    print(f"  train:      {X_tr.shape}")
    print(f"  validation: {X_va.shape}")
    print(f"  test:       {X_te.shape}")

    # Построение модели
    model = MODEL_FACTORY[model_name]()
    print(f"\nЧисло параметров: {model.count_params():,}")

    # Компиляция: оптимизатор Adam, кросс-энтропия + Top-3 accuracy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )

    # Настройка callback-функций
    callbacks = [
        # Ранняя остановка по валидационной потере
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_ES,
            restore_best_weights=True,
            verbose=1,
        ),
        # Адаптивное снижение скорости обучения
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=PATIENCE_LR,
            min_lr=1e-6,
            verbose=1,
        ),
        # Сохранение лучшей версии модели
        ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{model_name}_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Журналирование процесса обучения в CSV
        CSVLogger(filename=str(LOGS_DIR / f"{model_name}_log.csv")),
    ]

    # Цикл обучения
    print("\nЗапуск обучения...")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # === Оценка качества на тестовой выборке ===
    print("\nОценка на тестовой выборке...")
    y_pred_probs = model.predict(X_te, verbose=0)
    y_pred = y_pred_probs.argmax(axis=1)
    y_true = y_te.argmax(axis=1)

    test_acc = accuracy_score(y_true, y_pred)
    cm       = confusion_matrix(y_true, y_pred)
    report   = classification_report(
        y_true, y_pred,
        target_names=[DIALECT_RU[d] for d in DIALECTS],
        output_dict=True,
        zero_division=0,
    )
    print(f"\n>>> Test Accuracy: {test_acc:.4f}")

    # Top-3 accuracy
    top3_correct = sum(
        y_true[i] in y_pred_probs[i].argsort()[-3:]
        for i in range(len(y_true))
    )
    top3_acc = top3_correct / len(y_true)
    print(f">>> Top-3 Accuracy: {top3_acc:.4f}")

    # Сохранение результатов в JSON
    results = {
        "model": model_name,
        "test_accuracy":  float(test_acc),
        "top3_accuracy":  float(top3_acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "history": {
            k: [float(v) for v in vs] for k, vs in history.history.items()
        },
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "n_params":   int(model.count_params()),
    }
    results_path = RESULTS_DIR / f"{model_name}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ Результаты сохранены: {results_path}")

    # Построение графиков
    plot_training_curves(history.history, model_name)
    plot_confusion_matrix(cm, model_name)

    return test_acc


# =============================================================
# ТОЧКА ВХОДА — РАЗБОР АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ
# =============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение моделей классификации йеменских диалектов"
    )
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES + ["all"],
        default="all",
        help="Какую модель обучать (по умолчанию: все)",
    )
    args = parser.parse_args()

    if args.model == "all":
        accuracies = {}
        for name in MODEL_NAMES:
            accuracies[name] = train_one(name)

        # Итоговое сравнение
        print(f"\n{'='*65}")
        print("СВОДНАЯ ТАБЛИЦА ТОЧНОСТИ:")
        print(f"{'='*65}")
        for name, acc in accuracies.items():
            print(f"  {name.upper():>10} : {acc*100:.2f}%")
    else:
        train_one(args.model)
