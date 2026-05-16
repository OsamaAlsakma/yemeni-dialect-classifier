# Быстрый старт

Этот файл содержит минимальный набор команд для запуска проекта
от установки до открытия рабочего веб-интерфейса.

## Полный сценарий (с обучением)

```bash
# 1. Подготовка окружения
python3 -m venv venv
source venv/bin/activate              # Linux/macOS
# venv\Scripts\activate               # Windows
pip install -r requirements.txt

# 2. Распаковка датасета в ml/data/dataset/
#    (структура: train/, validate/, test/, каждый с audio/ и metadata.json)

# 3. Извлечение акустических признаков (≈10–15 мин)
cd ml/src
python preprocess.py
cd ../..

# 4. Обучение моделей (на GPU ≈2–3 часа)
cd ml/src
python train.py --model all
cd ../..

# 5. Запуск веб-приложения
cd backend
python manage.py migrate
python manage.py runserver
# → откройте http://localhost:8000
```

## Сценарий без обучения (если веса уже есть)

```bash
# 1. Подготовка окружения
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Поместите файл cnn_lstm_best.h5 в ml/saved_models/

# 3. Запуск веб-приложения
cd backend
python manage.py migrate
python manage.py runserver
# → http://localhost:8000
```

## Запуск отдельных компонентов

### Только обучение одной модели

```bash
cd ml/src
python train.py --model cnn_lstm
```

### Только инференс из командной строки

```bash
cd ml/src
python predict.py /path/to/audio.wav
```

### Только сервер без обучения моделей

```bash
cd backend
python manage.py runserver
# При первом запросе на /api/v1/classify/ модель загрузится автоматически
```

## Проверка работоспособности

```bash
# Статус API
curl http://localhost:8000/api/v1/health/

# Список диалектов
curl http://localhost:8000/api/v1/dialects/

# Классификация файла
curl -X POST http://localhost:8000/api/v1/classify/ \
     -F "audio=@test.wav"
```

## Типичные проблемы

| Ошибка | Решение |
|--------|---------|
| `Модель не найдена` | Запустите `python ml/src/train.py --model cnn_lstm` |
| `Файл metadata.json не найден` | Проверьте путь `ml/data/dataset/train/metadata.json` |
| `CUDA out of memory` | Уменьшите BATCH_SIZE в config.py (например, до 16) |
| `Librosa не читает MP3` | Установите ffmpeg: `sudo apt install ffmpeg` |
| `Module not found` | Проверьте, что venv активирован и зависимости установлены |
