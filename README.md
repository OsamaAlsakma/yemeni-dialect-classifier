# Yemeni Dialect Classifier

**Программный комплекс автоматической классификации йеменских диалектов
с веб-интерфейсом на Django**

Альсакма Осама Салех Мохаммед — группа 2310, кафедра ВТ, СПбГЭТУ «ЛЭТИ», 2026.

---

## Содержание

1. Общие сведения
2. Структура проекта
3. Требования
4. Установка
5. Подготовка датасета
6. Обучение моделей
7. Запуск веб-приложения
8. REST API
9. Описание модулей
10. Веб-интерфейс
11. Развёртывание

---

## 1. Общие сведения

Проект состоит из трёх логически разделённых компонентов:

| Компонент | Назначение |
|-----------|------------|
| **ml** | Машинное обучение: предобработка, обучение, инференс |
| **backend** | Django-сервер: REST API + страницы |
| **frontend** | Шаблоны HTML, CSS, JavaScript, локализация |

### Поддерживаемые диалекты (11 классов)

1. صنعاء — Сана
2. عدن — Аден
3. شبوة — Шабва
4. حضرموت الساحل — Хадрамаут (побережье)
5. ذمار — Зимар
6. الحديدة (تهامي — Ходейда (тихамий)
7. تعز — Таиз
8. الضالع — Эд-Далеа
9. إب — Ибб
10. البيضاء — Эль-Байда
11. المحويت — Эль-Махвит

### Статистика датасета YDC v1.0

| Параметр | Значение |
|----------|----------|
| Всего записей | 3 315 |
| Обучающая выборка (60%) | 2 302 |
| Валидационная выборка (20%) | 758 |
| Тестовая выборка (20%) | 255 |
| Уникальных фраз | 755 |
| Диалектов | 11 |

---

## 2. Структура проекта

```
yemeni-dialect-classifier/
│
├── ml/                           # ── Машинное обучение
│   ├── src/
│   │   ├── config.py             # глобальные настройки
│   │   ├── preprocess.py         # предобработка + признаки
│   │   ├── models.py             # CNN, LSTM, CNN+LSTM
│   │   ├── train.py              # обучение и оценка
│   │   └── predict.py            # инференс (используется API)
│   ├── data/dataset/             # сюда распаковать датасет
│   ├── features/                 # кэш features_*.npz
│   ├── saved_models/             # обученные веса .h5
│   ├── logs/                     # CSV-журналы обучения
│   └── results/                  # JSON результаты, графики
│
├── backend/                      # ── Django-сервер
│   ├── manage.py
│   ├── project_config/           # настройки проекта
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── classifier/               # приложение страниц
│   │   ├── views.py
│   │   └── urls.py
│   └── api/                      # приложение REST API
│       ├── views.py
│       └── urls.py
│
├── frontend/                     # ── Клиентская часть
│   ├── templates/
│   │   ├── base.html             # базовый шаблон
│   │   ├── home.html             # главная страница
│   │   └── about.html            # о проекте
│   ├── static/
│   │   ├── css/
│   │   │   ├── main.css
│   │   │   └── rtl.css           # стили для арабского
│   │   └── js/
│   │       ├── main.js           # общая логика
│   │       ├── recorder.js       # запись с микрофона
│   │       └── classifier.js     # отправка в API
│   └── locale/                   # переводы (i18n)
│
├── scripts/                      # вспомогательные скрипты
├── requirements.txt              # зависимости Python
└── README.md                     # настоящее руководство
```

---

## 3. Требования

### Минимальные
- Python 3.10 или выше
- 8 ГБ оперативной памяти
- 10 ГБ свободного места на диске
- Современный браузер (Chrome, Firefox, Safari, Edge)

### Рекомендуемые (для обучения)
- NVIDIA GPU с 8 ГБ VRAM
- CUDA 11.8 + cuDNN 8.6
- 16 ГБ оперативной памяти

### Альтернатива: облако
- Google Colab Pro (GPU Tesla T4)
- Kaggle Notebooks (GPU Tesla P100)

---

## 4. Установка

### Шаг 1. Клонирование репозитория

```bash
git clone https://github.com/alsakmaosama/yemeni-dialect-classifier.git
cd yemeni-dialect-classifier
```

### Шаг 2. Создание виртуального окружения

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Шаг 3. Установка зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Шаг 4. Проверка установки

```bash
python -c "import tensorflow as tf; print('TF:', tf.__version__)"
python -c "import django; print('Django:', django.get_version())"
```

---

## 5. Подготовка датасета

### Распаковка датасета

Поместите ZIP-архив с датасетом в `ml/data/`, затем распакуйте:

```bash
cd ml/data
unzip yemen-dialect-voice-dataset.zip -d dataset/
```

### Ожидаемая структура

После распаковки структура должна быть следующей:

```
ml/data/dataset/
├── train/
│   ├── audio/
│   │   ├── YEM_xxxxx_xxxx.wav
│   │   └── ...
│   └── metadata.json          # метаданные обучающей выборки
├── validate/
│   ├── audio/
│   └── metadata.json
├── test/
│   ├── audio/
│   └── metadata.json
└── metadata.json              # полные метаданные
```

### Извлечение признаков

```bash
cd ml/src
python preprocess.py
```

Время выполнения: примерно 10–15 минут для 3 315 записей.

В результате создаются три файла в `ml/features/`:
- `features_train.npz`
- `features_validate.npz`
- `features_test.npz`

---

## 6. Обучение моделей

### Запуск обучения

```bash
cd ml/src

# Обучение всех трёх моделей последовательно
python train.py --model all

# Или только одной модели
python train.py --model cnn_lstm   # рекомендуется
python train.py --model cnn
python train.py --model lstm
```

### Время обучения

| Платформа | Одна модель | Все три |
|-----------|-------------|---------|
| CPU (Intel i5) | 8–10 ч | 24–30 ч |
| GPU Tesla T4 | 30–60 мин | 2–3 ч |
| GPU RTX 3080 | 15–25 мин | 1–1.5 ч |

### Что сохраняется после обучения

| Файл | Назначение |
|------|------------|
| `ml/saved_models/<name>_best.h5` | веса лучшей версии модели |
| `ml/logs/<name>_log.csv` | журнал обучения по эпохам |
| `ml/results/<name>_results.json` | метрики, история, матрица ошибок |
| `ml/results/<name>_training_curves.png` | график loss/accuracy |
| `ml/results/<name>_confusion_matrix.png` | матрица ошибок |

---

## 7. Запуск веб-приложения

После того как хотя бы одна модель обучена:

```bash
cd backend

# Применение миграций базы данных
python manage.py migrate

# Сбор статических файлов
python manage.py collectstatic --noinput

# Запуск сервера разработки
python manage.py runserver
```

Откройте в браузере: **http://localhost:8000**

### Переключение языков

В правом верхнем углу страницы выберите язык:
- 🇷🇺 Русский
- 🇸🇦 العربية

---

## 8. REST API

### POST /api/v1/classify/

Классификация загруженного аудио.

**Запрос:**
```bash
curl -X POST http://localhost:8000/api/v1/classify/ \
     -F "audio=@my_recording.wav" \
     -F "top_n=3"
```

**Ответ:**
```json
{
  "success": true,
  "result": {
    "model_used": "cnn_lstm",
    "top_dialect_ar": "عدن",
    "top_dialect_ru": "Аден",
    "top_region_ru": "Юг Йемена, портовый город",
    "top_confidence": 0.873,
    "top_confidence_pct": 87.32,
    "top_n": [
      {
        "index": 1,
        "dialect_ar": "عدن",
        "dialect_ru": "Аден",
        "region_ru": "Юг Йемена, портовый город",
        "probability": 0.873,
        "confidence_pct": 87.32
      },
      ...
    ],
    "all_probabilities": {...}
  },
  "filename": "my_recording.wav",
  "file_size_kb": 124.5
}
```

### GET /api/v1/health/

Проверка статуса сервиса.

```bash
curl http://localhost:8000/api/v1/health/
```

### GET /api/v1/dialects/

Список всех 11 диалектов.

### GET /api/v1/stats/

Статистика датасета.

---

## 9. Описание модулей

### 9.1 ml/src/config.py

Централизованное хранение всех параметров: пути, диалекты,
параметры аудио, гиперпараметры обучения.

### 9.2 ml/src/preprocess.py

- `load_audio(path)` — загрузка и нормализация WAV-файла
- `extract_mel(y)` — извлечение мел-спектрограммы (128 × T)
- `extract_mfcc(y)` — MFCC + Δ + ΔΔ → (T × 120)
- `build_dataset(split, meta_path)` — сборка признаков выборки в NPZ

### 9.3 ml/src/models.py

- `build_cnn()` — 4 свёрточных блока + GAP + Dense
- `build_lstm()` — 2 BiLSTM + Attention Pooling
- `build_cnn_lstm()` — гибрид: 3 Conv + 2 BiLSTM + Attention

### 9.4 ml/src/train.py

- `load_features()` — загрузка предвычисленных признаков
- `add_specaugment()` — аугментация SpecAugment
- `train_one(model_name)` — полный цикл обучения и оценки
- Сохранение графиков обучения и матриц ошибок

### 9.5 ml/src/predict.py

- класс `DialectPredictor` — singleton-обёртка над моделью
- метод `predict(audio_path)` — полный инференс
- функция `get_predictor()` — кэшированный экземпляр

### 9.6 backend/api/views.py

- `classify_audio` — POST endpoint классификации
- `health_check` — статус сервиса
- `list_dialects` — список диалектов
- `dataset_stats` — статистика датасета

### 9.7 backend/classifier/views.py

- `home` — главная страница с формой
- `about` — страница о проекте

---

## 10. Веб-интерфейс

### Возможности

| Функция | Описание |
|---------|----------|
| Загрузка файла | Drag-and-drop или клик. WAV/MP3/FLAC/OGG/M4A до 25 МБ |
| Запись с микрофона | Прямая запись через браузер (MediaRecorder API) |
| Классификация | Отправка в API + отображение результата |
| Top-3 диалектов | С прогресс-барами и описанием регионов |
| Двуязычность | Русский + Арабский (с RTL-поддержкой) |
| Адаптивность | Корректно работает на мобильных устройствах |

### Архитектура взаимодействия

```
Браузер              Django                    ML-модель
   │                    │                          │
   │  POST /api/        │                          │
   │  classify/         │                          │
   ├───────────────────>│                          │
   │  (audio file)      │                          │
   │                    │  predictor.predict()     │
   │                    ├─────────────────────────>│
   │                    │                          │
   │                    │  result (JSON)           │
   │                    │<─────────────────────────┤
   │                    │                          │
   │  JSON response     │                          │
   │<───────────────────┤                          │
   │                    │                          │
   ▼                    │                          │
Отображение                                        │
результата                                         │
```

---

## 11. Развёртывание

### Production-настройки

В `backend/project_config/settings.py` для production:

```python
DEBUG = False
SECRET_KEY = "ваш-секретный-ключ-из-переменных-окружения"
ALLOWED_HOSTS = ["yourdomain.com"]
CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOWED_ORIGINS = ["https://yourdomain.com"]
```

### Запуск через Gunicorn + Nginx

```bash
pip install gunicorn
cd backend
gunicorn project_config.wsgi --bind 0.0.0.0:8000 --workers 3
```

Конфигурация Nginx (фрагмент):

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location /static/ {
        alias /path/to/project/backend/staticfiles/;
    }

    location /media/ {
        alias /path/to/project/backend/media/;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        client_max_body_size 25M;
    }
}
```

### Docker (опционально)

Пример `Dockerfile`:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
WORKDIR /app/backend
RUN python manage.py collectstatic --noinput
EXPOSE 8000
CMD ["gunicorn", "project_config.wsgi", "--bind", "0.0.0.0:8000"]
```

---

## Контакты

**Автор:** Альсакма Осама Салех Мохаммед
**Email:** alsakmaosama@gmail.com
**Группа:** 2310
**Кафедра:** ВТ, СПбГЭТУ «ЛЭТИ»

**Научный руководитель:** к.т.н., доц. Аббас Саддам Ахмед Мохаммед

---

Санкт-Петербург, 2026
