# Art Style Classifier

Система для автоматического определения стиля художественных произведений на основе глубокого обучения.

## Описание

Проект классифицирует картины по 27 стилистическим направлениям с использованием нейронных сетей. Система предназначена для художников, галерей и музеев.

**Датасет:** [WikiArt](https://huggingface.co/datasets/huggan/wikiart) (11,300+ изображений, 27 стилей)

**Модели:**

- Baseline: ResNet18
- Advanced: EfficientNet-B0

---

## Setup

### Установка окружения

```bash
# Клонируем репозиторий
git clone https://github.com/yourusername/art-style-classifier.git
cd art-style-classifier

# Создаём виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate  # На Linux/Mac
# или .venv\Scripts\activate на Windows

# Устанавливаем зависимости
pip install -e ".[dev]"
```

### Pre-commit хуки

```bash
pre-commit install
pre-commit run --all-files  # Проверить качество кода
```

---

## Train

### 1. Загрузка данных

```bash
# Полный датасет
python -m art_style_classifier.data.download

# Тестовая подвыборка (для быстрого тестирования)
python -m art_style_classifier.data.test_download
```

### 2. Обучение модели

**Базовый запуск:**

```bash
python -m art_style_classifier.commands train
```

**С переопределением параметров:**

```bash
python -m art_style_classifier.commands train \
  model.architecture=efficientnet_b0 \
  training.max_epochs=30 \
  training.learning_rate=0.0001
```

**Параметры Hydra:**

- `model.architecture` - модель (resnet18 или efficientnet_b0)
- `training.max_epochs` - количество эпох
- `training.learning_rate` - скорость обучения
- `data.batch_size` - размер батча

### 3. Мониторинг обучения

```bash
mlflow ui --host 127.0.0.1 --port 8080
```

Откройте `http://127.0.0.1:8080` для просмотра метрик.

---

## Результаты

### Метрики моделей

| Модель                     | Top-1 Accuracy | Top-3 Accuracy |
| -------------------------- | -------------- | -------------- |
| ResNet18 (Baseline)        | 0.42           | 0.73           |
| EfficientNet-B0 (Advanced) | 0.44           | 0.72           |

---

## API

### Запуск сервера

```bash
uvicorn art_style_classifier.api:app --host 0.0.0.0 --port 8000
```

### Пример использования

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.jpg" \
  -F "top_k=5"
```

---

## Структура проекта

```
art-style-classifier/
├── README.md
├── pyproject.toml
├── .pre-commit-config.yaml
├── configs/                    # Hydra конфиги
├── art_style_classifier/       # Основной пакет
│   ├── data/                   # Загрузка и обработка данных
│   ├── models/                 # Модели (baseline, advanced)
│   ├── commands.py             # CLI команды
│   ├── api.py                  # FastAPI приложение
│   └── config.py               # Конфигурация
├── data/raw/                   # Данные (управляются DVC)
└── outputs/                    # Результаты обучения
```

---

## Требования

- Python 3.9+
- GPU рекомендуется (CUDA 11.8+)
- 10GB свободного места для датасета

---

## Лицензия

MIT License
