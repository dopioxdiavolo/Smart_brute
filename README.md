# TapTransformer для Smart Brute Force

Реализация attention-based нейронной сети TapTransformer для задачи multi-label classification техник обхода защиты в системах умного брутфорса.

## Описание задачи

Модель предназначена для предсказания, какие техники обхода защиты (`applied_techniques`) следует применить на основе наблюдаемых аномалий поведения сервера.

### Входные данные

**Категориальные признаки:**
- `anomaly_type`: тип аномалии (STATUS_ANOMALY, TIME_ANOMALY, HEADER_ANOMALY, SIZE_ANOMALY)
- `encoding_type`: тип кодировки ответа
- `error_message`: сообщение об ошибке

**Числовые признаки:**
- `status`: HTTP статус-код
- `depth`: глубина пути
- `size`: размер ответа (для SIZE_ANOMALY)
- `response_time_ms`: время ответа (для TIME_ANOMALY)
- `retry_after_seconds`: время повтора
- Бинарные флаги: `has_location`, `has_set_cookie`, `has_www_authenticate`, `has_retry_after`, `has_auth_token`, `has_cookie`, `has_x_custom_auth`

**Целевая переменная:**
- `applied_techniques`: список техник обхода (multi-label)
- `reward`: вес семпла (0.0-1.0) для weighted loss

### Архитектура модели

1. **Embedding слои** для категориальных признаков
2. **Нормализация** числовых признаков
3. **Multi-Head Attention** механизм
4. **Transformer encoder** блоки
5. **Классификационная голова** для multi-label предсказания

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### Быстрый старт

```python
from tap_transformer_model import TapTransformer, SmartBruteTrainer, SmartBruteDataset
import pandas as pd

# Загрузка данных
df = pd.read_csv('your_data.csv')

# Создание и обучение модели
# (см. функцию main() в tap_transformer_model.py)
```

### Запуск демонстрации

```python
python tap_transformer_model.py
```

## Архитектура

### TapTransformer

Основная модель, включающая:
- Embedding слои для категориальных признаков
- Проекцию числовых признаков
- Позиционное кодирование
- Transformer блоки с Multi-Head Attention
- Классификационную голову

### SmartBruteDataset

Класс для подготовки и загрузки данных:
- Автоматическая обработка категориальных признаков
- Нормализация числовых признаков
- Multi-label кодирование целевых переменных
- Поддержка весов семплов

### SmartBruteTrainer

Класс для обучения модели:
- Weighted BCE loss с учетом reward
- Early stopping
- Learning rate scheduling
- Метрики для multi-label классификации

## Метрики

- **Loss**: Weighted Binary Cross-Entropy
- **Hamming Loss**: Среднее количество неправильно предсказанных меток
- **Subset Accuracy**: Доля точно предсказанных наборов меток

## Техники обхода

Модель предсказывает вероятности применения следующих техник:

- TB_HEADER_BYPASS
- TB_AUTH_ATTEMPT
- TB_ENCODING_BYPASS
- TB_METHOD_OVERRIDE
- TB_PARAM_POLLUTION
- TB_CASE_VARIATION
- TB_URL_ENCODING
- TB_DOUBLE_ENCODING
- TB_NULL_BYTE
- TB_PATH_TRAVERSAL
- TB_HTTP_SMUGGLING
- TB_RACE_CONDITION
- TB_CACHE_POISONING
- TB_SESSION_FIXATION
- TB_CSRF_BYPASS

## Параметры модели

```python
model = TapTransformer(
    categorical_features=categorical_features,  # Словарь {признак: размер_словаря}
    numerical_features=numerical_features,      # Список числовых признаков
    num_techniques=num_techniques,              # Количество техник
    d_model=128,                               # Размерность модели
    n_heads=8,                                 # Количество attention головок
    n_layers=4,                                # Количество Transformer слоев
    d_ff=512,                                  # Размерность Feed Forward
    dropout=0.1,                               # Dropout rate
    embedding_dim=32                           # Размерность embeddings
)
```

## Обучение

```python
trainer = SmartBruteTrainer(model)
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_epochs=50,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=10
)
```

## Предсказание

```python
model.eval()
with torch.no_grad():
    logits = model(categorical_inputs, numerical_inputs)
    probabilities = torch.sigmoid(logits)
    predicted_techniques = probabilities > 0.5
```

## Особенности реализации

1. **Weighted Loss**: Использование reward как веса семпла
2. **Multi-Label**: Поддержка предсказания нескольких техник одновременно
3. **Attention Mechanism**: Изучение взаимосвязей между признаками
4. **Robustness**: Обработка пропущенных значений и неизвестных категорий
5. **Early Stopping**: Предотвращение переобучения

## Результаты

Модель показывает хорошие результаты на задаче предсказания техник обхода:
- Низкий Hamming Loss
- Высокая Subset Accuracy для часто встречающихся комбинаций
- Стабильная сходимость при обучении 