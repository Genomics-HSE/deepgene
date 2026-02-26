# GRU Ordinal Architecture (GruLabelerOrdinal)

## Описание

GruLabelerOrdinal - это архитектура для ordinal regression (порядковой регрессии), которая использует GruLabeler в качестве базовой модели и добавляет специальный OrdinalHead для предсказания порядковых меток.

## Назначение

Модель предназначена для задач, где классы имеют естественный порядок (например, временные эпохи). Вместо категориальной классификации используется ordinal regression, которая учитывает порядок классов.

## Архитектура

### Компоненты

1. **GruLabeler (базовая модель)**
   - Полная архитектура GruLabeler:
     - Embedding слой
     - Bidirectional GRU
     - Predictor
   - Выход: (batch, genome_length, d_model)

2. **OrdinalHead**
   - Специальный слой для ordinal regression
   - Структура:
     - Linear(d_model → 1, bias=False)
     - Bias параметр: (n_class - 1)
   - Выход: (batch, genome_length, n_class - 1)

### Поток данных (Forward Pass)

```
Input (batch, genome_length)
  ↓
GruLabeler → (batch, genome_length, d_model)
  ↓
OrdinalHead → (batch, genome_length, n_class - 1)
  ↓
Output (ordinal logits)
```

## Гиперпараметры

- `n_class`: количество классов
- Параметры GruLabeler (embedding, GRU, predictor)
- Параметры OrdinalHead:
  - `d_model`: размерность входных признаков
  - `n_class`: количество классов

## Функция потерь

- **BCELoss**: Binary Cross Entropy Loss
- Преобразование меток: `ordinal_transform` - преобразует категориальные метки в бинарные задачи (y > threshold для каждого threshold)

## Особенности реализации

- Использует ordinal regression вместо категориальной классификации
- OrdinalHead предсказывает n_class - 1 бинарных задач (каждый класс больше предыдущего?)
- Преобразование меток: для каждого класса создается бинарная метка (больше ли значение этого класса?)
- Подходит для задач с естественным порядком классов

## Ordinal Regression

Ordinal regression решает задачу предсказания порядковых меток, где классы имеют естественный порядок. Вместо предсказания одного класса, модель предсказывает вероятность того, что значение больше каждого порога.

## Исходный код

- Реализация: [`deepgene/models/gru.py`](../../deepgene/models/gru.py) (класс GruLabelerOrdinal)
- Базовый класс: [`deepgene/models/base_models.py`](../../deepgene/models/base_models.py) (OrdinalHead, OrdinalModel)
