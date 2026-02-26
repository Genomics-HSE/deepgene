# GRU Architecture (GruLabeler)

## Описание

GruLabeler - это архитектура на основе двунаправленной рекуррентной нейронной сети GRU (Gated Recurrent Unit) для классификации последовательностей геномных данных.

## Назначение

Модель предназначена для предсказания категориальных меток (временных эпох) для каждой позиции в геномной последовательности. Используется для анализа демографической истории популяций.

## Архитектура

### Компоненты

1. **Embedding слой**
   - Поддерживает три типа embedding:
     - `NoEmbedding`: Простое расширение размерности (unsqueeze)
     - `Embedding`: Стандартный embedding слой PyTorch
     - `ConvEmbedding`: Сверточный embedding с несколькими слоями Conv1D
   - По умолчанию используется `NoEmbedding`

2. **Bidirectional GRU**
   - Двунаправленная GRU сеть
   - Параметры (из конфигурации):
     - `input_size`: 1
     - `hidden_size`: 32
     - `num_layers`: 4
     - `dropout`: 0.1
     - `batch_first`: True

3. **Predictor**
   - Трехслойный перцептрон:
     - Linear(d_model → d_model) + ReLU + Dropout
     - Linear(d_model → d_model) + ReLU + Dropout
     - Linear(d_model → n_class)
   - Параметры:
     - `d_model`: 64
     - `dropout`: 0.1
     - `n_class`: 32

### Поток данных (Forward Pass)

```
Input (batch, genome_length)
  ↓
Embedding
  ↓
Bidirectional GRU → (batch, genome_length, hidden_size*2)
  ↓
Predictor → (batch, genome_length, n_class)
  ↓
Output (logits)
```

## Гиперпараметры

- `n_class`: 32 (количество классов для классификации)
- `input_size`: 1
- `hidden_size`: 32
- `num_layers`: 4
- `learning_rate`: 1e-3
- `weight_decay`: 1e-4

## Функции потерь

Поддерживает различные функции потерь:
- `CrossEntropyLoss` (по умолчанию)
- `KLDivLoss`
- `EMD_squared_loss`
- `L1Loss`
- `HuberLoss`
- `NllLoss`
- `CEandEMD`
- `CDW_CELoss`

## Особенности реализации

- Использует PyTorch Lightning для обучения
- Поддерживает различные типы embedding через конфигурацию
- Двунаправленная архитектура позволяет учитывать контекст с обеих сторон последовательности
- Глубокая сеть (4 слоя) для лучшего извлечения признаков

## Исходный код

- Реализация: [`deepgene/models/gru.py`](../../deepgene/models/gru.py)
- Конфигурация: [`configs/model_gru.gin`](../../configs/model_gru.gin)
