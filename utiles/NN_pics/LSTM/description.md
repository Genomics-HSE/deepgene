# LSTM Architecture (LstmLabeler)

## Описание

LstmLabeler - это архитектура на основе двунаправленной LSTM (Long Short-Term Memory) сети для классификации последовательностей геномных данных.

## Назначение

Модель предназначена для предсказания категориальных меток (временных эпох) для каждой позиции в геномной последовательности. LSTM способна запоминать долгосрочные зависимости в последовательностях.

## Архитектура

### Компоненты

1. **ConvEmbedding**
   - Сверточный embedding слой (используется по умолчанию)
   - Параметры:
     - `n_layers`: 4
     - `in_channels`: 1
     - `out_channels`: 1 (для совместимости с LSTM input_size=1)
     - `kernel_size`: 21
     - `stride`: 1
     - `padding`: 'same'

2. **Bidirectional LSTM**
   - Двунаправленная LSTM сеть
   - Параметры (hardcoded в коде):
     - `input_size`: 1
     - `hidden_size`: n_class (32)
     - `num_layers`: 4
     - `dropout`: 0.1
     - `batch_first`: True
     - `bidirectional`: True

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
ConvEmbedding → (batch, genome_length, 1)
  ↓
Bidirectional LSTM → (batch, genome_length, hidden_size*2)
  ↓
Predictor → (batch, genome_length, n_class)
  ↓
Output (logits)
```

## Гиперпараметры

- `n_class`: 32 (количество классов для классификации)
- `learning_rate`: 1e-3
- `weight_decay`: 1e-4

## Функции потерь

Поддерживает различные функции потерь:
- `CrossEntropyLoss` (по умолчанию)
- `KLDivLoss`
- `EMD_squared_loss`

## Особенности реализации

- Использует PyTorch Lightning для обучения
- ConvEmbedding по умолчанию для предобработки входных данных
- LSTM параметры жестко заданы в коде (не конфигурируются через gin)
- Двунаправленная архитектура для учета контекста с обеих сторон
- Глубокая сеть (4 слоя) для извлечения сложных паттернов

## Исходный код

- Реализация: [`deepgene/models/lstm.py`](../../deepgene/models/lstm.py)
- Конфигурация: [`configs/model_lstm.gin`](../../configs/model_lstm.gin)
