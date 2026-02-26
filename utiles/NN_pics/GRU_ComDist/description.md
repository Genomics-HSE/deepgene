# GRU ComDist Architecture (GruComDistLabeler)

## Описание

GruComDistLabeler - это специализированная версия GRU архитектуры, оптимизированная для работы с распределениями вероятностей с использованием функции потерь KLDivLoss (Kullback-Leibler Divergence).

## Назначение

Модель предназначена для предсказания распределений вероятностей по временным эпохам для каждой позиции в геномной последовательности. Используется когда важна не только точная классификация, но и правильное распределение вероятностей.

## Архитектура

### Компоненты

1. **Embedding слой**
   - Поддерживает три типа embedding:
     - `NoEmbedding`
     - `Embedding`
     - `ConvEmbedding`

2. **Bidirectional GRU**
   - Двунаправленная GRU сеть
   - Параметры:
     - `input_size`: настраивается
     - `hidden_size`: настраивается
     - `num_layers`: настраивается
     - `dropout`: 0.1
     - `batch_first`: True

3. **Predictor**
   - Трехслойный перцептрон (аналогично GruLabeler)

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

- `n_class`: количество классов
- `input_size`: размер входных признаков
- `hidden_size`: размер скрытого состояния GRU
- `num_layers`: количество слоев GRU

## Функция потерь

- **KLDivLoss**: Kullback-Leibler Divergence Loss
  - Используется для сравнения распределений вероятностей
  - Подходит для задач, где важна не только точность, но и правильное распределение вероятностей

## Особенности реализации

- Архитектура идентична GruLabeler, но с фиксированной функцией потерь KLDivLoss
- Оптимизирована для работы с распределениями вероятностей
- Использует те же компоненты (embedding, GRU, predictor), что и стандартный GruLabeler

## Исходный код

- Реализация: [`deepgene/models/gru_com_dist.py`](../../deepgene/models/gru_com_dist.py)
