# Reformer Architecture (ReformerLabeler)

## Описание

ReformerLabeler - это архитектура на основе Reformer Transformer, оптимизированная для работы с длинными последовательностями. Reformer использует LSH (Locality-Sensitive Hashing) attention и axial positional encoding для эффективной обработки длинных последовательностей.

## Назначение

Модель предназначена для классификации очень длинных геномных последовательностей (до 30000+ позиций), где стандартные трансформеры неэффективны из-за квадратичной сложности attention механизма.

## Архитектура

### Компоненты

1. **ConvEmbedding**
   - Сверточный embedding слой
   - Параметры:
     - `n_layers`: 4
     - `in_channels`: 1
     - `out_channels`: 16 (совпадает с hidden_size Reformer)
     - `kernel_size`: 21
     - `stride`: 1

2. **Reformer Transformer**
   - Архитектура Reformer с чередующимися типами attention:
     - `attn_layers`: ["local", "lsh", "local", "lsh", "local", "lsh"]
     - 6 слоев attention (3 local + 3 LSH)
   - Параметры (из конфигурации):
     - `hidden_size`: 16
     - `num_attention_heads`: 2
     - `attention_head_size`: 16
     - `feed_forward_size`: 16
     - `axial_pos_embds_dim`: [6, 10] (сумма = hidden_size)
     - `axial_pos_shape`: [128, 235] (произведение = 30080)
     - `local_chunk_length`: 10000
     - `lsh_attn_chunk_length`: 128
     - `num_hashes`: 2

3. **Predictor**
   - Трехслойный перцептрон
   - Параметры:
     - `d_model`: 32
     - `dropout`: 0.1
     - `n_class`: 32

### Поток данных (Forward Pass)

```
Input (batch, genome_length)
  ↓
Padding до required_length (30080)
  ↓
ConvEmbedding → (batch, seq_len, hidden_size)
  ↓
Reformer Transformer → (batch, seq_len, hidden_size)
  ↓
Predictor → (batch, seq_len, n_class)
  ↓
Обрезка до original_seq_len
  ↓
Output (logits)
```

## Гиперпараметры

- `n_class`: 32
- `hidden_size`: 16
- `num_attention_heads`: 2
- `learning_rate`: 1e-3
- `weight_decay`: 1e-4

## Функции потерь

Поддерживает различные функции потерь:
- `CrossEntropyLoss` (по умолчанию)
- `KLDivLoss`
- `EMD_squared_loss`

## Особенности реализации

### LSH Attention
- Использует Locality-Sensitive Hashing для эффективного вычисления attention
- Сложность O(n log n) вместо O(n²) для стандартного attention
- `lsh_attn_chunk_length`: 128

### Local Attention
- Стандартный attention в локальных окнах
- `local_chunk_length`: 10000

### Axial Positional Encoding
- Позиционное кодирование разделено на две оси (2D)
- `axial_pos_shape`: [128, 235] - позиции представлены как 2D сетка
- `axial_pos_embds_dim`: [6, 10] - размерности embedding для каждой оси

### Padding и обрезка
- Входные последовательности дополняются до `required_length = prod(axial_pos_shape) = 30080`
- После обработки выходные данные обрезаются до исходной длины

## Преимущества Reformer

1. **Масштабируемость**: Может обрабатывать последовательности длиной до 30000+ позиций
2. **Эффективность**: LSH attention снижает сложность с O(n²) до O(n log n)
3. **Гибкость**: Чередование local и LSH attention для баланса между точностью и эффективностью

## Исходный код

- Реализация: [`deepgene/models/reformer.py`](../../deepgene/models/reformer.py)
- Конфигурация: [`configs/model_reformer.gin`](../../configs/model_reformer.gin)
