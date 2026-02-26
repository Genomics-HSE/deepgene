# DeepCompressor Architecture

## Описание

DeepCompressor - это сверточная архитектура для сжатия и анализа последовательностей геномных данных. Использует последовательность сверточных блоков с пулингом для постепенного уменьшения размерности последовательности.

## Назначение

Модель предназначена для сжатия длинных последовательностей и предсказания распределений вероятностей по временным эпохам. Используется для задач, где требуется уменьшить размерность данных при сохранении важной информации.

## Архитектура

### Компоненты

1. **ConvBlock(s)**
   - Последовательность сверточных блоков
   - Каждый блок содержит:
     - Conv1d(in_channels → out_channels, kernel_size, stride)
     - BatchNorm1d
     - ReLU активация
     - Dropout
     - MaxPool1d(kernel_size)
   - Структура каналов:
     - Первый блок: (1 → channel_size)
     - Промежуточные блоки: (channel_size → channel_size)
     - Последний блок: (channel_size → 1)

2. **Dense Layers**
   - Три полносвязных слоя:
     - Dense1: Linear(seq_len → seq_len) + ReLU + Dropout
     - Dense2: Linear(seq_len → seq_len) + ReLU + Dropout
     - Dense3: Linear(seq_len → n_output) + LogSoftmax

### Поток данных (Forward Pass)

```
Input (batch, seq_len)
  ↓
Unsqueeze → (batch, 1, seq_len)
  ↓
ConvBlock 1 → (batch, channel_size, seq_len_1)
ConvBlock 2 → (batch, channel_size, seq_len_2)
...
ConvBlock N → (batch, 1, seq_len_N)
  ↓
Squeeze → (batch, seq_len_N)
  ↓
Dense1 + ReLU + Dropout
  ↓
Dense2 + ReLU + Dropout
  ↓
Dense3 + LogSoftmax
  ↓
Output (batch, n_output)
```

## Гиперпараметры

- `channel_size`: количество каналов в сверточных слоях
- `conv_kernel_size`: размер ядра свертки
- `conv_stride`: шаг свертки
- `num_layers`: количество ConvBlock слоев
- `dropout_p`: вероятность dropout
- `pool_kernel_size`: размер ядра пулинга
- `n_output`: размер выходного распределения
- `seq_len`: исходная длина последовательности

## Особенности реализации

### Постепенное сжатие
- Каждый ConvBlock уменьшает размерность последовательности:
  - Conv1d: `output_size = floor((input_size - kernel_size + 2*padding) / stride + 1)`
  - MaxPool1d: `output_size = floor((input_size - pool_kernel_size) / pool_kernel_size + 1)`
- Последовательность постепенно сжимается от исходной длины до финальной

### Структура каналов
- Начинается с 1 канала (входные данные)
- Расширяется до `channel_size` каналов
- Сжимается обратно до 1 канала перед dense слоями

### Функция потерь
- **KLDivLoss**: Kullback-Leibler Divergence Loss
- Используется для предсказания распределений вероятностей

### Выходной формат
- LogSoftmax вместо обычного Softmax
- Позволяет напрямую использовать с KLDivLoss без дополнительных вычислений

## Пример расчета размеров

Для последовательности длиной `seq_len`:
- После каждого ConvBlock размер уменьшается
- Финальный размер перед dense слоями: `final_seq_len`
- Dense слои работают с этим размером

## Исходный код

- Реализация: [`deepgene/models/compress.py`](../../deepgene/models/compress.py)
