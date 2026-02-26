# SPIDNA Architecture

## Описание

SPIDNA (Spatial Position and Identity DNA) - это сверточная архитектура, специально разработанная для работы с геномными данными, использующая 2D свертки для обработки позиционной информации и SNP (Single Nucleotide Polymorphism) признаков.

## Назначение

Модель предназначена для анализа геномных данных с учетом как позиционной информации, так и SNP признаков. Использует residual connections и специальные блоки для извлечения признаков.

## Архитектура

### Компоненты

1. **Conv2D для Position**
   - Обрабатывает позиционную информацию
   - Conv2d(1 → num_feature, kernel=(1, 3))
   - BatchNorm2d
   - ReLU активация

2. **Conv2D для SNP**
   - Обрабатывает SNP признаки
   - Conv2d(1 → num_feature, kernel=(1, 3))
   - BatchNorm2d
   - ReLU активация

3. **SPIDNABlock(s)**
   - Повторяющиеся блоки для извлечения признаков
   - Каждый блок содержит:
     - Conv2d(num_feature*2 → num_feature, kernel=(1, 3))
     - BatchNorm2d(num_feature*2)
     - MaxPool2d((1, 2))
     - Linear(num_output → num_output) для residual output
   - Residual connections: выход каждого блока добавляется к общему output

4. **FC Layer**
   - Финальный линейный слой
   - Linear(num_output → num_output)

### Поток данных (Forward Pass)

```
Input (batch, channels, features, seq_len)
  ↓
Position: Conv2D + BatchNorm + ReLU
SNP: Conv2D + BatchNorm + ReLU
  ↓
Concatenate (position + SNP)
  ↓
SPIDNABlock 1 → (features, output_1)
SPIDNABlock 2 → (features, output_2)
...
SPIDNABlock N → (features, output_N)
  ↓
Sum all outputs (residual connections)
  ↓
FC Layer
  ↓
Output (batch, num_output)
```

## Гиперпараметры

- `num_output`: количество выходных классов/значений
- `num_block`: количество SPIDNABlock слоев
- `num_feature`: количество признаков в сверточных слоях

## Особенности реализации

### Residual Connections
- Каждый SPIDNABlock добавляет свой вклад к общему output
- `output = output + current_output` для каждого блока
- Позволяет модели обучаться глубже и избегать проблем с градиентами

### 2D Convolutions
- Работает с 2D данными: (batch, channels, features, sequence_length)
- Position и SNP обрабатываются отдельно, затем объединяются

### Pooling
- MaxPool2d((1, 2)) уменьшает размерность последовательности вдвое на каждом блоке
- Позволяет извлекать признаки на разных масштабах

### SPIDNABlock детали
1. Принимает входные данные и текущий output
2. Применяет BatchNorm и Conv2D
3. Вычисляет среднее по одному измерению (psi)
4. Генерирует текущий output через FC слой
5. Добавляет к общему output (residual)
6. Объединяет psi с входными данными
7. Применяет ReLU и MaxPool

## Варианты архитектуры

### SPIDNA (базовая)
- Использует BatchNorm2d

### SPIDNA2_adaptive
- Использует GroupNorm или InstanceNorm вместо BatchNorm
- Более адаптивная нормализация

## Исходный код

- Реализация: [`deepgene/models/spidna.py`](../../deepgene/models/spidna.py)
