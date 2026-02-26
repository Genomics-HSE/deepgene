#!/bin/bash
#SBATCH --job-name=${MODEL}_${DATASET}_${LOSS}_${OUTPUT_DIR}

# Формат job-name: MODEL_DATASET_LOSS_OUTPUT_DIR
# Пример: gru_const_ce_results


# Универсальный скрипт для запуска экспериментов с правильными конфигурациями
# Использование: sbatch --export=MODEL=gru,DATASET=const,LOSS=ce run_universal.sh

module load Python/Anaconda_v10.2019
source deactivate
source activate my_py_env

# CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1


# Проверка доступности CometML сервиса
echo "🔍 Проверка доступности CometML сервиса..."
echo "CometML service check:" 
COMET_RESPONSE=$(curl -s https://www.comet-ml.com/clientlib/isAlive/ver)
echo "$COMET_RESPONSE" 
if [ $? -eq 0 ]; then
    echo "✅ CometML сервис доступен"
    echo "✅ CometML service is available" 
    # Извлекаем версию из ответа
    COMET_VERSION=$(echo "$COMET_RESPONSE" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    if [ ! -z "$COMET_VERSION" ]; then
        echo "📊 Версия CometML: $COMET_VERSION"
        echo "📊 CometML version: $COMET_VERSION" 
    fi
else
    echo "❌ CometML сервис недоступен"
    echo "❌ CometML service is not available" 
fi
echo "---" 


# Определяем параметры
MODEL_NAME="${MODEL}_${DATASET}_${LOSS}.ckpt"
COMET_NAME="${MODEL}_${DATASET}_${LOSS}"
# Используем OUTPUT_DIR из переменной окружения или дефолтное значение
OUTPUT_DIR="${OUTPUT_DIR:-results}"
MODEL_OUTPUT_DIR="${OUTPUT_DIR}/model"
EXPERIMENT_NAME="${MODEL}_${DATASET}_${LOSS}"
# Используем SEQ_LEN из переменной окружения или дефолтное значение
SEQ_LEN="${SEQ_LEN:-3000000}"


# export COMET_LOGGING_CONSOLE=info
# export COMET_LOGGING_FILE="${OUTPUT_DIR}/out/${EXPERIMENT_NAME}_comet.log"
# export COMET_LOGGING_FILE_LEVEL=debug


# Определяем класс модели и конфигурацию
case $MODEL in
    "gru")
        MODEL_CLASS="GruLabeler"
        MODEL_CONFIG="configs/model_gru.gin"
        VAL_CHECK_INTERVAL=50
        ;;
    "lstm")
        MODEL_CLASS="LstmLabeler"
        MODEL_CONFIG="configs/model_lstm.gin"
        VAL_CHECK_INTERVAL=50
        ;;
    "reformer")
        MODEL_CLASS="ReformerLabeler"
        MODEL_CONFIG="configs/model_reformer.gin"
        VAL_CHECK_INTERVAL=125  # Примерно в 2.5 раза больше для компенсации скорости
        ;;
    *)
        echo "Ошибка: неизвестная модель $MODEL"
        exit 1
        ;;
esac

# Определяем генератор демографических событий
case $DATASET in
    "const")
        DEMOGRAPHIC_EVENTS="const"
        ;;
    "hl")
        DEMOGRAPHIC_EVENTS="humanlike"
        ;;
    "hl2")
        DEMOGRAPHIC_EVENTS="humanlike_2"
        ;;
    "hl2_1")
        DEMOGRAPHIC_EVENTS="humanlike_2_1"
        ;;
    "set1")
        DEMOGRAPHIC_EVENTS="set1"
        ;;
    "set2")
        DEMOGRAPHIC_EVENTS="set2"
        ;;
    "set2_plus")
        DEMOGRAPHIC_EVENTS="set2_plus"
        ;;
    "set3")
        DEMOGRAPHIC_EVENTS="set3"
        ;;
    *)
        echo "Ошибка: неизвестный датасет $DATASET"
        exit 1
        ;;
esac

# Определяем функцию потерь
case $LOSS in
    "ce")
        LOSS_FUNCTION="CrossEntropyLoss"
        ;;
    "kl")
        LOSS_FUNCTION="KLDivLoss"
        ;;
    *)
        echo "Ошибка: неизвестная функция потерь $LOSS"
        exit 1
        ;;
esac


# Создаем директорию для конфигураций
CONFIG_DIR="${OUTPUT_DIR}/configs"
mkdir -p "$CONFIG_DIR"

# Копируем конфигурационные файлы в OUTPUT_DIR
echo "📁 Копирование конфигурационных файлов в $CONFIG_DIR..."
cp configs/data_universal.gin "$CONFIG_DIR/"
cp configs/train_universal.gin "$CONFIG_DIR/"
cp "$MODEL_CONFIG" "$CONFIG_DIR/"
cp deepgene/data/limits.py "$CONFIG_DIR/"
echo "✅ Конфигурационные файлы скопированы:"
echo "  - data_universal.gin"
echo "  - train_universal.gin" 
echo "  - $(basename "$MODEL_CONFIG")"
echo "  - limits.py"
echo "  Директория: $CONFIG_DIR"
echo " "


echo "Запуск эксперимента:"
echo "  Модель: $MODEL ($MODEL_CLASS)"
echo "  Датасет: $DATASET ($DEMOGRAPHIC_EVENTS)"
echo "  Функция потерь: $LOSS ($LOSS_FUNCTION)"
echo "  Длина последовательности: $SEQ_LEN"
echo "  Имя модели: $MODEL_NAME"
echo "  Базовая директория: $OUTPUT_DIR"
echo "  Выходная директория для модели: $MODEL_OUTPUT_DIR"
echo "  Имя эксперимента: $EXPERIMENT_NAME"
echo "  Конфигурация модели: $MODEL_CONFIG"
echo "  Val check interval: $VAL_CHECK_INTERVAL"
echo " "

# Запускаем с параметрами

# Имя функции-генератора для gin
GEN_FN="get_${DEMOGRAPHIC_EVENTS}"
if [ "$DEMOGRAPHIC_EVENTS" = "const" ]; then
  GEN_FN="get_const_demographcs"
fi

CMD="srun python main.py \
    --model=\"$MODEL_CONFIG\" \
    --data=\"configs/data_universal.gin\" \
    --train=\"configs/train_universal.gin\" \
    --gin_param=\"model_name='$MODEL_NAME'\" \
    --gin_param=\"output_dir='$MODEL_OUTPUT_DIR'\" \
    --gin_param=\"experiment_name='$EXPERIMENT_NAME'\" \
    --gin_param=\"model_class='$MODEL_CLASS'\" \
    --gin_param=\"loss_function='$LOSS_FUNCTION'\" \
    --gin_param=\"get_liner_generator.demographic_events_generator = @${GEN_FN}\" \
    fit"

echo "$CMD"
echo " "

srun python main.py \
    --model="$MODEL_CONFIG" \
    --data="configs/data_universal.gin" \
    --train="configs/train_universal.gin" \
    --gin_param="model_name='$MODEL_NAME'" \
    --gin_param="output='$MODEL_OUTPUT_DIR'" \
    --gin_param="CometLogger.experiment_name='$COMET_NAME'" \
    --gin_param="model_class='$MODEL_CLASS'" \
    --gin_param="loss_function='$LOSS_FUNCTION'" \
    --gin_param="seq_len=$SEQ_LEN" \
    --gin_param="fit_model.model = @$MODEL_CLASS()" \
    --gin_param="test_model.model = @$MODEL_CLASS()" \
    --gin_param="CometLogger.save_dir = '$MODEL_OUTPUT_DIR'" \
    --gin_param="get_liner_generator.demographic_events_generator = @${GEN_FN}" \
    --gin_param="Trainer.val_check_interval=$VAL_CHECK_INTERVAL" \
    fit


