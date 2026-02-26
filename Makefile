# Упрощенный Makefile для DeepGene
# Параметры по умолчанию
GPU = 1
CPU = 8
T = 600
CONSTRAINT = "type_a|type_b|type_c|type_d"
PROJECT = proj_1801
OUTPUT_DIR = results
SEQ_LEN = 3000000

# Модели
MODELS = gru lstm reformer

# Датасеты
DATASETS = const hl hl2 hl2_1 set1 set2 set2_plus set3

# Функции потерь
LOSSES = ce kl

# Создание папок для логов и моделей
$(shell mkdir -p $(OUTPUT_DIR)/out $(OUTPUT_DIR)/model)

# Основная функция запуска универсального эксперимента
define run_universal_experiment
	@echo "Запуск $(1)_$(2)_$(3)..."
	sbatch --constraint=$(CONSTRAINT) --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t $(T) -A $(PROJECT) \
		--job-name="$(1)_$(2)_$(3)_$(OUTPUT_DIR)" \
		--output=$(OUTPUT_DIR)/out/$(1)_$(2)_$(3)_%j.out \
		--error=$(OUTPUT_DIR)/out/$(1)_$(2)_$(3)_%j.err \
		--export=ALL,MODEL=$(1),DATASET=$(2),LOSS=$(3),OUTPUT_DIR=$(OUTPUT_DIR),SEQ_LEN=$(SEQ_LEN) run_universal.sh
endef

# Функция запуска всех экспериментов для конкретной модели
define run_model_experiments
	$(foreach dataset,$(DATASETS), \
		$(foreach loss,$(LOSSES), \
			$(call run_universal_experiment,$(1),$(dataset),$(loss))))
endef

# Основные цели
.PHONY: all gru-run lstm-run reformer-run run clean clean-logs help mark-all

# Запуск всех экспериментов
all:
	@echo "Запуск всех экспериментов..."
	$(call run_model_experiments,gru)
	$(call run_model_experiments,lstm)
	$(call run_model_experiments,reformer)

# Запуск экспериментов для конкретных моделей
gru-run:
	@echo "Запуск экспериментов GRU..."
	$(call run_model_experiments,gru)

lstm-run:
	@echo "Запуск экспериментов LSTM..."
	$(call run_model_experiments,lstm)

reformer-run:
	@echo "Запуск экспериментов Reformer..."
	$(call run_model_experiments,reformer)

# Запуск всех архитектур на const демографии с KL loss
const-kl:
	@echo "Запуск всех архитектур на const демографии с KL loss..."
	$(call run_universal_experiment,gru,const,kl)
	$(call run_universal_experiment,lstm,const,kl)
	$(call run_universal_experiment,reformer,const,kl)

# Запуск всех архитектур на hl демографии с CE loss
hl-ce:
	@echo "Запуск всех архитектур на hl демографии с CE loss..."
	$(call run_universal_experiment,gru,hl,ce)
	$(call run_universal_experiment,lstm,hl,ce)
	$(call run_universal_experiment,reformer,hl,ce)

# Запуск конкретного эксперимента
# Использование: make run MODEL=gru DATASET=const LOSS=ce
run:
	@echo "Запуск эксперимента $(MODEL)_$(DATASET)_$(LOSS)..."
	$(call run_universal_experiment,$(MODEL),$(DATASET),$(LOSS))

# Очистка временных файлов
clean:
	@echo "Очистка временных файлов..."
	rm -rf $(OUTPUT_DIR)/out/*.out
	find . -name "*.out" -delete
	find . -name ".DS_Store" -delete

# Очистка только логов
clean-logs:
	@echo "Очистка логов..."
	rm -rf $(OUTPUT_DIR)/out/*.out

# Запуск анализа моделей через mark_all.py
# Использование: make mark-all INPUT_DIR=/path/to/models
mark-all:
	@if [ -z "$(INPUT_DIR)" ]; then \
		echo "Ошибка: необходимо указать INPUT_DIR"; \
		echo "Использование: make mark-all INPUT_DIR=/path/to/models"; \
		exit 1; \
	fi
	@echo "Запуск анализа моделей для директории $(INPUT_DIR)..."
	@mkdir -p $(INPUT_DIR)/logs
	@echo "Запуск анализа GRU..."
	@sbatch --constraint=$(CONSTRAINT) --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t 10:00:00 -A $(PROJECT) \
		--job-name="mark_all_gru_$(shell basename $(INPUT_DIR))" \
		--output=$(OUTPUT_DIR)/out/mark_all_gru_%j.out \
		--error=$(OUTPUT_DIR)/out/mark_all_gru_%j.err \
		--export=ALL,INPUT_DIR=$(INPUT_DIR),MODEL_TYPE=gru \
		--wrap="module load Python/Anaconda_v10.2019 && source deactivate && source activate my_py_env && mkdir -p $(INPUT_DIR)/logs && python mark_all.py --input_dir $(INPUT_DIR) --model_type gru --allow_mixed --device cuda 2>&1 | tee $(INPUT_DIR)/logs/mark_all_gru_\$$SLURM_JOB_ID.log"
	@echo "Запуск анализа LSTM..."
	@sbatch --constraint=$(CONSTRAINT) --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t 10:00:00 -A $(PROJECT) \
		--job-name="mark_all_lstm_$(shell basename $(INPUT_DIR))" \
		--output=$(OUTPUT_DIR)/out/mark_all_lstm_%j.out \
		--error=$(OUTPUT_DIR)/out/mark_all_lstm_%j.err \
		--export=ALL,INPUT_DIR=$(INPUT_DIR),MODEL_TYPE=lstm \
		--wrap="module load Python/Anaconda_v10.2019 && source deactivate && source activate my_py_env && mkdir -p $(INPUT_DIR)/logs && python mark_all.py --input_dir $(INPUT_DIR) --model_type lstm --allow_mixed --device cuda 2>&1 | tee $(INPUT_DIR)/logs/mark_all_lstm_\$$SLURM_JOB_ID.log"
	@echo "Запуск анализа Reformer..."
	@sbatch --constraint=$(CONSTRAINT) --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t 10:00:00 -A $(PROJECT) \
		--job-name="mark_all_reformer_$(shell basename $(INPUT_DIR))" \
		--output=$(OUTPUT_DIR)/out/mark_all_reformer_%j.out \
		--error=$(OUTPUT_DIR)/out/mark_all_reformer_%j.err \
		--export=ALL,INPUT_DIR=$(INPUT_DIR),MODEL_TYPE=reformer \
		--wrap="module load Python/Anaconda_v10.2019 && source deactivate && source activate my_py_env && mkdir -p $(INPUT_DIR)/logs && python mark_all.py --input_dir $(INPUT_DIR) --model_type reformer --allow_mixed --device cuda 2>&1 | tee $(INPUT_DIR)/logs/mark_all_reformer_\$$SLURM_JOB_ID.log"
	@echo "Все три задачи отправлены в очередь SLURM"
	@echo "Логи будут сохранены в $(INPUT_DIR)/logs/"


# Справка
help:
	@echo "Доступные команды:"
	@echo "  make all          - Запустить все эксперименты"
	@echo "  make gru-run      - Запустить все эксперименты GRU"
	@echo "  make lstm-run     - Запустить все эксперименты LSTM"
	@echo "  make reformer-run - Запустить все эксперименты Reformer"
	@echo "  make run MODEL=X DATASET=Y LOSS=Z - Запустить конкретный эксперимент"
	@echo "  make mark-all INPUT_DIR=/path/to/models - Запустить анализ моделей через mark_all.py"
	@echo ""
	@echo "  make clean        - Очистить временные файлы"
	@echo "  make clean-logs   - Очистить только логи в папке $(OUTPUT_DIR)/out/"
	@echo "  make help         - Показать эту справку"
	@echo ""
	@echo "Параметры:"
	@echo "  MODELS: $(MODELS)"
	@echo "  DATASETS: $(DATASETS)"
	@echo "  LOSSES: $(LOSSES)"
	@echo "  OUTPUT_DIR: $(OUTPUT_DIR) (по умолчанию: results)"
	@echo "  SEQ_LEN: $(SEQ_LEN) (по умолчанию: 3000000)"
	@echo ""
	@echo "Примеры:"
	@echo "  make run MODEL=gru DATASET=const LOSS=ce OUTPUT_DIR=my_experiments"
	@echo "  make run MODEL=lstm DATASET=hl LOSS=kl"
	@echo "  make all OUTPUT_DIR=my_experiments"
	@echo "  make gru-run OUTPUT_DIR=test_results"
	@echo "  make run MODEL=gru DATASET=const LOSS=ce SEQ_LEN=50000"
	@echo "  make mark-all INPUT_DIR=test_all_v1"


