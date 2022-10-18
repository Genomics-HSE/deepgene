GPU = 1
CPU = 8
T = 600

model = configs/min-config-gru/model.gin
data = configs/min-config-gru/data.gin
train = configs/min-config-gru/train.gin
gin_param = ""

hse-run:
	echo "#!/bin/bash" > run.sh;
	echo "srun python main.py --train=$(train) --model=$(model) --data=$(data) --gin_param=$(gin_param) fit" >> run.sh;
	sbatch --constraint="type_a|type_b|type_c|type_d" --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t $(T) run.sh;
	rm run.sh


hse-run-test:
	echo "#!/bin/bash" > run.sh;
	echo "srun python test.py" >> run.sh;
	sbatch --constraint="type_a|type_b|type_c|type_d" --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t $(T) run.sh;
	rm run.sh


ex = "123"
print:
	echo $(ex)