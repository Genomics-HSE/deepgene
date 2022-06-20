GPU = 1
CPU = 8
T = 600

config = configs/reformer.gin

hse-run:
	echo "#!/bin/bash" > run.sh;
	echo "srun python main.py --config=$(config) train" >> run.sh;
	sbatch --constraint="type_a|type_b|type_c|type_d" --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t $(T) run.sh;
	rm run.sh


hse-run-test:
	echo "#!/bin/bash" > run.sh;
	echo "srun python test.py" >> run.sh;
	sbatch --constraint="type_a|type_b|type_c|type_d" --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t $(T) run.sh;
	rm run.sh


ex = ""
print:
	echo $(ex)