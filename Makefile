GPU = 1
CPU = 2
T = 600

config = configs/reformer.gin

hse-run:
	echo "#!/bin/bash" > run.sh;
	echo "srun python main.py --config=$(config) train" >> run.sh;
	sbatch --signal=USR1@50 --gpus=$(GPU) -c $(CPU) -t $(T) run.sh;
	rm run.sh

print:
	echo $(config_file)
