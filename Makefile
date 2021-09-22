GPU = 1
CPU = 2
T = 600

config = configs/reformer.gin

hse-run:
	echo "#!/bin/bash" > run.sh;
	echo "python main.py --config=$(config) train" >> run.sh;
	sbatch --gpus=$(GPU) -c $(CPU) -t $(T) run.sh;
	rm run.sh

print:
	echo $(config_file)
