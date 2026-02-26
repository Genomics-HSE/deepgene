#!/bin/bash
#SBATCH -c 1
#SBATCH --output=/home/evkhomutov/deepgene_final/deepgene_new_10_v11/psmc_checker/runs/1000g/slurm.%j.out

set -eo pipefail
cd "/home/evkhomutov/deepgene_final/deepgene_new_10_v11/psmc_checker"
if [[ -n "module load Python/Anaconda_v10.2019 && source activate my_py_env" ]]; then eval "module load Python/Anaconda_v10.2019 && source activate my_py_env"; fi
srun -n 1 -A proj_1669 "/home/evkhomutov/deepgene_final/deepgene_new_10_v11/psmc_checker/psmc" -N25 -t15 -r5 -p 4+25*2+4+6 -o "/home/evkhomutov/deepgene_final/deepgene_new_10_v11/psmc_checker/runs/1000g/HG00096_chr22.psmc" "/home/evkhomutov/deepgene_final/deepgene_new_10_v11/psmc_checker/runs/1000g/HG00096_chr22.psmcfa"
