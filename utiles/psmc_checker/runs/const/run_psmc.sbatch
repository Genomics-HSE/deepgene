#!/bin/bash
#SBATCH -c 1
#SBATCH --output=/home/evkhomutov/deepgene_final/deepgene_new_10_v11/psmc_checker/runs/const/slurm.%j.out

set -eo pipefail
cd "/home/evkhomutov/deepgene_final/deepgene_new_10_v11/psmc_checker"
srun -n 1 -A proj_1669 "/home/evkhomutov/deepgene_final/deepgene_new_10_v11/psmc_checker/psmc" -N25 -t15 -r5 -p 4+25*2+4+6 -o "/home/evkhomutov/deepgene_final/deepgene_new_10_v11/psmc_checker/runs/const/exp1_const.psmc" "/home/evkhomutov/deepgene_final/deepgene_new_10_v11/psmc_checker/runs/const/exp1_const.psmcfa"
