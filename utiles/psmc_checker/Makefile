# Makefile for: generate .psmcfa -> submit PSMC to SLURM -> store .psmc next to .psmcfa
#
# Usage examples:
#   make -C psmc_checker psmc DEM=const OUT_DIR=runs NAME=run1
#   make -C psmc_checker run_all -j 3 OUT_DIR=runs NAME_PREFIX=exp1
#
# Notes:
# - This Makefile follows the style of `psmc_checker/command_ms6`:
#   sbatch script + inside it runs: srun -n 1 -A proj_1669 ./psmc ...
# - For local testing without SLURM, use SUBMIT=0 to only generate the sbatch script.

SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c

PSMC_CHECKER_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# ----- Defaults (override via `make VAR=value`) -----

DEM ?= const
OUT_DIR ?= runs
NAME ?= run
NAME_PREFIX ?= exp

SEQ_LEN ?= 3000000
BIN_SIZE ?= 100
NUM_REPS ?= 100
INITIAL_SIZE ?= 10000

# Generator params (per-base). The generator scales by BIN_SIZE internally.
MU ?=
RHO ?=

GEN ?= $(PSMC_CHECKER_DIR)/generate_psmcfa_deepgene.py
PYTHON ?= python

# Python environment setup (cluster defaults, override if needed):
#   make ... ENV_SETUP="module load ... && source activate ..."
# For local runs without modules/conda, you can disable it:
#   make ... ENV_SETUP=
ENV_SETUP ?= module load Python/Anaconda_v10.2019 && source activate my_py_env

PSMC ?= $(PSMC_CHECKER_DIR)/psmc
PSMC_ARGS ?= -N25 -t15 -r5 -p 4+25*2+4+6

# SLURM defaults (like `command_ms6`)
SBATCH ?= sbatch
SBATCH_ARGS ?= -c 1
SRUN ?= srun
SRUN_ARGS ?= -n 1 -A proj_1669

# If SUBMIT=0: only generate the sbatch script (no submission).
SUBMIT ?= 1

# Demography mapping (deepgene function names)
DEMOGRAPHY_const := get_const_demographcs
DEMOGRAPHY_hl := get_humanlike
DEMOGRAPHY_zigzag := get_zigzag

DEMOGRAPHY := $(DEMOGRAPHY_$(DEM))

ifeq ($(strip $(DEMOGRAPHY)),)
$(error Unknown DEM='$(DEM)'. Supported: const hl zigzag)
endif

DEM_DIR := $(OUT_DIR)/$(DEM)
PSMCFA := $(DEM_DIR)/$(NAME).psmcfa
PSMCOUT := $(DEM_DIR)/$(NAME).psmc
SBATCH_SCRIPT := $(DEM_DIR)/run_psmc.sbatch
JOBID_FILE := $(DEM_DIR)/jobid.txt

.PHONY: help psmcfa psmc run_all psmc_const psmc_hl psmc_zigzag psmc_1000g clean

help:
	@echo "Targets:"
	@echo "  psmcfa   - generate $(PSMCFA)"
	@echo "  psmc     - generate .psmcfa and submit PSMC via SLURM (or SUBMIT=0 to only write script)"
	@echo "  run_all  - submit for DEM in: const hl zigzag (parallel with -j)"
	@echo "  psmc_1000g - submit PSMC via SLURM for an existing real-data .psmcfa (SUBMIT=0 to only write script)"
	@echo ""
	@echo "Common vars:"
	@echo "  DEM={const|hl|zigzag} OUT_DIR=... NAME=... SEQ_LEN=... BIN_SIZE=... NUM_REPS=... INITIAL_SIZE=..."
	@echo "  PSMC_ARGS='...' SBATCH_ARGS='...' SRUN_ARGS='...'"
	@echo ""
	@echo "Real-data vars (for psmc_1000g):"
	@echo "  REAL_PSMC (psmc binary, default: $(PSMC))"
	@echo "  REAL_PSMCFA (default: $(PSMC_CHECKER_DIR)/runs/1000g/HG00096_chr22.psmcfa)"
	@echo "  REAL_PSMCOUT (default: same path with .psmc extension)"
	@echo "  SUBMIT=0 to only generate the sbatch script (no submission)"

$(DEM_DIR):
	@mkdir -p "$(DEM_DIR)"

psmcfa: $(PSMCFA)

$(PSMCFA): | $(DEM_DIR)
	@echo "[psmcfa] DEM=$(DEM) ($(DEMOGRAPHY)) -> $@"
	@bash -lc 'set -eo pipefail; \
		if [[ -n "$(ENV_SETUP)" ]]; then eval "$(ENV_SETUP)"; fi; \
		"$(PYTHON)" "$(GEN)" \
			--demography "$(DEMOGRAPHY)" \
			--initial-size "$(INITIAL_SIZE)" \
			--num-replicates "$(NUM_REPS)" \
			--sequence-length "$(SEQ_LEN)" \
			--bin-size "$(BIN_SIZE)" \
			$(if $(strip $(MU)),--mu $(MU),) \
			$(if $(strip $(RHO)),--rho $(RHO),) \
			--out "$(PSMCFA)" \
			--overwrite'

psmc: $(PSMCFA) $(SBATCH_SCRIPT)
	@echo "[psmc] DEM=$(DEM) -> $(PSMCOUT)"
	@if [[ "$(SUBMIT)" == "0" ]]; then \
		echo "[psmc] SUBMIT=0 -> sbatch script generated at $(SBATCH_SCRIPT)"; \
	else \
		jid="$$( "$(SBATCH)" $(SBATCH_ARGS) "$(SBATCH_SCRIPT)" | awk '{print $$4}' )"; \
		echo "$$jid" > "$(JOBID_FILE)"; \
		echo "[psmc] submitted job $$jid (saved to $(JOBID_FILE))"; \
	fi

$(SBATCH_SCRIPT): $(PSMCFA) | $(DEM_DIR)
	@echo "[sbatch] writing $@"
	@{ \
		echo '#!/bin/bash'; \
		echo '#SBATCH -c 1'; \
		echo '#SBATCH --output=$(abspath $(DEM_DIR))/slurm.%j.out'; \
		echo ''; \
		echo 'set -eo pipefail'; \
		echo 'cd "$(PSMC_CHECKER_DIR)"'; \
		echo '$(SRUN) $(SRUN_ARGS) "$(PSMC)" $(PSMC_ARGS) -o "$(abspath $(PSMCOUT))" "$(abspath $(PSMCFA))"'; \
	} > "$(SBATCH_SCRIPT)"
	@chmod +x "$(SBATCH_SCRIPT)"

run_all: psmc_const psmc_hl psmc_zigzag

psmc_const:
	@$(MAKE) psmc DEM=const OUT_DIR="$(OUT_DIR)" NAME="$(NAME_PREFIX)_const" SEQ_LEN="$(SEQ_LEN)" BIN_SIZE="$(BIN_SIZE)" NUM_REPS="$(NUM_REPS)" INITIAL_SIZE="$(INITIAL_SIZE)" SUBMIT="$(SUBMIT)"

psmc_hl:
	@$(MAKE) psmc DEM=hl OUT_DIR="$(OUT_DIR)" NAME="$(NAME_PREFIX)_hl" SEQ_LEN="$(SEQ_LEN)" BIN_SIZE="$(BIN_SIZE)" NUM_REPS="$(NUM_REPS)" INITIAL_SIZE="$(INITIAL_SIZE)" SUBMIT="$(SUBMIT)"

psmc_zigzag:
	@$(MAKE) psmc DEM=zigzag OUT_DIR="$(OUT_DIR)" NAME="$(NAME_PREFIX)_zigzag" SEQ_LEN="$(SEQ_LEN)" BIN_SIZE="$(BIN_SIZE)" NUM_REPS="$(NUM_REPS)" INITIAL_SIZE="$(INITIAL_SIZE)" SUBMIT="$(SUBMIT)"

############################
# Real data (1000 Genomes)
############################

# NOTE:
# - The bundled `psmc_checker/psmc` binary is Linux ELF and won't run on macOS.
# - On Linux/cluster, by default we use the bundled binary `psmc_checker/psmc`.
# - If you have `psmc` installed in PATH (e.g. via conda/bioconda), you can override: REAL_PSMC=psmc
REAL_PSMC ?= $(PSMC)
REAL_1000G_DIR ?= $(PSMC_CHECKER_DIR)/runs/1000g
REAL_PSMCFA ?= $(REAL_1000G_DIR)/HG00096_chr22.psmcfa
REAL_PSMCOUT ?= $(patsubst %.psmcfa,%.psmc,$(REAL_PSMCFA))
REAL_SBATCH_SCRIPT ?= $(REAL_1000G_DIR)/run_psmc_1000g.sbatch
REAL_JOBID_FILE ?= $(REAL_1000G_DIR)/jobid.txt

$(REAL_1000G_DIR):
	@mkdir -p "$(REAL_1000G_DIR)"

psmc_1000g: $(REAL_SBATCH_SCRIPT)
	@echo "[psmc_1000g] input=$(REAL_PSMCFA)"
	@echo "[psmc_1000g] output=$(REAL_PSMCOUT)"
	@if [[ ! -f "$(REAL_PSMCFA)" ]]; then \
		echo "ERROR: REAL_PSMCFA not found: $(REAL_PSMCFA)"; \
		echo "Hint: copy a .psmcfa into $(REAL_1000G_DIR) or pass REAL_PSMCFA=/path/to/file.psmcfa"; \
		exit 2; \
	fi
	@if [[ "$(SUBMIT)" == "0" ]]; then \
		echo "[psmc_1000g] SUBMIT=0 -> sbatch script generated at $(REAL_SBATCH_SCRIPT)"; \
	else \
		jid="$$( "$(SBATCH)" $(SBATCH_ARGS) "$(REAL_SBATCH_SCRIPT)" | awk '{print $$4}' )"; \
		echo "$$jid" > "$(REAL_JOBID_FILE)"; \
		echo "[psmc_1000g] submitted job $$jid (saved to $(REAL_JOBID_FILE))"; \
	fi

$(REAL_SBATCH_SCRIPT): | $(REAL_1000G_DIR)
	@echo "[sbatch] writing $@"
	@{ \
		echo '#!/bin/bash'; \
		echo '#SBATCH -c 1'; \
		echo '#SBATCH --output=$(abspath $(REAL_1000G_DIR))/slurm.%j.out'; \
		echo ''; \
		echo 'set -eo pipefail'; \
		echo 'cd "$(PSMC_CHECKER_DIR)"'; \
		echo 'if [[ -n "$(ENV_SETUP)" ]]; then eval "$(ENV_SETUP)"; fi'; \
		echo '$(SRUN) $(SRUN_ARGS) "$(REAL_PSMC)" $(PSMC_ARGS) -o "$(abspath $(REAL_PSMCOUT))" "$(abspath $(REAL_PSMCFA))"'; \
	} > "$(REAL_SBATCH_SCRIPT)"
	@chmod +x "$(REAL_SBATCH_SCRIPT)"

clean:
	@rm -rf "$(OUT_DIR)"

