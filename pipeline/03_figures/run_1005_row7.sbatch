#!/bin/bash
#SBATCH --job-name=tgrs_row7
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --exclude=ceewater-cpu006
#SBATCH --output=row7_1005_%j.out
set -uo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
$PY -u 67_table2_row7_classes.py || echo "FAILED row7"
