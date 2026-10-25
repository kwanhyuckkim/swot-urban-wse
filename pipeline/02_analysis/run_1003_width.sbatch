#!/bin/bash
#SBATCH --job-name=tgrs_width
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=width_1003_%j.out
#SBATCH --exclude=ceewater-cpu006
set -uo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
echo "host $(hostname) start $(date)"
$PY -u 33_sword_width_check.py "${SLURM_SUBMIT_DIR}" || echo "FAILED width"
echo "end $(date)"
