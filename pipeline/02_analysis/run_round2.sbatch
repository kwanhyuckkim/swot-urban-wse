#!/bin/bash
#SBATCH --job-name=swot_round2
#SBATCH --partition=cpu-preempt
#SBATCH --array=0-6
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=round2_%A_%a.out
set -euo pipefail
TASKS=(pops flags width sens huc boot cv)
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
echo "host $(hostname) task ${TASKS[$SLURM_ARRAY_TASK_ID]} start $(date)"
$PY -u 07_round2_analyses.py "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "end $(date)"
