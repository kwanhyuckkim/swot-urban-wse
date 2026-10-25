#!/bin/bash
#SBATCH --job-name=swot_coh
#SBATCH --partition=cpu-preempt
#SBATCH --array=0-11
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH --output=coh_%A_%a.out
set -euo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
echo "host $(hostname) site ${SLURM_ARRAY_TASK_ID} start $(date)"
$PY -u 16_coherence_check.py extract --site-index "${SLURM_ARRAY_TASK_ID}"
echo "end $(date)"
