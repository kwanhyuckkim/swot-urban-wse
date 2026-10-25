#!/bin/bash
#SBATCH --job-name=swot_ncoh
#SBATCH --partition=cpu-preempt
#SBATCH --array=0-11
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=ncoh_%A_%a.out
set -euo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
echo "host $(hostname) site ${SLURM_ARRAY_TASK_ID} start $(date)"
$PY -u 20_node_coherence.py extract --site-index "${SLURM_ARRAY_TASK_ID}"
echo "end $(date)"
