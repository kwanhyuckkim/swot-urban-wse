#!/bin/bash
#SBATCH --job-name=layovr_node
#SBATCH --partition=cpu-preempt
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=layovr_node_%A_%a.out
# Node-level layovr_val pull, sharded so the Hydrocron requests run concurrently
# across array tasks. Merge afterwards with: python 07_hydrocron_layovr_val.py merge
set -euo pipefail
PY=/home/kwanhyuckkim_umass_edu/.conda/envs/hyena/bin/python
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
echo "host $(hostname) shard ${SLURM_ARRAY_TASK_ID} start $(date)"
$PY -u 07_hydrocron_layovr_val.py node \
    --shard "${SLURM_ARRAY_TASK_ID}" --nshards "${SLURM_ARRAY_TASK_COUNT}" --workers 12
echo "end $(date)"
