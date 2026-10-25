#!/bin/bash
#SBATCH --job-name=coh_stream
#SBATCH --partition=cpu-preempt
#SBATCH --array=0-39
#SBATCH --cpus-per-task=6
#SBATCH --mem=42G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=cohstream_%A_%a.out
set -euo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
echo "host $(hostname) shard ${SLURM_ARRAY_TASK_ID} start $(date)"
$PY -u 10_pixc_stream_all_reaches.py extract \
    --shard "${SLURM_ARRAY_TASK_ID}" --nshards 40 --workers 4
echo "end $(date)"
