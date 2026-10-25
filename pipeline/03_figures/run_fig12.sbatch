#!/bin/bash
#SBATCH --job-name=swot_fig12
#SBATCH --partition=cpu-preempt
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=fig12_%A_%a.out
set -euo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
SCRIPTS=(fig01_print.py fig02_print_run.py)
$PY -u "${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"
echo "end $(date)"
