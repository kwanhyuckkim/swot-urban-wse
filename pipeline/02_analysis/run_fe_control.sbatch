#!/bin/bash
#SBATCH --job-name=swot_fe_ctrl
#SBATCH --partition=cpu-preempt
#SBATCH --array=0-1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=fe_ctrl_%A_%a.out
set -euo pipefail
SCRIPTS=(08_reach_fixed_effects.py 09_nonurban_control.py)
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
echo "host $(hostname) script ${SCRIPTS[$SLURM_ARRAY_TASK_ID]} start $(date)"
$PY -u "${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"
echo "end $(date)"
