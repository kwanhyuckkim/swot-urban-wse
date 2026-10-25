#!/bin/bash
#SBATCH --job-name=swot_pfig
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=pfig_%A.out
set -euo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
$PY -u fig_print_final.py
echo "end $(date)"
