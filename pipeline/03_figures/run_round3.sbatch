#!/bin/bash
#SBATCH --job-name=swot_r3fig
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=r3fig_%A.out
set -euo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
$PY -u 31_figS2_print.py
$PY -u 30_fig01_fig05_print.py fig05
$PY -u fig_print_final.py fig03 fig07 fig09
echo "end $(date)"
