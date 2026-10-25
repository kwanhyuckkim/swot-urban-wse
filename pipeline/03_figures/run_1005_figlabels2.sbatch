#!/bin/bash
#SBATCH --job-name=tgrs_figlab2
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=ceewater-cpu006
#SBATCH --output=figlabels2_1005_%j.out
set -uo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplconfig" XDG_CACHE_HOME="${SLURM_SUBMIT_DIR}/.cache" FIG_OUT="${SLURM_SUBMIT_DIR}"
echo "host $(hostname) start $(date)"
$PY -u 64_figS1_thresholds_1005.py || echo "FAILED S1"
$PY -u 66_fig09_ideal_label_1005.py || echo "FAILED fig09"
echo "end $(date)"
