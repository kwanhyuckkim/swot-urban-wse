#!/bin/bash
#SBATCH --job-name=tgrs_fig03
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=fig03_1004_%j.out
#SBATCH --exclude=ceewater-cpu006
set -uo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplconfig" XDG_CACHE_HOME="${SLURM_SUBMIT_DIR}/.cache" FIG_OUT="${SLURM_SUBMIT_DIR}"
echo "host $(hostname) start $(date)"
$PY -u 38_fig03_map_only_print.py || echo "FAILED fig03"
echo "end $(date)"
