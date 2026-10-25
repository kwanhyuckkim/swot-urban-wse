#!/bin/bash
#SBATCH --job-name=tgrs_pert
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --exclude=ceewater-cpu006
#SBATCH --output=pert_1005_%j.out
set -uo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplconfig" XDG_CACHE_HOME="${SLURM_SUBMIT_DIR}/.cache" FIG_OUT="${SLURM_SUBMIT_DIR}"
echo "host $(hostname) start $(date)"
$PY -u 56_height_perturbation.py || echo "FAILED cgridcv"
echo "end $(date)"
