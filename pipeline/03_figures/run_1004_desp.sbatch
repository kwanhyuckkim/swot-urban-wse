#!/bin/bash
#SBATCH --job-name=tgrs_desp
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --exclude=ceewater-cpu006
#SBATCH --output=desp_1004_%j.out
set -uo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplconfig" XDG_CACHE_HOME="${SLURM_SUBMIT_DIR}/.cache" FIG_OUT="${SLURM_SUBMIT_DIR}"
echo "host $(hostname) start $(date)"
$PY -u 47_desplaines_check.py || echo "FAILED cgridcv"
echo "end $(date)"
