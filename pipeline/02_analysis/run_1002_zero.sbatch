#!/bin/bash
#SBATCH --job-name=tgrs_zero
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=zero_1002_%j.out
set -uo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplconfig" XDG_CACHE_HOME="${SLURM_SUBMIT_DIR}/.cache"
echo "host $(hostname) start $(date)"
$PY -u 28_far_isa_zero.py || echo "FAILED 25_pixc_population_check"
echo "end $(date)"
