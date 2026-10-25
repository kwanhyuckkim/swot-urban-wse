#!/bin/bash
#SBATCH --job-name=tgrs_vc6
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=vc6_0928_%j.out
set -uo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplconfig" XDG_CACHE_HOME="${SLURM_SUBMIT_DIR}/.cache"
echo "host $(hostname) start $(date)"
$PY -u 26_vc_six_decimals.py || echo "FAILED 25_pixc_population_check"
echo "end $(date)"
