#!/bin/bash
#SBATCH --job-name=tgrs_fig02
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=ceewater-cpu006
#SBATCH --output=fig02_1005_%j.out
set -uo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplconfig" XDG_CACHE_HOME="${SLURM_SUBMIT_DIR}/.cache" STATES_ZIP="${SLURM_SUBMIT_DIR}/cb_2021_us_state_500k.zip"
echo "host $(hostname) start $(date)"
$PY -u fig02_study_map_epsg4269_1005.py || echo "FAILED fig02"
echo "end $(date)"
