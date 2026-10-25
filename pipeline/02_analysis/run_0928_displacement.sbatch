#!/bin/bash
#SBATCH --job-name=tgrs_disp_inc
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=disp_0928_%j.out
# 2026-09-28: layover displacement with the incidence angle (24_displacement_incidence.py),
# the incidence version of Supplementary Fig. S5 (36b_...), and the pre-submission
# verification pass (23_number_checks.py, which prints the population n's).
set -uo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplconfig" XDG_CACHE_HOME="${SLURM_SUBMIT_DIR}/.cache"
echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK}  start $(date)"
echo "=== 24_displacement_incidence"
$PY -u 24_displacement_incidence.py displacement_incidence.json || echo "FAILED 24_displacement_incidence"
echo "=== 36b_figS5_displacement_incidence"
$PY -u 36b_figS5_displacement_incidence.py . || echo "FAILED 36b_figS5"
echo "=== 23_number_checks"
$PY -u 23_number_checks.py || echo "FAILED 23_number_checks"
echo "end $(date)"
