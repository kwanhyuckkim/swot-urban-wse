#!/bin/bash
#SBATCH --job-name=swot_reviewer_checks
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=reviewer_checks_%j.out
# Reviewer-driven product checks for TGRS-2026-02661.
# Runs the pixel-level extraction across every matched observation in parallel,
# then the partial-correlation separation. Requires an environment with xarray,
# geopandas, shapely 2, rasterio and pyarrow.
set -euo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
# under sbatch, $0 points at the spool copy, so prefer the submission directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK}  start $(date)"
$PY -u 06_reviewer_product_checks.py crosslevel --workers "${SLURM_CPUS_PER_TASK}"
$PY -u 06_reviewer_product_checks.py partial
echo "end $(date)"
