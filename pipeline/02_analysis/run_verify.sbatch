#!/bin/bash
#SBATCH --job-name=swot_verify
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=verify_%A.out
set -euo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
RC=/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/reviewer_checks
if ! ls ${RC}/layovr_val_reach*.parquet >/dev/null 2>&1; then
  echo "reach layovr_val missing -> re-pulling via 07"
  $PY -u ../01_download/07_hydrocron_layovr_val.py reach || true
fi
$PY -u 11_verify_response_numbers.py
echo "end $(date)"
