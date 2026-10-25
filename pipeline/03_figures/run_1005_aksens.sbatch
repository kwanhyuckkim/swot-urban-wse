#!/bin/bash
#SBATCH --job-name=tgrs_aksens
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=ceewater-cpu006
#SBATCH --output=aksens_1005_%j.out
set -uo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
echo "host $(hostname) start $(date)"
$PY -u 60_alaska_sensitivity.py || echo "FAILED aksens"
echo "end $(date)"
