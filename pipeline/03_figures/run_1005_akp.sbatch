#!/bin/bash
#SBATCH --job-name=tgrs_akp
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --exclude=ceewater-cpu006
#SBATCH --output=akp_1005_%j.out
set -uo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
$PY -u 61c_paired_1723.py || echo "FAILED akp"
