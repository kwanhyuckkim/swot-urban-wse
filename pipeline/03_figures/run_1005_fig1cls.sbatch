#!/bin/bash
#SBATCH --job-name=tgrs_fig1cls
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --exclude=ceewater-cpu006
#SBATCH --output=fig1cls_1005_%j.out
set -uo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
$PY -u 63_fig1_class_counts.py || echo "FAILED fig1cls"
