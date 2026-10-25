#!/bin/bash
#SBATCH --job-name=boot_fast
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=boot_fast_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
echo "host $(hostname) cpus ${SLURM_CPUS_PER_TASK} start $(date)"
/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14 -u boot_fast.py 300
echo "end $(date)"
