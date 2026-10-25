#!/bin/bash
#SBATCH -J swot-nearfar
#SBATCH -p gpu-preempt #ceewater_kandread-cpu
#SBATCH -n 1
#SBATCH -c 48
#SBATCH -G 1
#SBATCH --mem=180G
#SBATCH -t 48:00:00
#SBATCH -o nearfarg-%j.out
#SBATCH -e nearfarg-%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"

module load conda/latest
conda activate /work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban

python 02_nearfar_pipeline_byreach.py

echo "Job finished: $(date)"
echo "=========================================="