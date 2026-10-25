#!/bin/bash
#SBATCH -J swot-download
#SBATCH -p ceewater_kandread-cpu        # CPU만 필요 (GPU 불필요)
#SBATCH -n 1
#SBATCH -c 8                    # ThreadPool은 CPU 많이 안 씀
#SBATCH --mem=256G               # 메모리도 줄여도 됨
#SBATCH -t 72:00:00
#SBATCH -o swot-reach_download-%j.out
#SBATCH -e swot-reach_download-%j.err

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

module load conda/latest
conda activate /work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban

python swot_hydrocron_download_ver2.py

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="