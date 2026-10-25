#!/bin/bash
#SBATCH -J swot-node
#SBATCH -p gpu-preempt   #ceewater_kandread-cpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=512G
#SBATCH -t 08:00:00
#SBATCH -G 1
#SBATCH -o swot-node5-%A_%a.out
#SBATCH -e swot-node5-%A_%a.err
#SBATCH --array=0-5

# SWOT Hydrocron Node Download - Parallel by Continent
# Submit: sbatch submit_node_download.sh

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "=========================================="


# Continent mapping
CONTINENTS=("AF" "AS" "EU" "NA" "OC" "SA")
CONTINENT=${CONTINENTS[$SLURM_ARRAY_TASK_ID]}

echo "Processing continent: $CONTINENT"
echo ""

# Activate environment
module load conda/latest
conda activate /work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban

# Run download script
python swot_hydrocron_download_node_ver3.py $CONTINENT

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
