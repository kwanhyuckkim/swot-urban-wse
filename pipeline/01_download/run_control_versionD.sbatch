#!/bin/bash
# Three-stage Version-D control chain. Submit with: bash submit_control_chain.sh
# Stage selected by CTRL_STAGE env: list+hydrocron shards / usgs shards / merge+analysis
#SBATCH --job-name=swot_ctrlD
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=ctrlD_%x_%A_%a.out
set -euo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
echo "host $(hostname) stage ${CTRL_STAGE} task ${SLURM_ARRAY_TASK_ID:-0} start $(date)"
case "${CTRL_STAGE}" in
  hydrocron)
    if [ "${SLURM_ARRAY_TASK_ID}" = "0" ] && [ ! -f /project/pi_kandread_umass_edu/swot-urban/3_analysis_results/control_versionD/control_pairs.csv ]; then
      $PY -u 08_control_reaches_pull.py list
    fi
    until [ -f /project/pi_kandread_umass_edu/swot-urban/3_analysis_results/control_versionD/control_pairs.csv ]; do sleep 5; done
    $PY -u 08_control_reaches_pull.py hydrocron --shard "${SLURM_ARRAY_TASK_ID}" --nshards 8 --workers 12
    ;;
  usgs)
    if [ "${SLURM_ARRAY_TASK_ID}" = "0" ] && [ ! -f /project/pi_kandread_umass_edu/swot-urban/3_analysis_results/control_versionD/swot_reach_D.parquet ]; then
      $PY -u 08_control_reaches_pull.py merge --nshards 8
    fi
    until [ -f /project/pi_kandread_umass_edu/swot-urban/3_analysis_results/control_versionD/swot_reach_D.parquet ]; do sleep 10; done
    $PY -u 09_control_usgs_iv.py --shard "${SLURM_ARRAY_TASK_ID}" --nshards 8 --workers 8
    ;;
  analyze)
    $PY -u 09_control_usgs_iv.py --merge
    $PY -u ../02_analysis/10_control_versionD.py
    ;;
esac
echo "end $(date)"
