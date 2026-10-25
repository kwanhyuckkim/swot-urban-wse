#!/bin/bash
#SBATCH --job-name=tgrs_vcD2
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --exclude=ceewater-cpu006,ceewater-cpu001
#SBATCH --output=vcD2_1004_%j.out
set -uo pipefail
PY=/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python3.14
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplconfig" XDG_CACHE_HOME="${SLURM_SUBMIT_DIR}/.cache" FIG_OUT="${SLURM_SUBMIT_DIR}"
echo "host $(hostname) start $(date)"
$PY -u 45_vc_D_fixedfolds.py || echo "FAILED cgridcv"
echo "end $(date)"
$PY - <<'PYEOF'
import pandas as pd
n = pd.read_parquet("/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/reviewer_checks/layovr_val_node.parquet", columns=["node_q_b"])
q = n.node_q_b.dropna().astype("int64")
print(f"full node pull rows {len(n):,}; node_q_b present {len(q):,}; bit 7 (128) set {int(((q & 128) > 0).sum()):,}")
PYEOF
echo "end2 $(date)"
