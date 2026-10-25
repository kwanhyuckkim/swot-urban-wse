#!/bin/bash
# Submit the three dependent stages of the Version-D control chain.
set -euo pipefail
cd "$(dirname "$0")"
J1=$(sbatch --parsable --array=0-7 --export=ALL,CTRL_STAGE=hydrocron run_control_versionD.sbatch)
J2=$(sbatch --parsable --array=0-7 --dependency=afterok:${J1} --export=ALL,CTRL_STAGE=usgs run_control_versionD.sbatch)
J3=$(sbatch --parsable --dependency=afterok:${J2} --export=ALL,CTRL_STAGE=analyze run_control_versionD.sbatch)
echo "submitted: hydrocron=${J1} usgs=${J2} analyze=${J3}"
