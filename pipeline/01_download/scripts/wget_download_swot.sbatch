#!/bin/bash
#SBATCH -J swot-download
#SBATCH -p ceewater_kandread-cpu
#SBATCH -n 1
#SBATCH -c 24         
#SBATCH --mem=96G          
#SBATCH -t 48:00:00
#SBATCH -o slurm-%j.out

module load conda/latest
conda activate /work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban

echo "Start: $(date)"
start_time=$(date +%s)

# Base directories
URL_DIR="/home/kwanhyuckkim_umass_edu/_4_swot-urban/swot-urban/earthaccess-pixel/PIXC_url_all"
SAVE_DIR="/project/pi_kandread_umass_edu/swot-urban/0_data_collection/SWOT_PIXC"

# Loop through all URL files
for url_file in ${URL_DIR}/swot_PIXC_vD_fctimp_*_urls.txt; do
    # Extract fctimp value: swot_PIXC_vD_fctimp_0.17_urls.txt -> 0.17
    fctimp=$(basename "$url_file" | sed -E 's/swot_PIXC_vD_fctimp_(.*)_urls.txt/\1/')
    
    # Create output directory
    out_dir="${SAVE_DIR}/fctimp_${fctimp}"
    mkdir -p "$out_dir"
    
    echo "Downloading fctimp ${fctimp} to ${out_dir}..."
    
    # Download (with -nc to skip existing files)
    cat "$url_file" | \
    xargs -n 1 -P 24 wget -P "$out_dir" \
         --load-cookies ~/.urs_cookies -q -nc
    
    echo "  Completed fctimp ${fctimp}"
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "End: $(date)"
echo "Total time: $((elapsed / 3600))h $((elapsed % 3600 / 60))m $((elapsed % 60))s"