#!/bin/bash
#SBATCH -J swot-download-aria2
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
    fctimp=$(basename "$url_file" | sed -E 's/swot_PIXC_vD_fctimp_(.*)_urls.txt/\1/')
    out_dir="${SAVE_DIR}/fctimp_${fctimp}"
    
    mkdir -p "$out_dir"
    echo "Downloading fctimp ${fctimp} to ${out_dir}..."
    
    # aria2c 옵션 설명:
    # -i: 입력 파일 (URL 리스트)
    # -d: 저장할 디렉토리
    # -j 24: 동시에 24개 파일 다운로드 (병렬성)
    # -x 4: 파일 하나당 최대 4개 연결 (속도 향상 핵심)
    # -c: 중단된 파일 이어받기 (wget의 -nc보다 강력함)
    # --load-cookies: NASA 인증 쿠키 사용
    
    aria2c -i "$url_file" \
           -d "$out_dir" \
           -j 24 \
           -x 4 \
           -s 4 \
           -c \
           --load-cookies ~/.urs_cookies \
           --console-log-level=warn \
           --summary-interval=0
           
    echo "  Completed fctimp ${fctimp}"
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "End: $(date)"
echo "Total time: $((elapsed / 3600))h $((elapsed % 3600 / 60))m $((elapsed % 60))s"