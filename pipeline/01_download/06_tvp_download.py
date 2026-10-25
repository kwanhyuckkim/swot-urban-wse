import earthaccess
import h5py
import pandas as pd
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

url_dir = Path("PIXC_url_all")
save_dir = Path("/work/pi_kandread_umass_edu/swot-urban/6_pixc_tvp/SWOT_TVP_Data")

def read_single_url(url):
    """각 프로세스에서 독립적으로 실행"""
    try:
        files = earthaccess.open([url])
        with h5py.File(files[0], 'r') as f:
            tvp = f['tvp']
            df = pd.DataFrame({
                'time': tvp['time'][:],
                'latitude': tvp['latitude'][:],
                'longitude': tvp['longitude'][:],
                'altitude': tvp['altitude'][:]
            })
            df['source_file'] = url.split('/')[-1]
            return df
    except:
        return None

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    earthaccess.login()
    save_dir.mkdir(exist_ok=True)
    
    url_files = sorted([f for f in url_dir.glob("*_urls.txt")])
    print(f"총 {len(url_files)}개 Reach, workers=48")
    
    total_start = time.time()
    
    for i, url_txt in enumerate(url_files, 1):
        reach_id = url_txt.stem.split('_')[-2]
        out_path = save_dir / f"tvp_reach_{reach_id}.parquet"
        
        if out_path.exists():
            print(f"[{i}/{len(url_files)}] Reach {reach_id}: Skip")
            continue
        
        with open(url_txt, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        if not urls:
            continue
        
        print(f"[{i}/{len(url_files)}] Reach {reach_id}: {len(urls)}개 파일...", end=" ")
        start = time.time()
        
        with ProcessPoolExecutor(max_workers=48) as ex:
            results = list(ex.map(read_single_url, urls))
        
        dfs = [df for df in results if df is not None]
        
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            final_df.to_parquet(out_path)
            print(f"완료 ({len(final_df):,} rows, {time.time()-start:.1f}초)")
        else:
            print("데이터 없음")
    
    print(f"\n전체 완료! 총 {(time.time()-total_start)/3600:.1f}시간")