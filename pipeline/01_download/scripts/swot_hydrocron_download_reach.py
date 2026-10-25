#!/usr/bin/env python
# swot_hydrocron_download.py

import pandas as pd
import geopandas as gpd
import requests
from io import StringIO
from pathlib import Path
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import gc
import pyarrow.parquet as pq
import pyarrow as pa


def load_sword_reaches(sword_path, continents=None):
    """Load and merge SWORD reaches shapefiles from specified or all continents."""
    
    sword_dir = Path(sword_path)
    all_continents = ['AF', 'AS', 'EU', 'NA', 'OC', 'SA']
    
    if continents is None:
        continents = all_continents
    
    reaches_files = []
    for continent in continents:
        pattern = str(sword_dir / continent / '*reaches*.shp')
        files = glob.glob(pattern)
        reaches_files.extend(files)
        print(f"{continent}: {len(files)} files found")
    
    if not reaches_files:
        raise FileNotFoundError(f"No reaches shapefiles found in {sword_path}")
    
    print(f"\nLoading {len(reaches_files)} total files...")
    
    reaches_list = [gpd.read_file(f) for f in sorted(reaches_files)]
    reaches = gpd.GeoDataFrame(
        pd.concat(reaches_list, ignore_index=True),
        crs=reaches_list[0].crs
    )
    
    # Free memory
    del reaches_list
    gc.collect()
    
    return reaches


def query_hydrocron(reach_id, query_url, start_time, end_time, fields, collection_name):
    """Query Hydrocron for reach-level time series data."""
    
    params = {
        "feature": "Reach",
        "feature_id": reach_id,
        "output": "csv",
        "start_time": start_time,
        "end_time": end_time,
        "fields": fields,
        "collection_name": collection_name
    }
    
    try:
        response = requests.get(query_url, params=params, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        if "results" in data and data["results"].get("csv"):
            df = pd.read_csv(StringIO(data["results"]["csv"]))
            return df
    except Exception as e:
        pass
    
    return None


def download_with_incremental_save(reach_ids, query_url, start_time, end_time, fields, 
                                   collection_name, output_dir, max_workers=100, 
                                   chunk_size=5000):
    """Download data and save incrementally to avoid memory issues."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 기존 chunk에서 이미 다운로드된 reach_id 확인
    existing_reaches = set()
    existing_chunks = sorted(output_dir.glob("chunk_*.parquet"))
    
    if existing_chunks:
        print(f"Found {len(existing_chunks)} existing chunks, checking downloaded reaches...")
        for chunk_file in tqdm(existing_chunks, desc="Scanning existing"):
            df = pd.read_parquet(chunk_file, columns=['reach_id'])
            existing_reaches.update(df['reach_id'].unique())
            del df
        gc.collect()
        
        # 다음 chunk 번호 설정
        last_chunk = existing_chunks[-1].stem  # "chunk_0042"
        chunk_num = int(last_chunk.split('_')[1]) + 1
        print(f"Already downloaded: {len(existing_reaches):,} reaches")
        print(f"Resuming from chunk {chunk_num}")
    else:
        chunk_num = 0
    
    # 이미 다운로드된 reach 제외
    reach_ids = [r for r in reach_ids if r not in existing_reaches]
    print(f"Remaining to download: {len(reach_ids):,} reaches")
    
    if not reach_ids:
        print("All reaches already downloaded!")
        return output_dir, chunk_num
    
    failed_count = 0
    success_count = 0
    chunk_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_reach = {
            executor.submit(
                query_hydrocron, reach_id, query_url, start_time, end_time, fields, collection_name
            ): reach_id 
            for reach_id in reach_ids
        }
        
        for future in tqdm(as_completed(future_to_reach), total=len(reach_ids), desc="Downloading"):
            result = future.result()
            if result is not None and len(result) > 0:
                chunk_results.append(result)
                success_count += 1
                
                # Save chunk when buffer is full
                if len(chunk_results) >= chunk_size:
                    chunk_df = pd.concat(chunk_results, ignore_index=True)
                    chunk_file = output_dir / f"chunk_{chunk_num:04d}.parquet"
                    chunk_df.to_parquet(chunk_file, index=False)
                    print(f"\n  Saved chunk {chunk_num}: {len(chunk_df):,} rows")
                    
                    # Clear memory
                    del chunk_df, chunk_results
                    chunk_results = []
                    chunk_num += 1
                    gc.collect()
            else:
                failed_count += 1
    
    # Save remaining data
    if chunk_results:
        chunk_df = pd.concat(chunk_results, ignore_index=True)
        chunk_file = output_dir / f"chunk_{chunk_num:04d}.parquet"
        chunk_df.to_parquet(chunk_file, index=False)
        print(f"\n  Saved final chunk {chunk_num}: {len(chunk_df):,} rows")
        del chunk_df, chunk_results
        gc.collect()
        chunk_num += 1
    
    print(f"\nDownload complete!")
    print(f"  Successful: {success_count:,}")
    print(f"  Failed/Empty: {failed_count:,}")
    print(f"  Total chunks: {chunk_num}")
    
    return output_dir, chunk_num


def merge_chunks_memory_efficient(chunk_dir, output_file, filter_wse=True):
    """Merge chunk files using pyarrow for memory efficiency."""
    
    chunk_dir = Path(chunk_dir)
    output_file = Path(output_file)
    
    chunks = sorted(chunk_dir.glob("chunk_*.parquet"))
    print(f"\nMerging {len(chunks)} chunk files...")
    
    if not chunks:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")
    
    # Get schema from first chunk
    schema = pq.read_schema(chunks[0])
    
    total_rows = 0
    filtered_rows = 0
    
    with pq.ParquetWriter(str(output_file), schema) as writer:
        for chunk_file in tqdm(chunks, desc="Merging"):
            table = pq.read_table(chunk_file)
            
            if filter_wse:
                # Convert to pandas for filtering, then back to arrow
                df = table.to_pandas()
                original_len = len(df)
                df = df.loc[df["wse"] != -999999999999.0]
                filtered_len = len(df)
                
                total_rows += original_len
                filtered_rows += filtered_len
                
                if len(df) > 0:
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    writer.write_table(table)
                
                del df
            else:
                total_rows += table.num_rows
                filtered_rows += table.num_rows
                writer.write_table(table)
            
            del table
            gc.collect()
    
    print(f"\nMerge complete!")
    print(f"  Total rows before filter: {total_rows:,}")
    print(f"  Total rows after filter: {filtered_rows:,}")
    print(f"  Output: {output_file}")
    
    return output_file


def print_summary(parquet_file):
    """Print summary statistics of the final dataset."""
    
    print("Summary")
    
    # Read only necessary columns for summary
    df = pd.read_parquet(parquet_file, columns=['reach_id', 'time_str', 'wse'])
    
    df['time_str'] = pd.to_datetime(df['time_str'])
    
    print(f"Total observations: {len(df):,}")
    print(f"Unique reaches: {df['reach_id'].nunique():,}")
    print(f"Time range: {df['time_str'].min()} to {df['time_str'].max()}")
    
    # Get column count from schema
    schema = pq.read_schema(parquet_file)
    print(f"Columns: {len(schema)}")
    
    del df
    gc.collect()


if __name__ == "__main__":
    
    print("SWOT Hydrocron Data Download (Memory-Efficient)")
    
    # Configuration
    HYDROCRON_URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
    COLLECTION_NAME = "SWOT_L2_HR_RiverSP_D"
    START_TIME = "2023-07-27T00:00:00Z"
    END_TIME = "2026-01-31T23:59:59Z"
    
    SWORD_PATH = '/nas/cee-ice/data/SWORD/SWORD_v17b/shp'
    CHUNK_DIR = "/nas/cee-ice/data/hydrocron/vD/reach_chunks"
    OUTPUT_FILE = "/nas/cee-ice/data/hydrocron/vD/SP_reaches_vD_all.parquet"
    
    MAX_WORKERS = 100
    CHUNK_SIZE = 5000  # Save every 5000 successful reaches
    
    # Fields
    all_riversp_fields = (
        "reach_id,time,time_tai,time_str,cycle_id,pass_id,"
        "p_lat,p_lon,river_name,"
        "wse,wse_u,wse_r_u,wse_c,wse_c_u,"
        "slope,slope_u,slope_r_u,slope2,slope2_u,slope2_r_u,"
        "width,width_u,width_c,width_c_u,"
        "area_total,area_tot_u,area_detct,area_det_u,area_wse,"
        "d_x_area,d_x_area_u,"
        "layovr_val,node_dist,loc_offset,xtrk_dist,"
        "dschg_c,dschg_c_u,dschg_csf,dschg_c_q,"
        "dschg_gc,dschg_gc_u,dschg_gcsf,dschg_gc_q,"
        "dschg_m,dschg_m_u,dschg_msf,dschg_m_q,"
        "dschg_gm,dschg_gm_u,dschg_gmsf,dschg_gm_q,"
        "dschg_b,dschg_b_u,dschg_bsf,dschg_b_q,"
        "dschg_gb,dschg_gb_u,dschg_gbsf,dschg_gb_q,"
        "dschg_h,dschg_h_u,dschg_hsf,dschg_h_q,"
        "dschg_gh,dschg_gh_u,dschg_ghsf,dschg_gh_q,"
        "dschg_o,dschg_o_u,dschg_osf,dschg_o_q,"
        "dschg_go,dschg_go_u,dschg_gosf,dschg_go_q,"
        "dschg_s,dschg_s_u,dschg_ssf,dschg_s_q,"
        "dschg_gs,dschg_gs_u,dschg_gssf,dschg_gs_q,"
        "dschg_i,dschg_i_u,dschg_isf,dschg_i_q,"
        "dschg_gi,dschg_gi_u,dschg_gisf,dschg_gi_q,"
        "dschg_q_b,dschg_gq_b,reach_q,reach_q_b,"
        "dark_frac,ice_clim_f,ice_dyn_f,partial_f,n_good_nod,obs_frac_n,xovr_cal_q,"
        "geoid_hght,geoid_slop,solid_tide,load_tidef,load_tideg,pole_tide,"
        "dry_trop_c,wet_trop_c,iono_c,xovr_cal_c,"
        "n_reach_up,n_reach_dn,rch_id_up,rch_id_dn,"
        "p_wse,p_wse_var,p_width,p_wid_var,p_n_nodes,p_dist_out,"
        "p_length,p_maf,p_dam_id,p_n_ch_max,p_n_ch_mod,p_low_slp,"
        "geometry"
    )
    fields = ",".join([f.strip() for f in all_riversp_fields.split(',') if f.strip()])
    
    # Step 1: Load SWORD reaches
    print("\n[Step 1] Loading SWORD reaches...")
    reaches = load_sword_reaches(SWORD_PATH)
    reach_ids = reaches.reach_id.drop_duplicates().tolist()
    
    print(f"\nNumber of reaches: {len(reach_ids):,}")
    print(f"Time range: {START_TIME} to {END_TIME}")
    
    # Free memory - don't need geometry anymore
    del reaches
    gc.collect()
    
    # Step 2: Download data with incremental save
    print("\n[Step 2] Downloading data...")
    start = time.time()
    
    chunk_dir, num_chunks = download_with_incremental_save(
        reach_ids=reach_ids,
        query_url=HYDROCRON_URL,
        start_time=START_TIME,
        end_time=END_TIME,
        fields=fields,
        collection_name=COLLECTION_NAME,
        output_dir=CHUNK_DIR,
        max_workers=MAX_WORKERS,
        chunk_size=CHUNK_SIZE
    )
    
    download_time = time.time() - start
    print(f"\nDownload time: {download_time/60:.1f} minutes")
    
    # Free memory
    del reach_ids
    gc.collect()
    
    # Step 3: Merge chunks (memory-efficient)
    print("\n[Step 3] Merging chunks...")
    start = time.time()
    
    merge_chunks_memory_efficient(
        chunk_dir=CHUNK_DIR,
        output_file=OUTPUT_FILE,
        filter_wse=True  # Filter out invalid wse values
    )
    
    merge_time = time.time() - start
    print(f"Merge time: {merge_time/60:.1f} minutes")
    
    # Step 4: Summary
    print_summary(OUTPUT_FILE)
    
    print(f"\nTotal time: {(download_time + merge_time)/60:.1f} minutes")
    print("\nDone!")