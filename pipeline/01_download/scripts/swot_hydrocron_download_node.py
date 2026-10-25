#!/usr/bin/env python
# swot_hydrocron_download_node.py
# Usage: python swot_hydrocron_download_node.py [CONTINENT]
# Example: python swot_hydrocron_download_node.py NA

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
import sys


def load_sword_nodes(sword_path, continents=None):
    """Load and merge SWORD nodes shapefiles from specified or all continents."""
    
    sword_dir = Path(sword_path)
    all_continents = ['AF', 'AS', 'EU', 'NA', 'OC', 'SA']
    
    if continents is None:
        continents = all_continents
    
    nodes_files = []
    for continent in continents:
        pattern = str(sword_dir / continent / '*nodes*.shp')
        files = glob.glob(pattern)
        nodes_files.extend(files)
        print(f"{continent}: {len(files)} files found")
    
    if not nodes_files:
        raise FileNotFoundError(f"No nodes shapefiles found in {sword_path}")
    
    print(f"\nLoading {len(nodes_files)} total files...")
    
    nodes_list = [gpd.read_file(f) for f in sorted(nodes_files)]
    nodes = gpd.GeoDataFrame(
        pd.concat(nodes_list, ignore_index=True),
        crs=nodes_list[0].crs
    )
    
    # Free memory
    del nodes_list
    gc.collect()
    
    return nodes


def query_hydrocron(node_id, query_url, start_time, end_time, fields, collection_name):
    """Query Hydrocron for node-level time series data."""
    
    params = {
        "feature": "Node",
        "feature_id": node_id,
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


def get_downloaded_nodes(output_dir):
    """Get list of already downloaded node_ids from existing chunk files."""
    
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return set()
    
    downloaded_nodes = set()
    chunk_files = sorted(output_dir.glob("chunk_*.parquet"))
    
    print(f"Found {len(chunk_files)} existing chunk files...")
    
    for chunk_file in chunk_files:
        try:
            # Read only node_id column for efficiency
            df = pd.read_parquet(chunk_file, columns=['node_id'])
            downloaded_nodes.update(df['node_id'].tolist())
            del df
        except Exception as e:
            print(f"Warning: Could not read {chunk_file.name}: {e}")
    
    print(f"Already downloaded: {len(downloaded_nodes):,} unique nodes")
    gc.collect()
    return downloaded_nodes


def download_with_incremental_save(node_ids, query_url, start_time, end_time, fields, 
                                   collection_name, output_dir, max_workers=100, 
                                   chunk_size=5000, resume=True):
    """Download data and save incrementally to avoid memory issues."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume functionality - skip already downloaded nodes
    if resume:
        downloaded_nodes = get_downloaded_nodes(output_dir)
        original_count = len(node_ids)
        node_ids = [nid for nid in node_ids if nid not in downloaded_nodes]
        skipped_count = original_count - len(node_ids)
        print(f"Resume mode: Skipping {skipped_count:,} already downloaded nodes")
        print(f"Remaining to download: {len(node_ids):,} nodes")
    
    if not node_ids:
        print("All nodes already downloaded!")
        return output_dir, len(list(output_dir.glob("chunk_*.parquet")))
    
    # Get starting chunk number from existing files
    existing_chunks = sorted(output_dir.glob("chunk_*.parquet"))
    chunk_num = len(existing_chunks)
    
    failed_count = 0
    success_count = 0
    chunk_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_node = {
            executor.submit(
                query_hydrocron, node_id, query_url, start_time, end_time, fields, collection_name
            ): node_id 
            for node_id in node_ids
        }
        
        for future in tqdm(as_completed(future_to_node), total=len(node_ids), desc="Downloading"):
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
    
    print(f"\nDownload complete!")
    print(f"  Successful: {success_count:,}")
    print(f"  Failed/Empty: {failed_count:,}")
    print(f"  Chunk files: {chunk_num + 1}")
    
    return output_dir, chunk_num + 1


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
    df = pd.read_parquet(parquet_file, columns=['node_id', 'time_str', 'wse'])
    
    df['time_str'] = pd.to_datetime(df['time_str'])
    
    print(f"Total observations: {len(df):,}")
    print(f"Unique nodes: {df['node_id'].nunique():,}")
    print(f"Time range: {df['time_str'].min()} to {df['time_str'].max()}")
    
    # Get column count from schema
    schema = pq.read_schema(parquet_file)
    print(f"Columns: {len(schema)}")
    
    del df
    gc.collect()


if __name__ == "__main__":
    
    # Parse command line argument for continent
    ALL_CONTINENTS = ['AF', 'AS', 'EU', 'NA', 'OC', 'SA']
    
    if len(sys.argv) > 1:
        continent = sys.argv[1].upper()
        if continent not in ALL_CONTINENTS:
            print(f"Error: Invalid continent '{continent}'")
            print(f"Valid options: {ALL_CONTINENTS}")
            sys.exit(1)
        continents = [continent]
        continent_suffix = f"_{continent}"
    else:
        continents = None  # All continents
        continent_suffix = "_global"
    
    print(f"SWOT Hydrocron Node Data Download - {continents or 'ALL'}")
    
    # Configuration
    HYDROCRON_URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
    COLLECTION_NAME = "SWOT_L2_HR_RiverSP_D"
    START_TIME = "2023-07-27T00:00:00Z"
    END_TIME = "2026-01-31T23:59:59Z"
    
    SWORD_PATH = '/nas/cee-ice/data/SWORD/SWORD_v17b/shp'
    BASE_DIR = "/nas/cee-ice/data/hydrocron/vD"
    CHUNK_DIR = f"{BASE_DIR}/node_chunks{continent_suffix}"
    OUTPUT_FILE = f"{BASE_DIR}/SP_nodes_vD{continent_suffix}.parquet"
    
    MAX_WORKERS = 100   # Reduced to prevent OOM
    CHUNK_SIZE = 2000  # Reduced to save memory more frequently
    
    # Fields (53 valid fields for Node - excludes wse_sm* fields)
    all_riversp_node_fields = (
        # Identifiers (5)
        "reach_id,node_id,time,time_tai,time_str,"
        
        # Location (5)
        "lat,lon,lat_u,lon_u,river_name,"
        
        # Water Surface Elevation (3) - wse_sm* fields are invalid
        "wse,wse_u,wse_r_u,"
        
        # Width (2)
        "width,width_u,"
        
        # Area (5)
        "area_total,area_tot_u,area_detct,area_det_u,area_wse,"
        
        # Geometry/Distance (4)
        "layovr_val,node_dist,xtrk_dist,flow_angle,"
        
        # Quality flags (7)
        "node_q,node_q_b,dark_frac,ice_clim_f,ice_dyn_f,n_good_pix,xovr_cal_q,"
        
        # Radar backscatter (3)
        "rdr_sig0,rdr_sig0_u,rdr_pol,"
        
        # Geophysical corrections (9)
        "geoid_hght,solid_tide,load_tidef,load_tideg,pole_tide,"
        "dry_trop_c,wet_trop_c,iono_c,xovr_cal_c,"
        
        # Prior database info (9)
        "p_wse,p_wse_var,p_width,p_wid_var,p_dist_out,p_length,p_dam_id,p_n_ch_max,p_n_ch_mod,"
        
        # Geometry (1)
        "geometry"
    )
    fields = ",".join([f.strip() for f in all_riversp_node_fields.split(',') if f.strip()])
    
    # Step 1: Load SWORD nodes
    print("\n[Step 1] Loading SWORD nodes...")
    nodes = load_sword_nodes(SWORD_PATH, continents=continents)
    node_ids = nodes.node_id.drop_duplicates().tolist()
    
    print(f"\nNumber of nodes: {len(node_ids):,}")
    print(f"Time range: {START_TIME} to {END_TIME}")
    
    # Free memory - don't need geometry anymore
    del nodes
    gc.collect()
    
    # Step 2: Download data with incremental save
    print("\n[Step 2] Downloading data...")
    start = time.time()
    
    chunk_dir, num_chunks = download_with_incremental_save(
        node_ids=node_ids,
        query_url=HYDROCRON_URL,
        start_time=START_TIME,
        end_time=END_TIME,
        fields=fields,
        collection_name=COLLECTION_NAME,
        output_dir=CHUNK_DIR,
        max_workers=MAX_WORKERS,
        chunk_size=CHUNK_SIZE,
        resume=True  # Enable resume functionality
    )
    
    download_time = time.time() - start
    print(f"\nDownload time: {download_time/60:.1f} minutes")
    
    # Free memory
    del node_ids
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
