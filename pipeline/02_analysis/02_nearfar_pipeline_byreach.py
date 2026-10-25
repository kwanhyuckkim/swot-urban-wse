#!/usr/bin/env python
"""
SWOT urban near/far pipeline, v6.5
==================================
- J2000 epoch is 2000-01-01 00:00:00 (checked against real data)
- rasterio files are opened once per worker (this was the I/O bottleneck)
- vectorized sampling with src.sample()
- one reach at a time to keep memory down
- output goes to /project (more storage)
- temp files are kept for debugging

Checked:
- TVP 매칭: 100%
- Near/Far ISA: 정상 계산 확인 (reach 74282700231, fctimp=60.5%)
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, LineString
from shapely import wkt
import rasterio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from collections import defaultdict
import os
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
BASE_PATH = Path('/work/pi_kandread_umass_edu/swot-urban')
PARQUET_PATH = BASE_PATH / '5_fctimp'
TVP_PATH = BASE_PATH / '6_pixc_tvp' / 'SWOT_TVP_Data'
POLYGON_PATH = BASE_PATH / '2_urban_reaches' / 'all_reaches_vD.parquet'
NLCD_FCTIMP_PATH = str(BASE_PATH / '0_data_collection/1_nlcd/Annual_NLCD_FctImp_2024_CU_C1V1_EPSG4326.tif')
BUILDING_HEIGHT_PATH = str(BASE_PATH / '0_data_collection/3_150m_building/GBH2020_150m_GEDI_4326.tif')

# Output to /project (more storage space)
OUTPUT_PATH = Path('/project/pi_kandread_umass_edu/swot-urban/3_analysis_results')
TEMP_PATH = OUTPUT_PATH / 'temp_reaches_v65'

# J2000 epoch for SWOT TVP data: 00:00:00, not 12:00:00 (checked)
J2000_EPOCH = pd.Timestamp('2000-01-01 00:00:00')

# Core counts managed by Slurm
N_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 48))


# =============================================================================
# Global Worker State (Initialize once per worker)
# =============================================================================
src_isa = None
src_bld = None


def init_worker(isa_path, bld_path):
    """Worker initializer: Opens raster files once and keeps them open."""
    global src_isa, src_bld
    try:
        src_isa = rasterio.open(isa_path)
    except Exception as e:
        print(f"Worker Error opening ISA: {e}")
        src_isa = None
        
    if Path(bld_path).exists():
        try:
            src_bld = rasterio.open(bld_path)
        except Exception as e:
            src_bld = None
    else:
        src_bld = None


# =============================================================================
# Sampling Functions (Optimized)
# =============================================================================
def sample_raster_along_line_optimized(line, node_pt, dataset_type='isa', n_samples=100, valid_range=(0, 100)):
    """Optimized sampling using src.sample() instead of repeated read()"""
    empty_result = {
        'mean_val': 0.0, 'nonzero_ratio': 0.0,
        'dw_mean_val': 0.0, 'dw_ratio': 0.0,
    }
    
    if dataset_type == 'isa':
        src = src_isa
    else:
        src = src_bld
    
    if src is None or line is None or line.is_empty:
        return empty_result
    
    # 1. Geometry Prep
    target_line = None
    if line.geom_type == 'MultiLineString':
        segments = list(line.geoms)
        if not segments:
            return empty_result
        target_line = min(segments, key=lambda s: s.distance(node_pt))
        if target_line.length == 0:
            return empty_result
    elif line.geom_type == 'LineString':
        if line.length == 0:
            return empty_result
        target_line = line
    else:
        return empty_result
    
    # 2. Coordinate Generation
    start_pt = Point(target_line.coords[0])
    end_pt = Point(target_line.coords[-1])
    reverse = node_pt.distance(start_pt) > node_pt.distance(end_pt)
    
    coords = []
    indices = []
    
    for i in range(n_samples + 1):
        t = 1.0 - (i / n_samples) if reverse else i / n_samples
        pt = target_line.interpolate(t, normalized=True)
        coords.append((pt.x, pt.y))
        indices.append(i)
    
    # 3. Fast Sampling (Vectorized)
    values = []
    try:
        sampled_gen = src.sample(coords, indexes=1)
        
        for idx, val in zip(indices, sampled_gen):
            v = float(val[0])
            
            if dataset_type == 'building':
                if v >= 0 and not np.isnan(v):
                    values.append((idx, v))
            else:  # ISA
                if valid_range[0] <= v <= valid_range[1]:
                    values.append((idx, v))
    except Exception:
        return empty_result
    
    if not values:
        return empty_result
    
    # 4. Statistics Calculation
    all_vals = [v for _, v in values]
    mean_val = np.mean(all_vals)
    nonzero_ratio = sum(1 for v in all_vals if v > 0) / (n_samples + 1)
    
    weighted_sum = 0.0
    weight_sum = 0.0
    weighted_nonzero_sum = 0.0
    
    for idx, val in values:
        weight = (n_samples - idx) / n_samples
        weighted_sum += weight * val
        weight_sum += weight
        if val > 0:
            weighted_nonzero_sum += weight
    
    dw_mean_val = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    dw_ratio = weighted_nonzero_sum / weight_sum if weight_sum > 0 else 0.0
    
    return {
        'mean_val': mean_val, 'nonzero_ratio': nonzero_ratio,
        'dw_mean_val': dw_mean_val, 'dw_ratio': dw_ratio,
    }


def calculate_near_far_metrics(node_lon, node_lat, sat_lon, sat_lat, polygon):
    """Calculate Near/Far metrics using optimized sampling."""
    empty_result = {
        'near_isa': 0.0, 'far_isa': 0.0, 'near_ratio': 0.0, 'far_ratio': 0.0,
        'dw_near_isa': 0.0, 'dw_far_isa': 0.0, 'dw_near_ratio': 0.0, 'dw_far_ratio': 0.0,
        'near_building': 0.0, 'far_building': 0.0, 'near_ratio_building': 0.0, 'far_ratio_building': 0.0,
        'dw_near_building': 0.0, 'dw_far_building': 0.0, 'dw_near_ratio_building': 0.0, 'dw_far_ratio_building': 0.0,
    }
    
    node_pt = Point(node_lon, node_lat)
    dx = sat_lon - node_lon
    dy = sat_lat - node_lat
    length = np.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return empty_result
    
    ux, uy = dx / length, dy / length
    extend = 0.2
    
    near_line = LineString([node_pt, Point(node_lon + ux * extend, node_lat + uy * extend)])
    far_line = LineString([node_pt, Point(node_lon - ux * extend, node_lat - uy * extend)])
    
    try:
        near_in_poly = near_line.intersection(polygon) if near_line.intersects(polygon) else None
        far_in_poly = far_line.intersection(polygon) if far_line.intersects(polygon) else None
    except:
        return empty_result
    
    result = empty_result.copy()
    
    # ISA sampling
    near_isa = sample_raster_along_line_optimized(near_in_poly, node_pt, dataset_type='isa')
    far_isa = sample_raster_along_line_optimized(far_in_poly, node_pt, dataset_type='isa')
    
    result.update({
        'near_isa': near_isa['mean_val'], 'far_isa': far_isa['mean_val'],
        'near_ratio': near_isa['nonzero_ratio'], 'far_ratio': far_isa['nonzero_ratio'],
        'dw_near_isa': near_isa['dw_mean_val'], 'dw_far_isa': far_isa['dw_mean_val'],
        'dw_near_ratio': near_isa['dw_ratio'], 'dw_far_ratio': far_isa['dw_ratio'],
    })
    
    # Building sampling
    if src_bld is not None:
        near_bld = sample_raster_along_line_optimized(near_in_poly, node_pt, dataset_type='building')
        far_bld = sample_raster_along_line_optimized(far_in_poly, node_pt, dataset_type='building')
        
        result.update({
            'near_building': near_bld['mean_val'], 'far_building': far_bld['mean_val'],
            'near_ratio_building': near_bld['nonzero_ratio'], 'far_ratio_building': far_bld['nonzero_ratio'],
            'dw_near_building': near_bld['dw_mean_val'], 'dw_far_building': far_bld['dw_mean_val'],
            'dw_near_ratio_building': near_bld['dw_ratio'], 'dw_far_ratio_building': far_bld['dw_ratio'],
        })
    
    return result


def get_geometry_category(near_val, far_val):
    has_near, has_far = near_val > 0, far_val > 0
    if has_near and has_far:
        return 'Near/Far'
    elif has_near:
        return 'Near'
    elif has_far:
        return 'Far'
    else:
        return 'NA'


def get_sample_geometry_category(near_val, far_val, near_ratio, far_ratio, min_ratio=0.5):
    has_near = near_val > 0 and near_ratio >= min_ratio
    has_far = far_val > 0 and far_ratio >= min_ratio
    if has_near and has_far:
        return 'Near/Far'
    elif has_near:
        return 'Near'
    elif has_far:
        return 'Far'
    else:
        return 'NA'


def process_single_task(task):
    """Process single observation using global raster handles."""
    idx, node_lon, node_lat, sat_lon, sat_lat, polygon_wkt = task
    
    result = {
        'idx': idx,
        'near_isa': 0.0, 'far_isa': 0.0, 'near_ratio': 0.0, 'far_ratio': 0.0,
        'dw_near_isa': 0.0, 'dw_far_isa': 0.0, 'dw_near_ratio': 0.0, 'dw_far_ratio': 0.0,
        'near_building': 0.0, 'far_building': 0.0, 'near_ratio_building': 0.0, 'far_ratio_building': 0.0,
        'dw_near_building': 0.0, 'dw_far_building': 0.0, 'dw_near_ratio_building': 0.0, 'dw_far_ratio_building': 0.0,
        'geometry_category': 'NA', 'sample_geometry_category': 'NA',
        'geometry_category_building': 'NA', 'sample_geometry_category_building': 'NA',
    }
    
    if sat_lon is None or sat_lat is None or np.isnan(sat_lon) or np.isnan(sat_lat):
        result['geometry_category'] = 'No_Sat_Data'
        return result
    
    if not (np.isfinite(node_lon) and np.isfinite(sat_lon)):
        return result
    
    try:
        polygon = wkt.loads(polygon_wkt)
    except:
        return result
    
    metrics = calculate_near_far_metrics(node_lon, node_lat, sat_lon, sat_lat, polygon)
    result.update(metrics)
    
    # Categories
    result['geometry_category'] = get_geometry_category(result['near_isa'], result['far_isa'])
    result['sample_geometry_category'] = get_sample_geometry_category(
        result['near_isa'], result['far_isa'], result['near_ratio'], result['far_ratio']
    )
    result['geometry_category_building'] = get_geometry_category(result['near_building'], result['far_building'])
    result['sample_geometry_category_building'] = get_sample_geometry_category(
        result['near_building'], result['far_building'], result['near_ratio_building'], result['far_ratio_building']
    )
    
    return result


# =============================================================================
# Process Single Reach
# =============================================================================
def process_single_reach(reach_id, node_files_list, tvp_file, polygon_wkt, temp_dir):
    """Process a single reach with optimized parallel workers."""
    out_file = temp_dir / f'reach_{reach_id}.parquet'
    
    # Skip if already processed
    if out_file.exists():
        return out_file
    
    if not isinstance(node_files_list, list):
        node_files_list = [node_files_list]
    
    try:
        # Load ALL node files for this reach
        node_dfs = []
        for nf in node_files_list:
            try:
                df = pd.read_parquet(nf)
                node_dfs.append(df)
            except:
                pass
        
        if not node_dfs:
            return None
        
        node_df = pd.concat(node_dfs, ignore_index=True)
        del node_dfs
        
        # Load TVP data if exists
        if tvp_file and tvp_file.exists():
            tvp_df = pd.read_parquet(tvp_file)
            
            # Time conversion - J2000 epoch (00:00:00) for TVP data!
            node_df['time_dt'] = pd.to_datetime(node_df['time_str']).dt.tz_localize(None)
            tvp_df['time_dt'] = J2000_EPOCH + pd.to_timedelta(tvp_df['time'], unit='s')
            
            node_df = node_df.sort_values('time_dt').reset_index(drop=True)
            tvp_df = tvp_df.sort_values('time_dt')
            
            node_df = pd.merge_asof(
                node_df,
                tvp_df[['time_dt', 'latitude', 'longitude', 'altitude']],
                on='time_dt',
                direction='nearest',
                tolerance=pd.Timedelta('30s')
            ).rename(columns={
                'latitude': 'sat_lat',
                'longitude': 'sat_lon',
                'altitude': 'sat_altitude'
            })
            
            del tvp_df
        else:
            node_df['sat_lon'] = np.nan
            node_df['sat_lat'] = np.nan
            node_df['sat_altitude'] = np.nan
        
        # Prepare tasks
        tasks = []
        for idx, row in node_df.iterrows():
            task = (idx, row['lon'], row['lat'], row['sat_lon'], row['sat_lat'], polygon_wkt)
            tasks.append(task)
        
        # Process with parallel workers
        if len(tasks) > 0:
            with ProcessPoolExecutor(
                max_workers=N_WORKERS,
                initializer=init_worker,
                initargs=(NLCD_FCTIMP_PATH, BUILDING_HEIGHT_PATH)
            ) as executor:
                results = list(executor.map(process_single_task, tasks, chunksize=100))
            
            # Merge results
            results_df = pd.DataFrame(results).set_index('idx')
            
            result_cols = [
                'near_isa', 'far_isa', 'near_ratio', 'far_ratio',
                'dw_near_isa', 'dw_far_isa', 'dw_near_ratio', 'dw_far_ratio',
                'near_building', 'far_building', 'near_ratio_building', 'far_ratio_building',
                'dw_near_building', 'dw_far_building', 'dw_near_ratio_building', 'dw_far_ratio_building',
                'geometry_category', 'sample_geometry_category',
                'geometry_category_building', 'sample_geometry_category_building',
            ]
            
            for col in result_cols:
                if col in results_df.columns:
                    node_df[col] = results_df[col]
            
            del results, results_df
        
        # Save
        node_df.to_parquet(out_file)
        
        n_rows = len(node_df)
        del node_df
        gc.collect()
        
        return out_file
        
    except Exception as e:
        print(f"\n      Error processing reach {reach_id}: {e}")
        return None


# =============================================================================
# Main Pipeline
# =============================================================================
def run_pipeline():
    """Run per-reach pipeline with optimized workers."""
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    TEMP_PATH.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SWOT urban near/far pipeline v6.5")
    print(f"  Workers per reach: {N_WORKERS}")
    print(f"  Rasterio I/O: Global worker state (open once per worker)")
    print(f"  TVP time: J2000 epoch (2000-01-01 00:00:00)")
    print(f"  Output: {OUTPUT_PATH}")
    print("=" * 70)
    
    total_start = time.time()
    
    # ==========================================================================
    # PHASE 1: Build lookup tables
    # ==========================================================================
    print("\n[PHASE 1] Building lookup tables...")
    
    # Load polygons -> dict
    print("  Loading polygons...")
    polygons_gdf = gpd.read_parquet(POLYGON_PATH)
    if polygons_gdf.crs != 'EPSG:4326':
        polygons_gdf = polygons_gdf.to_crs('EPSG:4326')
    
    polygon_wkt_dict = {}
    for _, row in polygons_gdf.iterrows():
        rid = row['reach_id']
        polygon_wkt_dict[int(rid)] = row.geometry.wkt
    
    print(f"    {len(polygon_wkt_dict)} polygons")
    del polygons_gdf
    gc.collect()
    
    # Build node file lookup: reach_id -> LIST of file paths
    print("  Scanning node files...")
    node_files = sorted(PARQUET_PATH.glob('fctimp_mean_*_node_*.parquet'))
    
    node_file_dict = defaultdict(list)
    
    for f in node_files:
        try:
            df_sample = pd.read_parquet(f, columns=['reach_id'])
            if len(df_sample) > 0:
                reach_id = int(df_sample['reach_id'].iloc[0])
                node_file_dict[reach_id].append(f)
        except Exception as e:
            print(f"    Warning: Could not read {f.name}: {e}")
    
    node_file_dict = dict(node_file_dict)
    
    total_files = sum(len(files) for files in node_file_dict.values())
    print(f"    {total_files} node files mapped to {len(node_file_dict)} reaches")
    
    # Build TVP file lookup: reach_id -> file path
    print("  Scanning TVP files...")
    tvp_files = sorted(TVP_PATH.glob('tvp_reach_*.parquet'))
    
    tvp_file_dict = {}
    for f in tvp_files:
        try:
            reach_id = int(f.stem.split('_')[-1])
            tvp_file_dict[reach_id] = f
        except:
            pass
    
    print(f"    {len(tvp_file_dict)} TVP files")
    
    # ==========================================================================
    # PHASE 2: Process each reach
    # ==========================================================================
    valid_reaches = set(polygon_wkt_dict.keys()) & set(node_file_dict.keys())
    valid_reaches = sorted(valid_reaches)
    
    print(f"\n[PHASE 2] Processing {len(valid_reaches)} reaches...")
    
    processed_files = []
    total_rows = 0
    
    for i, reach_id in enumerate(valid_reaches, 1):
        node_files_list = node_file_dict[reach_id]
        tvp_file = tvp_file_dict.get(reach_id)
        polygon_wkt = polygon_wkt_dict[reach_id]
        
        print(f"  [{i}/{len(valid_reaches)}] Reach {reach_id} ({len(node_files_list)} node files)...", end=" ", flush=True)
        start = time.time()
        
        result_file = process_single_reach(reach_id, node_files_list, tvp_file, polygon_wkt, TEMP_PATH)
        
        if result_file:
            processed_files.append(result_file)
            try:
                n_rows = len(pd.read_parquet(result_file, columns=['reach_id']))
                total_rows += n_rows
                print(f"done ({n_rows} rows, {time.time()-start:.1f}s)")
            except:
                print(f"done ({time.time()-start:.1f}s)")
        else:
            print("failed")
        
        gc.collect()
    
    # ==========================================================================
    # PHASE 3: Merge all results
    # ==========================================================================
    print(f"\n[PHASE 3] Merging {len(processed_files)} reach files...")
    
    all_dfs = []
    for f in processed_files:
        try:
            all_dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  Error reading {f}: {e}")
    
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs
        gc.collect()
        
        # Save final results (parquet only)
        out_parquet = OUTPUT_PATH / 'swot_urban_nearfar_v6.5_results.parquet'
        final_df.to_parquet(out_parquet)
        print(f"  Saved: {out_parquet}")
        
        # Keep temp files for debugging
        print(f"  Temp files kept at: {TEMP_PATH}")
        
        # ==========================================================================
        # Summary
        # ==========================================================================
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total rows: {len(final_df)}")
        print(f"Total reaches: {len(valid_reaches)}")
        print(f"Total time: {(time.time()-total_start)/60:.1f} minutes")
        
        # TVP match rate
        sat_matched = final_df['sat_lon'].notna().sum()
        print(f"\n[TVP Match Rate: {sat_matched}/{len(final_df)} ({100*sat_matched/len(final_df):.1f}%)]")
        
        print(f"\n[ISA Geometry Category]")
        print(final_df['geometry_category'].value_counts())
        
        print(f"\n[Building Geometry Category]")
        print(final_df['geometry_category_building'].value_counts())
        
        valid_mask = final_df['geometry_category'] != 'No_Sat_Data'
        n_valid = valid_mask.sum()
        print(f"\n[Valid observations: {n_valid}/{len(final_df)} ({100*n_valid/len(final_df):.1f}%)]")
        
        if valid_mask.any():
            print(f"\n[ISA Stats (valid only)]")
            print(f"  near_isa:    mean={final_df.loc[valid_mask, 'near_isa'].mean():.2f}, "
                  f"median={final_df.loc[valid_mask, 'near_isa'].median():.2f}")
            print(f"  far_isa:     mean={final_df.loc[valid_mask, 'far_isa'].mean():.2f}, "
                  f"median={final_df.loc[valid_mask, 'far_isa'].median():.2f}")
            print(f"  dw_near_isa: mean={final_df.loc[valid_mask, 'dw_near_isa'].mean():.2f}, "
                  f"median={final_df.loc[valid_mask, 'dw_near_isa'].median():.2f}")
            print(f"  dw_far_isa:  mean={final_df.loc[valid_mask, 'dw_far_isa'].mean():.2f}, "
                  f"median={final_df.loc[valid_mask, 'dw_far_isa'].median():.2f}")
            
            print(f"\n[Building Stats (valid only)]")
            print(f"  near_building:    mean={final_df.loc[valid_mask, 'near_building'].mean():.2f}, "
                  f"median={final_df.loc[valid_mask, 'near_building'].median():.2f}")
            print(f"  far_building:     mean={final_df.loc[valid_mask, 'far_building'].mean():.2f}, "
                  f"median={final_df.loc[valid_mask, 'far_building'].median():.2f}")
            print(f"  dw_near_building: mean={final_df.loc[valid_mask, 'dw_near_building'].mean():.2f}, "
                  f"median={final_df.loc[valid_mask, 'dw_near_building'].median():.2f}")
            print(f"  dw_far_building:  mean={final_df.loc[valid_mask, 'dw_far_building'].mean():.2f}, "
                  f"median={final_df.loc[valid_mask, 'dw_far_building'].median():.2f}")
        
        return final_df
    else:
        print("  Error: No data to merge")
        return None


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    run_pipeline()