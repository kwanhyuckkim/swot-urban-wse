#!/usr/bin/env python
# Print-size rebuild of Figure 1 (paired PIXC comparison); ported from the notebook cell.
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Polygon as MplPolygon
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
import rasterio
from pathlib import Path
import xarray as xr
from shapely.geometry import Point
from datetime import datetime, timedelta
import re

# Helper functions

def filter_best_pixc_files(files):
    file_dict = {}
    for f in files:
        match = re.match(r'(.+)_(P[IG]D\d)_(\d+)\.nc', f.name)
        if not match:
            continue
        base, processor, counter = match.group(1), match.group(2), int(match.group(3))
        if base not in file_dict:
            file_dict[base] = (f, processor, counter)
        else:
            _, old_proc, old_cnt = file_dict[base]
            if (processor == 'PGD0' and old_proc == 'PID0') or (processor == old_proc and counter > old_cnt):
                file_dict[base] = (f, processor, counter)
    return [f for f, _, _ in file_dict.values()]


def parse_filename_info(filename):
    parts = filename.split('_')
    if len(parts) >= 7:
        cycle_id = parts[4]
        pass_id = parts[5]
        tile_id = parts[6]
        return cycle_id, pass_id, tile_id
    return 'UNK', 'UNK', 'UNK'


def match_and_group_pixc(time_str, files, crid, window_minutes=5):
    dt = pd.to_datetime(time_str)
    if dt.tzinfo:
        dt = dt.tz_localize(None)
    start, end = dt - timedelta(minutes=window_minutes), dt + timedelta(minutes=window_minutes)
    groups = {}
    for f in files:
        parts = f.name.split('_')
        if len(parts) >= 10:
            try:
                fs, fe = pd.to_datetime(parts[7], format='%Y%m%dT%H%M%S'), pd.to_datetime(parts[8], format='%Y%m%dT%H%M%S')
                if not (fe < start or fs > end) and parts[9] == crid:
                    key = f"{parts[7]}_{parts[8]}"
                    groups.setdefault(key, []).append(f)
            except:
                continue
    return groups


def find_matching_pixcvec(pixc_file, pixcvec_files):
    pixc_name = pixc_file.name.replace('PIXC_', 'PIXCVec_')
    for pvf in pixcvec_files:
        if pvf.name == pixc_name:
            return pvf
    return None


def extract_and_merge_pixels(pixc_files, pixcvec_files, polygon_gdf):
    all_data = [extract_pixels_with_vec(pf, pvf, polygon_gdf) for pf, pvf in zip(pixc_files, pixcvec_files)]
    all_data = [d for d in all_data if d]
    if not all_data:
        return None
    if len(all_data) == 1:
        return all_data[0]
    merged = {k: np.concatenate([d[k] for d in all_data]) for k in all_data[0].keys() 
              if k not in ['heading', 'sat_info']}
    merged['heading'] = np.nanmean([d['heading'] for d in all_data])
    merged['sat_info'] = all_data[0]['sat_info']
    return merged


def extract_pixels_with_vec(pixc_file, pixcvec_file, polygon_gdf):
    ds = xr.open_dataset(pixc_file, group='pixel_cloud')
    lon, lat = ds['longitude'].values, ds['latitude'].values
    bounds = polygon_gdf.total_bounds
    
    ds_tvp = xr.open_dataset(pixc_file, group='tvp')
    heading = np.nanmedian(ds_tvp['velocity_heading'].values)
    sat_lon = np.nanmedian(ds_tvp['longitude'].values)
    sat_lat = np.nanmedian(ds_tvp['latitude'].values)
    sat_alt = np.nanmedian(ds_tvp['altitude'].values)
    ds_tvp.close()
    
    bbox_mask = ((lon >= bounds[0]) & (lon <= bounds[2]) & (lat >= bounds[1]) & (lat <= bounds[3]))
    if bbox_mask.sum() == 0:
        ds.close()
        return None
    
    classification = ds['classification'].values[bbox_mask]
    water_mask = classification > 2
    if water_mask.sum() == 0:
        ds.close()
        return None
    
    lon_filt, lat_filt = lon[bbox_mask][water_mask], lat[bbox_mask][water_mask]
    poly = polygon_gdf.geometry.iloc[0]
    inside_mask = np.array([poly.contains(Point(x, y)) for x, y in zip(lon_filt, lat_filt)])
    if inside_mask.sum() == 0:
        ds.close()
        return None
    
    final_idx = np.where(bbox_mask)[0][water_mask][inside_mask]
    
    cross_track = ds['cross_track'].values[final_idx]
    median_cross_track = np.nanmedian(cross_track)
    
    ds_vec = xr.open_dataset(pixcvec_file)
    reach_id = ds_vec['reach_id'].values[final_idx]
    node_id = ds_vec['node_id'].values[final_idx]
    obs_id = ds_vec['obs_id'].values[final_idx]
    lake_id = ds_vec['lake_id'].values[final_idx]
    
    def is_valid(arr):
        return np.array([len(x) > 0 and x != b'' for x in arr]) if arr.dtype.kind == 'S' else arr > 0
    
    selected_mask = is_valid(reach_id) & is_valid(node_id) & ~is_valid(obs_id) & ~is_valid(lake_id)
    
    result = {
        'lon': lon[final_idx], 'lat': lat[final_idx],
        'height': ds['height'].values[final_idx],
        'classification': ds['classification'].values[final_idx],
        'azimuth': ds['azimuth_index'].values[final_idx],
        'range': ds['range_index'].values[final_idx],
        'cross_track': cross_track,
        'selected_mask': selected_mask,
        'heading': heading,
        'sat_info': {
            'lon': sat_lon,
            'lat': sat_lat,
            'altitude': sat_alt,
            'cross_track': median_cross_track
        }
    }
    ds.close()
    ds_vec.close()
    return result


# Constants

CLASSIFICATION_COLORS = [
    'maroon', 'gold', 'skyblue', 'mediumblue', 'dimgray', 'cyan', 'darkturquoise'
]

CLASSIFICATION_LABELS = [
    'Land', 'Land (near water)', 'Water (near land)', 'Open Water',
    'Dark Water', 'Low Coh. (near land)', 'Low Coh. (open)'
]

NLCD_COLORS = {
    11: '#476BA1', 12: '#D1DDF9', 21: '#DDC9C9', 22: '#D89382',
    23: '#ED0000', 24: '#AA0000', 31: '#B2ADA3', 41: '#68AA63',
    42: '#1C6330', 43: '#B5C98E', 52: '#CCBA7C', 71: '#E2E2C1',
    81: '#DDD83E', 82: '#AE7229', 90: '#BBD7ED', 95: '#71A4C1'
}

NLCD_LABELS = {
    11: 'Open Water', 12: 'Ice/Snow', 21: 'Developed (Open)', 22: 'Developed (Low)',
    23: 'Developed (Med)', 24: 'Developed (High)', 31: 'Barren', 41: 'Deciduous Forest',
    42: 'Evergreen Forest', 43: 'Mixed Forest', 52: 'Shrub/Scrub', 71: 'Grassland',
    81: 'Pasture/Hay', 82: 'Cultivated', 90: 'Woody Wetlands', 95: 'Herbaceous Wetlands'
}

POLYGON_COLOR = '#404040'


def lon_formatter(x, pos):
    if x < 0:
        return f'{abs(x):.2f}°W'
    elif x > 0:
        return f'{x:.2f}°E'
    else:
        return '0°'

def lat_formatter(y, pos):
    if y > 0:
        return f'{y:.2f}°N'
    elif y < 0:
        return f'{abs(y):.2f}°S'
    else:
        return '0°'


def get_satellite_side(heading, cross_track):
    """위성이 왼쪽에서 오는지 오른쪽에서 오는지 결정"""
    if (0 <= heading < 90) or (270 <= heading < 360):
        return 'left' if cross_track >= 0 else 'right'
    else:
        return 'right' if cross_track >= 0 else 'left'


def draw_satellite_swath_range(ax, side, heading, cycle_id, pass_id, tile_id, 
                                x_min, x_max, y_min, y_max, band_width, text_color='black'):
    """
    Classification plot에 기울어진 위성 경로 band 그리기
    heading에 따라 14도 또는 166도 기울기
    """
    # 북위 기준 각도
    going_north = (0 <= heading < 90) or (270 <= heading < 360)
    angle = 14 if going_north else 166
    angle_rad = np.radians(angle)
    
    # band 중심 x 위치
    if side == 'left':
        center_x = x_min + band_width / 2
    else:
        center_x = x_max - band_width / 2
    
    # 기울기에 따른 offset 계산
    y_span = y_max - y_min
    x_offset = y_span * np.tan(np.radians(angle - 90)) * 0.3  # 살짝 기울기
    
    # 4개 꼭지점 (기울어진 평행사변형)
    if going_north:  # 14도 - 위로 갈수록 오른쪽으로
        points = [
            (center_x - band_width/2 - x_offset, y_min),
            (center_x + band_width/2 - x_offset, y_min),
            (center_x + band_width/2 + x_offset, y_max),
            (center_x - band_width/2 + x_offset, y_max),
        ]
    else:  # 166도 - 아래로 갈수록 오른쪽으로
        points = [
            (center_x - band_width/2 + x_offset, y_min),
            (center_x + band_width/2 + x_offset, y_min),
            (center_x + band_width/2 - x_offset, y_max),
            (center_x - band_width/2 - x_offset, y_max),
        ]
    
    band_patch = MplPolygon(points, facecolor='#505050', edgecolor='#404040', 
                             alpha=0.8, linewidth=2.5, zorder=3)
    ax.add_patch(band_patch)
    
    # 텍스트 위치 (band 중앙)
    text_x = center_x
    text_y_center = (y_min + y_max) / 2
    text_spacing = y_span * 0.12
    text_size = 12
    
    ax.text(text_x, text_y_center + text_spacing, f'Tile: {tile_id}', fontsize=text_size, fontweight='bold',
            color=text_color, ha='center', va='center', zorder=5)
    ax.text(text_x, text_y_center, f'Pass: {pass_id}', fontsize=text_size, fontweight='bold',
            color=text_color, ha='center', va='center', zorder=5)
    ax.text(text_x, text_y_center - text_spacing, f'Cycle: {cycle_id}', fontsize=text_size, fontweight='bold',
            color=text_color, ha='center', va='center', zorder=5)
    
    # Heading 각도 표시 (band 위 또는 아래)
    if going_north:
        ax.text(text_x, y_max + y_span * 0.03, f'{heading:.0f}°', fontsize=text_size + 1, fontweight='bold',
                color=text_color, ha='center', va='bottom', zorder=5)
    else:
        ax.text(text_x, y_min - y_span * 0.03, f'{heading:.0f}°', fontsize=text_size + 1, fontweight='bold',
                color=text_color, ha='center', va='top', zorder=5)


def draw_satellite_swath_lonlat(ax, side, heading, cycle_id, pass_id, tile_id,
                                 x_min, x_max, y_min, y_max, band_width, text_color='white'):
    """
    NLCD plot에 기울어진 위성 경로 band 그리기
    텍스트는 안쪽으로 (짤리지 않게)
    """
    going_north = (0 <= heading < 90) or (270 <= heading < 360)
    
    if side == 'left':
        center_x = x_min + band_width / 2
    else:
        center_x = x_max - band_width / 2
    
    y_span = y_max - y_min
    x_offset = y_span * 0.15
    
    if going_north:
        points = [
            (center_x - band_width/2 - x_offset * 0.3, y_min),
            (center_x + band_width/2 - x_offset * 0.3, y_min),
            (center_x + band_width/2 + x_offset * 0.3, y_max),
            (center_x - band_width/2 + x_offset * 0.3, y_max),
        ]
    else:
        points = [
            (center_x - band_width/2 + x_offset * 0.3, y_min),
            (center_x + band_width/2 + x_offset * 0.3, y_min),
            (center_x + band_width/2 - x_offset * 0.3, y_max),
            (center_x - band_width/2 - x_offset * 0.3, y_max),
        ]
    
    band_patch = MplPolygon(points, facecolor='#505050', edgecolor='#404040', 
                             alpha=0.8, linewidth=2.5, zorder=3)
    ax.add_patch(band_patch)
    
    # 텍스트 위치 - 더 안쪽으로 (band 중심에서 안쪽으로 offset)
    if side == 'left':
        text_x = center_x + band_width * 0.15  # 오른쪽으로
    else:
        text_x = center_x - band_width * 0.15  # 왼쪽으로
    
    text_y_center = (y_min + y_max) / 2
    text_spacing = y_span * 0.10  # 간격 줄임
    text_size = 11  # 글씨 크기 약간 줄임
    
    ax.text(text_x, text_y_center + text_spacing, f'Tile: {tile_id}', fontsize=text_size, fontweight='bold',
            color=text_color, ha='center', va='center', zorder=5)
    ax.text(text_x, text_y_center, f'Pass: {pass_id}', fontsize=text_size, fontweight='bold',
            color=text_color, ha='center', va='center', zorder=5)
    ax.text(text_x, text_y_center - text_spacing, f'Cycle: {cycle_id}', fontsize=text_size, fontweight='bold',
            color=text_color, ha='center', va='center', zorder=5)
    
    # Heading 각도 - band 안쪽 상단/하단
    if going_north:
        ax.text(text_x, y_max - y_span * 0.05, f'{heading:.0f}°', fontsize=text_size + 1, fontweight='bold',
                color=text_color, ha='center', va='top', zorder=5)
    else:
        ax.text(text_x, y_min + y_span * 0.05, f'{heading:.0f}°', fontsize=text_size + 1, fontweight='bold',
                color=text_color, ha='center', va='bottom', zorder=5)


# Main function

def process_fctimp(fctimp_str, spike_threshold=0.1832, output_dir=None, year=2025):
    if output_dir is None:
        output_dir = f'isa_{fctimp_str}_v2'
    Path(output_dir).mkdir(exist_ok=True)
    
    selected_12 = gpd.read_file('/work/pi_kandread_umass_edu/swot-urban/2_urban_reaches/selected_12_vD.shp')
    fctimp_val = float(fctimp_str)
    sample = selected_12[np.isclose(selected_12["fctimp_mea"], fctimp_val, atol=0.01)].copy()
    if sample.empty:
        print(f"No reach for {fctimp_str}")
        return
    
    df = pd.read_parquet(f'/work/pi_kandread_umass_edu/swot-urban/1_data_processing/fctimp_mean_{fctimp_str}_sample.parquet')
    d = df.copy()
    d['time_dt'] = pd.to_datetime(d['time_str'])
    d = d.sort_values('time_dt').reset_index(drop=True)
    d = d[d['time_dt'].dt.year == year].reset_index(drop=True)
    
    if 'error_relative_m' in d.columns:
        d['error'] = d['error_relative_m']
    else:
        d['error'] = d['wse_relative_m'] - d['wse_usgs_relative_m']
    
    d['is_spike'] = d['error'].abs() >= spike_threshold
    
    print(f"ISA = {fctimp_str} ({year}): {len(d)} observations")
    print(f"  High Error (|error| >= {spike_threshold}m): {d['is_spike'].sum()}")
    print(f"  Low Error (|error| < {spike_threshold}m): {(~d['is_spike']).sum()}")
    
    pixc_dir = Path(f'/project/pi_kandread_umass_edu/swot-urban/0_data_collection/SWOT_PIXC/fctimp_{fctimp_str}')
    pixcvec_dir = Path(f'/project/pi_kandread_umass_edu/swot-urban/0_data_collection/SWOT_PIXCVec/fctimp_{fctimp_str}')
    pixc_files = filter_best_pixc_files(list(pixc_dir.glob('SWOT_L2_HR_PIXC_*.nc')))
    pixcvec_files = filter_best_pixc_files(list(pixcvec_dir.glob('SWOT_L2_HR_PIXCVec_*.nc')))
    
    print(f"  PIXC files: {len(pixc_files)}")
    
    polygon = sample.to_crs('EPSG:4326')
    bounds = polygon.total_bounds
    nlcd_path = '/work/pi_kandread_umass_edu/swot-urban/0_data_collection/1_nlcd/Annual_NLCD_LndCov_2024_CU_C1V1_EPSG4326.tif'
    
    for idx, spike_row in d[d['is_spike']].iterrows():
        print(f"High Error: {spike_row['time_str']} (error={spike_row['error']:.2f}m)")
        
        normals = d[~d['is_spike']].copy()
        if len(normals) == 0:
            continue
        normals['time_diff'] = (normals['time_dt'] - spike_row['time_dt']).abs()
        closest_normal = normals.loc[normals['time_diff'].idxmin()]
        if not (abs(abs(spike_row['error']) - 0.37) < 0.03
                and abs(closest_normal['error']) < 0.05):
            print('  (skip: not the manuscript example pair)')
            continue
        print(f"Low Error: {closest_normal['time_str']} (error={closest_normal['error']:.2f}m)")
        
        # Get Low Error data
        normal_data = None
        normal_info = None
        for group_key, nf_list in match_and_group_pixc(closest_normal['time_str'], pixc_files, closest_normal['crid']).items():
            vec_list = [v for v in [find_matching_pixcvec(nf, pixcvec_files) for nf in nf_list] if v]
            if not vec_list:
                continue
            data = extract_and_merge_pixels(nf_list, vec_list, polygon)
            if data:
                cycle_id, pass_id, tile_id = parse_filename_info(nf_list[0].name)
                normal_data = data
                normal_info = {
                    'time_str': closest_normal['time_dt'].strftime('%Y%m%dT%H%M%S'),
                    'error': closest_normal['error'],
                    'cycle_id': cycle_id,
                    'pass_id': pass_id,
                    'tile_id': tile_id,
                    'heading': data['heading'],
                    'sat_info': data['sat_info']
                }
                break
        
        # Get High Error data
        spike_data = None
        spike_info = None
        for group_key, sf_list in match_and_group_pixc(spike_row['time_str'], pixc_files, spike_row['crid']).items():
            vec_list = [v for v in [find_matching_pixcvec(sf, pixcvec_files) for sf in sf_list] if v]
            if not vec_list:
                continue
            data = extract_and_merge_pixels(sf_list, vec_list, polygon)
            if data:
                cycle_id, pass_id, tile_id = parse_filename_info(sf_list[0].name)
                spike_data = data
                spike_info = {
                    'time_str': spike_row['time_dt'].strftime('%Y%m%dT%H%M%S'),
                    'error': spike_row['error'],
                    'cycle_id': cycle_id,
                    'pass_id': pass_id,
                    'tile_id': tile_id,
                    'heading': data['heading'],
                    'sat_info': data['sat_info']
                }
                break
        
        if normal_data and spike_data:
            save_comparison_plot(normal_data, spike_data, normal_info, spike_info, 
                                 polygon, bounds, output_dir, fctimp_str, nlcd_path)
        else:
            if not normal_data:
                print("  -> Could not extract Low Error data")
            if not spike_data:
                print("  -> Could not extract High Error data")
    
    print(f"\nDone: {output_dir}")


# Plot function

def save_comparison_plot(normal_data, spike_data, normal_info, spike_info, polygon, bounds, output_dir, fctimp_str, nlcd_path):
    time_str = spike_info['time_str']
    
    FONT_SIZE = 27
    LON_EXTEND = 0.06  # NLCD만 확장
    BAND_WIDTH_LON = 0.035
    
    fig = plt.figure(figsize=(21, 16))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 0.15, 0.20], 
                  hspace=0.42, wspace=0.22)
    
    unique_vals = None
    
    for row, (data, info, label) in enumerate([
        (normal_data, normal_info, 'Low Error'),
        (spike_data, spike_info, 'High Error')
    ]):
        heading = info['heading']
        error = info['error']
        cycle_id = info['cycle_id']
        pass_id = info['pass_id']
        tile_id = info['tile_id']
        cross_track = info['sat_info']['cross_track']
        
        sel = data['selected_mask']
        
        frame_color = 'green' if label == 'Low Error' else 'red'
        sat_side = get_satellite_side(heading, cross_track)
        
        # Range limits - 딱 맞게
        range_min, range_max = np.nanmin(data['range'][sel]), np.nanmax(data['range'][sel])
        az_min, az_max = np.nanmin(data['azimuth'][sel]), np.nanmax(data['azimuth'][sel])
        
        # Left: SWOT Classification - 위성 shading 없이, Range 확장 없이
        ax1 = fig.add_subplot(gs[row, 0])
        
        if sel.sum() > 0:
            ax1.scatter(data['range'][sel], data['azimuth'][sel], c=data['classification'][sel],
                       cmap=ListedColormap(CLASSIFICATION_COLORS), s=8, alpha=0.85,
                       marker='o', vmin=1, vmax=7, edgecolors='none')
        
        # 딱 맞게 (약간의 여백만)
        range_margin = (range_max - range_min) * 0.02
        az_margin = (az_max - az_min) * 0.02
        ax1.set_xlim(range_min - range_margin, range_max + range_margin)
        ax1.set_ylim(az_min - az_margin, az_max + az_margin)
        
        ax1.set_xlabel('Range' if row == 1 else '', fontsize=FONT_SIZE, fontweight='bold')
        ax1.set_ylabel('Azimuth', fontsize=FONT_SIZE, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=FONT_SIZE - 2)
        
        ax1.set_title(f'{label}: PIXC classification ({error:.2f} m)', 
                      fontsize=FONT_SIZE + 2, fontweight='bold', color=frame_color, pad=15)
        ax1.grid(True, alpha=0.3)
        
        for spine in ax1.spines.values():
            spine.set_linewidth(5)
            spine.set_color(frame_color)
        
        # 위성 shading 안 그림!
        
        # Right: NLCD Land Cover
        ax2 = fig.add_subplot(gs[row, 1])
        
        buffer = 0.008
        lon_min, lon_max = bounds[0] - buffer, bounds[2] + buffer
        lat_min, lat_max = bounds[1] - buffer, bounds[3] + buffer
        lat_span = lat_max - lat_min
        
        # xlim 확장 (NLCD만)
        if sat_side == 'left':
            plot_lon_min = lon_min - LON_EXTEND
            plot_lon_max = lon_max
        else:
            plot_lon_min = lon_min
            plot_lon_max = lon_max + LON_EXTEND
        
        plot_lat_min = lat_min - lat_span * 0.08
        plot_lat_max = lat_max + lat_span * 0.08
        
        with rasterio.open(nlcd_path) as src:
            window = src.window(plot_lon_min, plot_lat_min, plot_lon_max, plot_lat_max)
            nlcd_data_arr = src.read(1, window=window)
            unique_vals = np.unique(nlcd_data_arr[nlcd_data_arr > 0])
            extent = [plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max]
            
            nlcd_cmap_colors = ['#FFFFFF'] * 256
            for val, clr in NLCD_COLORS.items():
                nlcd_cmap_colors[val] = clr
            nlcd_cmap = ListedColormap(nlcd_cmap_colors)
            
            ax2.imshow(nlcd_data_arr, extent=extent, cmap=nlcd_cmap,
                      vmin=0, vmax=255, interpolation='nearest')
        
        polygon.boundary.plot(ax=ax2, color=POLYGON_COLOR, linewidth=5)
        
        ax2.set_xlim(plot_lon_min, plot_lon_max)
        ax2.set_ylim(plot_lat_min, plot_lat_max)
        ax2.set_aspect('equal')
        
        ax2.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
        ax2.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
        ax2.set_xlabel('', fontsize=FONT_SIZE, fontweight='bold')
        ax2.set_ylabel('Latitude', fontsize=FONT_SIZE, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=FONT_SIZE - 2)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        ax2.set_title(f'{label}: NLCD land cover ({error:.2f} m)', 
                      fontsize=FONT_SIZE + 2, fontweight='bold', color=frame_color, pad=15)
        ax2.grid(True, alpha=0.3)
        
        for spine in ax2.spines.values():
            spine.set_linewidth(5)
            spine.set_color(frame_color)
        
        # 위성 경로 band 그리기 (NLCD만, 글씨 안쪽으로)
        draw_satellite_swath_lonlat(ax2, sat_side, heading, cycle_id, pass_id, tile_id,
                                     plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max, 
                                     BAND_WIDTH_LON, text_color='white')
    
    # Legends
    ax_leg1 = fig.add_subplot(gs[2, :])
    ax_leg1.axis('off')
    pixc_legend = [Patch(facecolor=CLASSIFICATION_COLORS[i], edgecolor='black',
                         linewidth=0.5, label=CLASSIFICATION_LABELS[i]) for i in range(7)]
    ax_leg1.legend(handles=pixc_legend, loc='center', ncol=7, fontsize=FONT_SIZE - 3,
                   frameon=True, fancybox=True, edgecolor='gray',
                   title='PIXC Classification', title_fontsize=FONT_SIZE,
                   columnspacing=1, handletextpad=0.5)
    
    ax_leg2 = fig.add_subplot(gs[3, :])
    ax_leg2.axis('off')
    if unique_vals is not None:
        nlcd_legend = [Patch(facecolor=NLCD_COLORS[val], edgecolor='black',
                             linewidth=0.5, label=NLCD_LABELS[val])
                       for val in sorted(unique_vals) if val in NLCD_COLORS]
        ax_leg2.legend(handles=nlcd_legend, loc='center', ncol=8, fontsize=FONT_SIZE - 3,
                       frameon=True, fancybox=True, edgecolor='gray',
                       title='NLCD Land Cover', title_fontsize=FONT_SIZE,
                       columnspacing=0.8, handletextpad=0.4)
    
    save_path = f'{output_dir}/fig01_paired_print.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_path}")
    plt.close()

# 실행
process_fctimp('0.06', spike_threshold=0.1832,
               output_dir='/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/figures_print', year=2025)