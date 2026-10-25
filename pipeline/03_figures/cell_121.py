import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, LineString
import rasterio
import matplotlib.pyplot as plt

BASE_PATH = Path('/work/pi_kandread_umass_edu/swot-urban')
PARQUET_PATH = BASE_PATH / '5_fctimp'
TVP_PATH = BASE_PATH / '6_pixc_tvp' / 'SWOT_TVP_Data'
POLYGON_PATH = BASE_PATH / '2_urban_reaches' / 'all_reaches_vD.parquet'
NLCD_FCTIMP_PATH = str(BASE_PATH / '0_data_collection/1_nlcd/Annual_NLCD_FctImp_2024_CU_C1V1_EPSG4326.tif')
BUILDING_HEIGHT_PATH = str(BASE_PATH / '0_data_collection/3_150m_building/GBH2020_150m_GEDI_4326.tif')

J2000_EPOCH = pd.Timestamp('2000-01-01 00:00:00')
TARGET_REACH_ID = 74282700231

# =============================================================================
# Sampling Function
# =============================================================================
def sample_raster_along_line(line, node_pt, raster_path, n_samples=100, is_building=False):
    empty_result = {'mean_val': 0.0, 'nonzero_ratio': 0.0, 'dw_mean_val': 0.0, 'dw_ratio': 0.0, 'values': [], 'coords': []}
    
    if line is None or line.is_empty:
        return empty_result
    
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
    
    start_pt = Point(target_line.coords[0])
    end_pt = Point(target_line.coords[-1])
    reverse = node_pt.distance(start_pt) > node_pt.distance(end_pt)
    
    values = []
    coords_sampled = []
    
    with rasterio.open(raster_path) as src:
        for i in range(n_samples + 1):
            t = 1.0 - (i / n_samples) if reverse else i / n_samples
            pt = target_line.interpolate(t, normalized=True)
            coords_sampled.append((pt.x, pt.y))
            
            try:
                row, col = src.index(pt.x, pt.y)
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                    
                    if is_building:
                        if val >= 0 and not np.isnan(val):
                            values.append((i, float(val)))
                    else:
                        if 0 <= val <= 100:
                            values.append((i, float(val)))
            except:
                pass
    
    if not values:
        return empty_result
    
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
        'mean_val': mean_val,
        'nonzero_ratio': nonzero_ratio,
        'dw_mean_val': dw_mean_val,
        'dw_ratio': dw_ratio,
        'values': values,
        'coords': coords_sampled,
    }


# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("=" * 70)
print(f"v6.4 Near/Far ISA 검증 - Reach {TARGET_REACH_ID}")
print("=" * 70)

# 높은 fctimp 파일들 모두 로드
print("\n[1] Node 파일 로드...")
all_node_files = list(PARQUET_PATH.glob('fctimp_mean_0.605247_node_*.parquet'))
print(f"  Found {len(all_node_files)} files with fctimp=0.605")

node_dfs = []
for nf in all_node_files:
    df = pd.read_parquet(nf)
    if int(df['reach_id'].iloc[0]) == TARGET_REACH_ID:
        node_dfs.append(df)

node_df = pd.concat(node_dfs, ignore_index=True)
print(f"  Total rows: {len(node_df)}")
print(f"  fctimp_mean: {node_df['fctimp_mean'].iloc[0]:.4f}")

# Polygon
print("\n[2] Polygon 로드...")
polygons_gdf = gpd.read_parquet(POLYGON_PATH)
if polygons_gdf.crs != 'EPSG:4326':
    polygons_gdf = polygons_gdf.to_crs('EPSG:4326')
polygon = polygons_gdf[polygons_gdf['reach_id'] == TARGET_REACH_ID].geometry.iloc[0]
print(f"  Polygon bounds: {[round(x, 4) for x in polygon.bounds]}")

# TVP
print("\n[3] TVP 로드...")
tvp_file = TVP_PATH / f'tvp_reach_{TARGET_REACH_ID}.parquet'
tvp_df = pd.read_parquet(tvp_file)
print(f"  TVP rows: {len(tvp_df)}")

# =============================================================================
# 2. 시간 매칭
# =============================================================================
print("\n[4] TVP 매칭...")
node_df['time_dt'] = pd.to_datetime(node_df['time_str']).dt.tz_localize(None)
tvp_df['time_dt'] = J2000_EPOCH + pd.to_timedelta(tvp_df['time'], unit='s')

node_df = node_df.sort_values('time_dt').reset_index(drop=True)
tvp_df = tvp_df.sort_values('time_dt')

merged_df = pd.merge_asof(
    node_df,
    tvp_df[['time_dt', 'latitude', 'longitude', 'altitude']],
    on='time_dt',
    direction='nearest',
    tolerance=pd.Timedelta('30s')
).rename(columns={'latitude': 'sat_lat', 'longitude': 'sat_lon', 'altitude': 'sat_altitude'})

matched = merged_df['sat_lon'].notna().sum()
print(f"  TVP 매칭: {matched}/{len(merged_df)} ({100*matched/len(merged_df):.1f}%)")

if matched == 0:
    print("\n  ⚠️ 매칭 안됨! 시간 범위 확인:")
    print(f"    Node: {node_df['time_dt'].min()} ~ {node_df['time_dt'].max()}")
    print(f"    TVP:  {tvp_df['time_dt'].min()} ~ {tvp_df['time_dt'].max()}")
else:
    # =============================================================================
    # 3. Near/Far ISA 계산
    # =============================================================================
    print("\n[5] Near/Far ISA 계산...")
    
    test_row = merged_df[merged_df['sat_lon'].notna()].iloc[0]
    print(f"  Node: ({test_row['lon']:.6f}, {test_row['lat']:.6f})")
    print(f"  Sat:  ({test_row['sat_lon']:.6f}, {test_row['sat_lat']:.6f})")
    print(f"  Time: {test_row['time_dt']}")
    
    node_pt = Point(test_row['lon'], test_row['lat'])
    sat_lon, sat_lat = test_row['sat_lon'], test_row['sat_lat']
    
    dx = sat_lon - test_row['lon']
    dy = sat_lat - test_row['lat']
    length = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / length, dy / length
    
    extend = 0.2
    
    near_line = LineString([node_pt, Point(test_row['lon'] + ux * extend, test_row['lat'] + uy * extend)])
    far_line = LineString([node_pt, Point(test_row['lon'] - ux * extend, test_row['lat'] - uy * extend)])
    
    near_in_poly = near_line.intersection(polygon) if near_line.intersects(polygon) else None
    far_in_poly = far_line.intersection(polygon) if far_line.intersects(polygon) else None
    
    print(f"\n  Near line: {near_in_poly.geom_type if near_in_poly and not near_in_poly.is_empty else 'None'}")
    print(f"  Far line: {far_in_poly.geom_type if far_in_poly and not far_in_poly.is_empty else 'None'}")
    
    # ISA 샘플링
    near_isa = sample_raster_along_line(near_in_poly, node_pt, NLCD_FCTIMP_PATH)
    far_isa = sample_raster_along_line(far_in_poly, node_pt, NLCD_FCTIMP_PATH)
    
    print(f"\n  === Near ISA ===")
    print(f"    mean_val: {near_isa['mean_val']:.2f}")
    print(f"    nonzero_ratio: {near_isa['nonzero_ratio']:.2f}")
    print(f"    dw_mean_val: {near_isa['dw_mean_val']:.2f}")
    print(f"    샘플 수: {len(near_isa['values'])}")
    if near_isa['values']:
        vals = [v for _, v in near_isa['values'][:10]]
        print(f"    처음 10개 값: {vals}")
    
    print(f"\n  === Far ISA ===")
    print(f"    mean_val: {far_isa['mean_val']:.2f}")
    print(f"    nonzero_ratio: {far_isa['nonzero_ratio']:.2f}")
    print(f"    dw_mean_val: {far_isa['dw_mean_val']:.2f}")
    print(f"    샘플 수: {len(far_isa['values'])}")
    if far_isa['values']:
        vals = [v for _, v in far_isa['values'][:10]]
        print(f"    처음 10개 값: {vals}")
    
    # =============================================================================
    # 4. 시각화
    # =============================================================================
    print("\n[6] 시각화...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    if polygon.geom_type == 'MultiPolygon':
        for poly in polygon.geoms:
            x, y = poly.exterior.xy
            ax1.fill(x, y, alpha=0.3, color='blue')
            ax1.plot(x, y, color='blue', linewidth=1)
    else:
        x, y = polygon.exterior.xy
        ax1.fill(x, y, alpha=0.3, color='blue')
        ax1.plot(x, y, color='blue', linewidth=1)
    
    ax1.scatter(test_row['lon'], test_row['lat'], color='green', s=100, zorder=5, label='Node')
    ax1.scatter(sat_lon, sat_lat, color='red', s=50, marker='^', label='Satellite')
    ax1.annotate('', xy=(sat_lon, sat_lat), xytext=(test_row['lon'], test_row['lat']),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    if near_in_poly and not near_in_poly.is_empty:
        if near_in_poly.geom_type == 'LineString':
            x, y = near_in_poly.xy
            ax1.plot(x, y, color='orange', linewidth=3, label=f'Near (ISA={near_isa["mean_val"]:.1f})')
        elif near_in_poly.geom_type == 'MultiLineString':
            for seg in near_in_poly.geoms:
                x, y = seg.xy
                ax1.plot(x, y, color='orange', linewidth=3)
            ax1.plot([], [], color='orange', linewidth=3, label=f'Near (ISA={near_isa["mean_val"]:.1f})')
    
    if far_in_poly and not far_in_poly.is_empty:
        if far_in_poly.geom_type == 'LineString':
            x, y = far_in_poly.xy
            ax1.plot(x, y, color='purple', linewidth=3, label=f'Far (ISA={far_isa["mean_val"]:.1f})')
        elif far_in_poly.geom_type == 'MultiLineString':
            for seg in far_in_poly.geoms:
                x, y = seg.xy
                ax1.plot(x, y, color='purple', linewidth=3)
            ax1.plot([], [], color='purple', linewidth=3, label=f'Far (ISA={far_isa["mean_val"]:.1f})')
    
    ax1.set_title(f'Reach {TARGET_REACH_ID} (fctimp={test_row["fctimp_mean"]:.1%})')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend(loc='best')
    ax1.set_aspect('equal')
    
    ax2 = axes[1]
    if near_isa['values']:
        near_idx = [i for i, _ in near_isa['values']]
        near_vals = [v for _, v in near_isa['values']]
        ax2.plot(near_idx, near_vals, 'o-', color='orange', label=f'Near (mean={near_isa["mean_val"]:.1f})')
    
    if far_isa['values']:
        far_idx = [i for i, _ in far_isa['values']]
        far_vals = [v for _, v in far_isa['values']]
        ax2.plot(far_idx, far_vals, 's-', color='purple', label=f'Far (mean={far_isa["mean_val"]:.1f})')
    
    ax2.set_xlabel('Sample Index (0 = closest to node)')
    ax2.set_ylabel('ISA Value (%)')
    ax2.set_title('ISA Values Along Near/Far Lines')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 결론
    print("\n" + "=" * 70)
    print("검증 결과")
    print("=" * 70)
    print(f"  fctimp_mean (원본): {test_row['fctimp_mean']:.1%}")
    print(f"  Near ISA: {near_isa['mean_val']:.2f} (dw: {near_isa['dw_mean_val']:.2f})")
    print(f"  Far ISA:  {far_isa['mean_val']:.2f} (dw: {far_isa['dw_mean_val']:.2f})")
    
    if near_isa['mean_val'] > 0 or far_isa['mean_val'] > 0:
        print("\n✅ ISA 값이 정상적으로 계산됨!")
    else:
        print("\n⚠️ ISA 값이 0 - 확인 필요")