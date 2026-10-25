### import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Polygon as MplPolygon, Circle
from matplotlib.ticker import FuncFormatter
import geopandas as gpd
import contextily as cx
from pathlib import Path
from shapely.geometry import box
import rasterio

# 1. Reach 로드 (ISA 0.06)
selected_12 = gpd.read_file('/work/pi_kandread_umass_edu/swot-urban/2_urban_reaches/selected_12_vD.shp')
fctimp_str = '0.06'
sample = selected_12[np.isclose(selected_12["fctimp_mea"], float(fctimp_str), atol=0.01)].copy()

# 4326 좌표
sample_4326 = sample.to_crs('EPSG:4326')
bounds_4326 = sample_4326.total_bounds

# 양쪽으로 넓히기
lon_buffer = 0.1
lat_buffer = 0.005

expanded_bounds_4326 = [
    bounds_4326[0] - lon_buffer,
    bounds_4326[1] - lat_buffer,
    bounds_4326[2] + lon_buffer,
    bounds_4326[3] + lat_buffer
]

# 3857로 변환
expanded_box = gpd.GeoDataFrame(
    geometry=[box(*expanded_bounds_4326)], 
    crs='EPSG:4326'
).to_crs('EPSG:3857')
expanded_bounds_3857 = expanded_box.total_bounds
sample_3857 = sample.to_crs('EPSG:3857')

# 2. NLCD 데이터 로드
nlcd_path = '/work/pi_kandread_umass_edu/swot-urban/0_data_collection/1_nlcd/Annual_NLCD_LndCov_2024_CU_C1V1_EPSG4326.tif'

with rasterio.open(nlcd_path) as src:
    window = src.window(*expanded_bounds_4326)
    nlcd_data = src.read(1, window=window)
    nlcd_bounds = rasterio.windows.bounds(window, src.transform)

# NLCD bounds를 3857로 변환
nlcd_box_3857 = gpd.GeoDataFrame(
    geometry=[box(nlcd_bounds[0], nlcd_bounds[1], nlcd_bounds[2], nlcd_bounds[3])],
    crs='EPSG:4326'
).to_crs('EPSG:3857')
nlcd_extent_3857 = nlcd_box_3857.total_bounds

# 3. NLCD 오버레이 이미지 생성 (RGBA)
nlcd_overlay = np.zeros((*nlcd_data.shape, 4), dtype=np.float32)

developed_mask = np.isin(nlcd_data, [22, 23, 24])
nlcd_overlay[developed_mask] = [1.0, 0.0, 0.0, 0.3]

water_mask = nlcd_data == 11
nlcd_overlay[water_mask] = [0.0, 0.4, 1.0, 0.7]

other_mask = (nlcd_data > 0) & ~developed_mask & ~water_mask
nlcd_overlay[other_mask] = [0.5, 0.5, 0.5, 0.7]

# 4. 3857 좌표를 4326으로 변환하는 함수 (축 라벨용)
from pyproj import Transformer

transformer_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

def lon_formatter_3857(x, pos):
    lon, lat = transformer_to_4326.transform(x, 0)
    if lon < 0:
        return f'{abs(lon):.2f}°W'
    else:
        return f'{lon:.2f}°E'

def lat_formatter_3857(y, pos):
    lon, lat = transformer_to_4326.transform(0, y)
    if lat > 0:
        return f'{lat:.2f}°N'
    else:
        return f'{abs(lat):.2f}°S'


# 5. 위성 궤도 그리기 함수
def draw_satellite_path(ax, x_min, x_max, y_min, y_max, heading, cycle_id, pass_id, tile_id, 
                        position='left', color='#303030', alpha=0.8):
    going_north = (0 <= heading < 90) or (270 <= heading < 360)
    
    x_span = x_max - x_min
    y_span = y_max - y_min
    band_width = x_span * 0.06
    
    if position == 'left':
        center_x = x_min + x_span * 0.15
    else:
        center_x = x_max - x_span * 0.15
    
    x_offset = y_span * 0.08
    
    if going_north:
        points = [
            (center_x - band_width/2 - x_offset, y_min),
            (center_x + band_width/2 - x_offset, y_min),
            (center_x + band_width/2 + x_offset, y_max),
            (center_x - band_width/2 + x_offset, y_max),
        ]
    else:
        points = [
            (center_x - band_width/2 + x_offset, y_min),
            (center_x + band_width/2 + x_offset, y_min),
            (center_x + band_width/2 - x_offset, y_max),
            (center_x - band_width/2 - x_offset, y_max),
        ]
    
    band_patch = MplPolygon(points, facecolor=color, edgecolor='black', 
                             alpha=alpha, linewidth=1.5, zorder=6)
    ax.add_patch(band_patch)
    
    # 텍스트 회전 (경로 방향)
    if going_north:
        rotation = 76
    else:
        rotation = -76
    
    text_y_center = (y_min + y_max) / 2
    text_spacing = y_span * 0.10
    text_size = 10
    
    # 위성 위치 (흰색 원)
    if going_north:
        sat_x = center_x + x_offset * 0.5
        sat_y = text_y_center + y_span * 0.25
    else:
        sat_x = center_x - x_offset * 0.5
        sat_y = text_y_center + y_span * 0.25
    
    satellite_circle = Circle((sat_x, sat_y), radius=x_span * 0.0012, 
                               facecolor='white', edgecolor='black', 
                               linewidth=2, zorder=8)
    ax.add_patch(satellite_circle)
    
    # ax.text(sat_x, sat_y - y_span * 0.06, 'Path of\nSatellite', 
    #         fontsize=text_size, fontweight='bold', color='white', 
    #         ha='center', va='top', rotation=rotation, zorder=7)
    
    # info_y = text_y_center - y_span * 0.15
    # ax.text(center_x, info_y, f'Tile: {tile_id}', 
    #         fontsize=text_size, fontweight='bold', color='white', 
    #         ha='center', va='center', rotation=rotation, zorder=7)
    # ax.text(center_x, info_y - text_spacing * 1.2, f'Pass: {pass_id}', 
    #         fontsize=text_size, fontweight='bold', color='white', 
    #         ha='center', va='center', rotation=rotation, zorder=7)
    # ax.text(center_x, info_y - text_spacing * 2.4, f'Cycle: {cycle_id}', 
    #         fontsize=text_size, fontweight='bold', color='white', 
    #         ha='center', va='center', rotation=rotation, zorder=7)
    
    # if going_north:
    #     ax.text(center_x + x_offset, y_max - y_span * 0.03, f'{heading:.0f}°', 
    #             fontsize=text_size + 2, fontweight='bold', color='white', 
    #             ha='center', va='top', rotation=rotation, zorder=7)
    # else:
    #     ax.text(center_x - x_offset, y_min + y_span * 0.03, f'{heading:.0f}°', 
    #             fontsize=text_size + 2, fontweight='bold', color='white', 
    #             ha='center', va='bottom', rotation=rotation, zorder=7)


# 6. 플롯
fig, ax = plt.subplots(figsize=(18, 14))

ax.set_xlim(expanded_bounds_3857[0], expanded_bounds_3857[2])
ax.set_ylim(expanded_bounds_3857[1], expanded_bounds_3857[3])

# 1) ESRI World Imagery 배경
cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zoom=16)

# 2) NLCD 오버레이
ax.imshow(nlcd_overlay, 
          extent=[nlcd_extent_3857[0], nlcd_extent_3857[2], nlcd_extent_3857[1], nlcd_extent_3857[3]],
          origin='upper', zorder=2, interpolation='nearest')

# 3) Reach 경계
sample_3857.boundary.plot(ax=ax, color='black', linewidth=4, zorder=5)

# 4) 위성 궤도 두 개
draw_satellite_path(ax, expanded_bounds_3857[0], expanded_bounds_3857[2],
                    expanded_bounds_3857[1], expanded_bounds_3857[3],
                    heading=166, cycle_id='039', pass_id='272', tile_id='082R',
                    position='right', color='teal', alpha=0.5)

draw_satellite_path(ax, expanded_bounds_3857[0], expanded_bounds_3857[2],
                    expanded_bounds_3857[1], expanded_bounds_3857[3],
                    heading=14, cycle_id='039', pass_id='315', tile_id='227R',
                    position='left', color='coral', alpha=0.5)

# 5) 축 라벨 (E/W/N/S 형식)
ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter_3857))
ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter_3857))
ax.tick_params(axis='both', labelsize=12)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')

# ax.set_title(f'ISA = {fctimp_str} | Urban (Red) | Water (Blue) | Other (Gray)', 
#              fontsize=16, fontweight='bold')

# 범례
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch

legend_elements = [
    Line2D([0], [0], color='coral', linewidth=3, marker='>', label='Measurement Path'),
    Line2D([0], [0], color='teal', linewidth=3, marker='>', label='Measurement Path'),
    Patch(facecolor='none', edgecolor='cyan', linewidth=2, label='Reach Buffer Zone'),
    Line2D([0], [0], color='black', linewidth=2, marker='>', markersize=8,
       label='Satellite Path'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
       markeredgecolor='black', markersize=10, label='SWOT Node'),
    Patch(facecolor='red', alpha=0.35, edgecolor='black', label='Developed (ISA)'),
    Patch(facecolor='blue', alpha=0.4, edgecolor='black', label='Open Water'),
    Patch(facecolor='gray', alpha=0.5, edgecolor='black', label='Other Land Cover'),
]

ax.legend(handles=legend_elements, loc='lower center', fontsize=14, bbox_to_anchor=(0.3, 0.05))

plt.savefig('figures/money_figure_with_paths.png', dpi=600, bbox_inches='tight')
plt.show()
print("✓ Saved: figures/money_figure_with_paths.png")