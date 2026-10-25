#!/usr/bin/env python
"""
Figure 2 (TGRS revision): study-area map, 409 SWORD reaches + 12 sites.

Parameterized rebuild for the double-column layout (Reviewer 1 comment 1,
Reviewer 4 comments 9-10): selectable overview background, adjustable fonts,
two-line inset titles, white legends, bordered overview, 2x6 insets.
Trimmed from the interactive session version; see fig02_study_map.ipynb for
the original notebook.
"""
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.windows import from_bounds
from shapely.ops import nearest_points
from concurrent.futures import ThreadPoolExecutor, as_completed

NLCD_COLORS = {
    11: '#476BA1', 12: '#D1DDF9', 21: '#DDC9C9', 22: '#D89382',
    23: '#ED0000', 24: '#AA0000', 31: '#B2ADA3', 41: '#68AA63',
    42: '#1C6330', 43: '#B5C98E', 52: '#CCBA7C', 71: '#E2E2C1',
    81: '#DDD83E', 82: '#AE7229', 90: '#BBD7ED', 95: '#71A4C1'}
NLCD_LABELS = {
    11: 'Open Water', 12: 'Ice/Snow', 21: 'Developed (Open)',
    22: 'Developed (Low)', 23: 'Developed (Med)',
    24: 'Developed (High)', 31: 'Barren', 41: 'Deciduous Forest',
    42: 'Evergreen Forest', 43: 'Mixed Forest', 52: 'Shrub/Scrub',
    71: 'Grassland', 81: 'Pasture/Hay', 82: 'Cultivated',
    90: 'Woody Wetlands', 95: 'Herbaceous Wetlands'}


def lon_formatter(x, pos):
    return f'{abs(x):.0f}°W' if x < 0 else (f'{x:.0f}°E' if x > 0 else '0°')


def lat_formatter(y, pos):
    return f'{y:.0f}°N' if y > 0 else (f'{abs(y):.0f}°S' if y < 0 else '0°')


def get_nlcd_cmap():
    colors = ['#000000'] * 256
    for v, c in NLCD_COLORS.items():
        colors[v] = c
    return ListedColormap(colors)


def get_overview_cmap(mode):
    base = dict(NLCD_COLORS)
    if mode == 'nlcd_gray_urban':
        base[21] = '#E8E8E8'; base[22] = '#CDCDCD'; base[23] = '#9E9E9E'; base[24] = '#6F6F6F'
    cols = [(0, 0, 0, 0)] * 256
    for v, hx in base.items():
        cols[v] = (int(hx[1:3], 16) / 255, int(hx[3:5], 16) / 255, int(hx[5:7], 16) / 255, 1.0)
    return ListedColormap(cols)


def load_nlcd_fast(nlcd_path, view_bounds, downsample=1):
    with rasterio.open(nlcd_path) as src:
        window = from_bounds(*view_bounds, transform=src.transform)
        if downsample > 1:
            out_shape = (max(1, int(window.height // downsample)),
                         max(1, int(window.width // downsample)))
            data = src.read(1, window=window, out_shape=out_shape,
                            resampling=rasterio.enums.Resampling.nearest)
        else:
            data = src.read(1, window=window)
        b = rasterio.windows.bounds(window, src.transform)
    return data, [b[0], b[2], b[1], b[3]]


def preload_all_nlcd(nlcd_path, bounds_list, downsample=1, max_workers=6):
    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(load_nlcd_fast, nlcd_path, b, downsample): i
                for i, b in enumerate(bounds_list)}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
    return out


def create_paper_figure(
    all_reaches_gdf, selected_gdf, sword_shp_path, usgs_csv_path, nlcd_path,
    save_path=None, dpi=300, figsize=(20, 22),
    overview_bg='nlcd_gray_urban', overview_downsample=16,
    height_ratios=(2.6, 1.0, 1.2, 1.2, 0.22), hspace=0.30, wspace=0.05,
    overview_legend_loc='lower left',
    overview_border_lw=2.0, overview_border_color='black',
    legend_facecolor='white', legend_edgecolor='gray',
    fs_title=24, fs_legend=24, fs_tick=24, fs_site_num=20,
    fs_nlcd_legend=22, fs_site_title=24, fs_data_legend=18, nlcd_legend_ncol=4,
    study_pt_size=60, selected_pt_size=130, gage_marker_size=90,
    legend_ms_study=13, legend_ms_selected=19,
    buffer_lw=3.5, sword_lw=3.5, state_lw=0.9,
    study_color='cyan', selected_color='cyan', state_color=None,
):
    usgs_loc = pd.read_csv(usgs_csv_path, dtype={'gage_id': object}).rename(columns={
        'gage_id': 'site_no', 'gage_latitude': 'dec_lat_va',
        'gage_longitude': 'dec_long_va', 'gage_name': 'station_nm'})
    sword = gpd.read_file(sword_shp_path).to_crs(epsg=4269)
    sword_sel = sword[sword['reach_id'].isin(selected_gdf['reach_id'].tolist())].copy()
    del sword; gc.collect()
    try:
        states = gpd.read_file(
            'https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_state_500k.zip')
        keep = set('WA OR CA NV ID MT WY UT AZ CO NM ND SD NE KS OK TX MN IA MO AR LA WI IL MS '
                   'MI IN KY TN AL OH WV VA NC SC GA FL PA NY VT NH MA RI CT NJ DE MD ME DC'.split())
        states = states[states['STUSPS'].isin(keep)].to_crs(epsg=4269)
    except Exception:
        states = None

    all_reaches_gdf = all_reaches_gdf.to_crs(epsg=4269).copy()
    selected_gdf = selected_gdf.to_crs(epsg=4269).copy()
    if 'site_number' not in selected_gdf.columns:
        selected_gdf['site_number'] = range(1, len(selected_gdf) + 1)
    selected_gdf = selected_gdf.sort_values('site_number').reset_index(drop=True)

    snapped = []
    for _, row in selected_gdf.iterrows():
        m = usgs_loc[usgs_loc['site_no'] == row['site_no']]
        sm = sword_sel[sword_sel['reach_id'] == row['reach_id']]
        if len(m) and len(sm):
            from shapely.geometry import Point
            sp = nearest_points(Point(m.iloc[0]['dec_long_va'], m.iloc[0]['dec_lat_va']),
                                sm.iloc[0].geometry)[1]
            snapped.append({'site_number': row['site_number'], 'x': sp.x, 'y': sp.y})
    gages = pd.DataFrame(snapped)

    bounds_list = []
    for _, row in selected_gdf.iterrows():
        pb = row.geometry.bounds
        mr = max(pb[2] - pb[0], pb[3] - pb[1])
        cx, cy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
        half = mr / 2 + mr * 0.15
        hx = half / np.cos(np.radians(cy))   # square panel in ground distance
        bounds_list.append([cx - hx, cy - half, cx + hx, cy + half])
    tiles = preload_all_nlcd(nlcd_path, bounds_list, downsample=1)

    cmap_nlcd = get_nlcd_cmap()
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(6, 4, figure=fig, height_ratios=list(height_ratios),
                  hspace=hspace, wspace=wspace)

    ax = fig.add_subplot(gs[0, :]); ax.set_facecolor('white')
    vb = [-125, 24, -66, 50]
    ax.set_xlim(vb[0], vb[2]); ax.set_ylim(vb[1], vb[3])
    if overview_bg in ('nlcd', 'nlcd_gray_urban'):
        ov, ext = load_nlcd_fast(nlcd_path, vb, overview_downsample)
        ax.imshow(np.ma.masked_where(ov == 0, ov), cmap=get_overview_cmap(overview_bg),
                  extent=ext, aspect='auto', interpolation='nearest', zorder=1,
                  vmin=0, vmax=255)
        del ov; gc.collect()
        auto_state = 'white'
    else:
        auto_state = '#6e6e6e'
    if states is not None:
        states.boundary.plot(ax=ax, color=state_color or auto_state,
                             linewidth=state_lw, alpha=0.95, zorder=2)
    c = all_reaches_gdf.geometry.centroid
    ax.scatter(c.x.values, c.y.values, c=study_color, s=study_pt_size, alpha=0.95,
               edgecolor='black', linewidth=0.7, zorder=4)
    cen = [(row.geometry.centroid.x, row.geometry.centroid.y, int(row['site_number'])) for _, row in selected_gdf.iterrows()]
    klat = np.cos(np.radians((vb[1] + vb[3]) / 2.0))   # metric scaling of longitude for distance tests
    def dist(p, q): return np.hypot((p[0] - q[0]) * klat, p[1] - q[1])
    stars = [(x, y) for x, y, _ in cen]
    dense = {n: sum(1 for q in stars if dist((x, y), q) < 6.0) for x, y, n in cen}
    order = sorted(cen, key=lambda t: -dense[t[2]])           # place labels in the densest cluster first
    placed = []; label_pos = {}
    cand = [(np.radians(a), r) for r in (2.4, 3.6, 5.0) for a in range(0, 360, 20)]
    for x, y, n in order:
        best, best_score = None, -1e9
        for a, r in cand:
            lx, ly = x + r * np.cos(a) / klat, y + r * np.sin(a)
            if not (vb[0] + 1.5 < lx < vb[2] - 1.5 and vb[1] + 1.5 < ly < vb[3] - 1.0): continue
            d_star = min(dist((lx, ly), q) for q in stars if q != (x, y))
            d_lab = min([dist((lx, ly), q) for q in placed] + [99.0])
            score = min(d_star, d_lab) - 0.15 * r        # keep labels apart, prefer short leaders
            if score > best_score: best, best_score = (lx, ly), score
        placed.append(best); label_pos[n] = best
    for x, y, n in cen:
        ax.scatter(x, y, c=selected_color, s=selected_pt_size, marker='*',
                   edgecolor='black', linewidth=1.1, zorder=5)
        lx, ly = label_pos[n]
        ax.plot([x, lx], [y, ly], color='black', linewidth=1.0, zorder=4, alpha=0.8)
        ax.annotate(str(n), xy=(lx, ly), fontsize=fs_site_num, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='black', linewidth=0.8, alpha=0.95), zorder=6)
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=study_color,
                   markeredgecolor='black', markersize=legend_ms_study,
                   label='Study Reaches (n=409)'),
        plt.Line2D([0], [0], marker='*', linestyle='None', markerfacecolor=selected_color,
                   markeredgecolor='black', markeredgewidth=1.1,
                   markersize=legend_ms_selected, label='Selected Sites (n=12)')],
        loc=overview_legend_loc, fontsize=fs_legend, frameon=True, fancybox=True,
        framealpha=1.0, edgecolor=legend_edgecolor, facecolor=legend_facecolor)
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    ax.tick_params(axis='both', labelsize=fs_tick, width=1.5)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight('bold')
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(overview_border_lw)
        sp.set_edgecolor(overview_border_color); sp.set_zorder(10)
    ax.set_title('(a) Study Reaches (n=409)', fontsize=fs_title, fontweight='bold', pad=15)
    ax.set_aspect(1.0 / np.cos(np.radians((vb[1] + vb[3]) / 2.0)))

    axl = fig.add_subplot(gs[1, :]); axl.axis('off')
    axl.legend(handles=[mpatches.Patch(facecolor=NLCD_COLORS[k], edgecolor='black',
                                       linewidth=0.5, label=NLCD_LABELS[k])
                        for k in sorted(NLCD_LABELS)],
               loc='center', ncol=nlcd_legend_ncol, frameon=True, fancybox=True,
               framealpha=1.0, facecolor=legend_facecolor, edgecolor=legend_edgecolor,
               prop={'weight': 'bold', 'size': fs_nlcd_legend},
               title='NLCD Land Cover',
               title_fontproperties={'weight': 'bold', 'size': fs_nlcd_legend + 2},
               columnspacing=1.0, handletextpad=0.5, labelspacing=0.5, borderpad=0.6)

    labels = 'bcdefghijklm'
    for i, (_, row) in enumerate(selected_gdf.iterrows()):
        axi = fig.add_subplot(gs[(i // 4) + 2, i % 4])
        data, ext = tiles[i]
        axi.imshow(data, cmap=cmap_nlcd, extent=ext, aspect='auto',
                   interpolation='nearest', zorder=1, vmin=0, vmax=255)
        axi.set_xlim(bounds_list[i][0], bounds_list[i][2])
        axi.set_ylim(bounds_list[i][1], bounds_list[i][3])
        selected_gdf[selected_gdf['site_number'] == row['site_number']].plot(
            ax=axi, facecolor='none', edgecolor='black', linewidth=buffer_lw, zorder=3)
        sm = sword_sel[sword_sel['reach_id'] == row['reach_id']]
        if len(sm):
            sm.plot(ax=axi, color='cyan', linewidth=sword_lw, zorder=4)
        g = gages[gages['site_number'] == row['site_number']]
        if len(g):
            axi.scatter(g.x, g.y, color='black', s=gage_marker_size, zorder=6,
                        edgecolor='white', linewidth=2)
        axi.set_title(f"({labels[i]}) S{int(row['site_number'])}: "
                      f"{row['fctimp_mean'] * 100:.1f}%",
                      fontsize=fs_site_title, fontweight='bold', pad=5, linespacing=1.1,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                alpha=0.9, edgecolor='none'))
        axi.set_aspect(1.0 / np.cos(np.radians((bounds_list[i][1] + bounds_list[i][3]) / 2.0))); axi.axis('off')

    axd = fig.add_subplot(gs[5, :]); axd.axis('off')
    axd.legend(handles=[
        mpatches.Patch(facecolor='none', edgecolor='black', linewidth=buffer_lw,
                       label='Reach Buffer Zones'),
        plt.Line2D([0], [0], color='cyan', linewidth=4, label='SWOT Reaches'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markeredgecolor='white', markersize=12, label='USGS Gages',
                   linestyle='None')],
        loc='center', ncol=3, frameon=True, fancybox=True, framealpha=1.0,
        facecolor=legend_facecolor, edgecolor=legend_edgecolor,
        columnspacing=1.6, prop={'weight': 'bold', 'size': fs_data_legend})

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        fig.savefig(save_path.replace('.png', '.pdf'), dpi=dpi,
                    bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    return fig


if __name__ == '__main__':
    W = '/work/pi_kandread_umass_edu/swot-urban'
    all_reaches = gpd.read_parquet(f'{W}/2_urban_reaches/all_reaches_vD.parquet')
    selected_12 = gpd.read_parquet(f'{W}/2_urban_reaches/selected_12_vD.parquet')
    create_paper_figure(
        all_reaches, selected_12,
        sword_shp_path=f'{W}/1_data_processing/reaches_v1.shp',
        usgs_csv_path=f'{W}/0_data_collection/5_sword_usgs/paired_reach_SWOT_gage.csv',
        nlcd_path=f'{W}/0_data_collection/1_nlcd/Annual_NLCD_LndCov_2024_CU_C1V1_EPSG4326.tif',
        save_path='/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/figures_print/fig02_study_map_epsg4269_v2.png', overview_bg='nlcd', dpi=600,
        selected_color='magenta', selected_pt_size=560, legend_ms_selected=26,
        fs_nlcd_legend=20, nlcd_legend_ncol=4, fs_data_legend=20, fs_site_title=24,
        height_ratios=(11.2, 2.6, 4.7, 4.7, 4.7, 1.0), hspace=0.32, figsize=(20, 31))
