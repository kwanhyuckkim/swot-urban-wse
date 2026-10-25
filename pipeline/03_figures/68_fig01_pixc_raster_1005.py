#!/usr/bin/env python
"""원본 디자인 5종(Fig 1,4,6,7,8)을 인쇄 규격으로 재현.
2026-10-05 copy of 32 (user: "PIXC 패널 다시 그리기"): fig01_orig draws the PIXC classes as a filled range-azimuth grid.
Run with argument fig01_orig; output fig01_orig_print_1005.png/.pdf.

원본 노트북 셀 23/84/172/182/213의 시각 언어를 그대로 유지하고, 크기·글꼴만
2단 인쇄 폭(7.16 in, 글꼴 6-9 pt)으로 재조정한다. 주석 수치는 전부 데이터에서
재계산되므로 리뷰어가 요구한 수치 정합(Fig6 내장표, Fig7 0.49-13.05,
Fig8 0.50-0.97)이 자동 충족된다. Fig1은 (a)/(b) 라벨과 제품명 제목(R3/R4),
레전드 공백 축소를 반영한다.
"""
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Polygon as MplPolygon
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import timedelta
import xarray as xr
import rasterio
from shapely.geometry import Point

WORK = "/work/pi_kandread_umass_edu/swot-urban"
PROJ = "/project/pi_kandread_umass_edu/swot-urban"
OUT = f"{PROJ}/3_analysis_results/figures_print"
Path(OUT).mkdir(exist_ok=True)
T = 0.1832

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def get_flow_angle_zone(fa):
    fa = fa % 360
    if (fa >= 350 or fa <= 10) or (170 <= fa <= 190):
        return 'Parallel'
    elif (80 <= fa <= 100) or (260 <= fa <= 280):
        return 'Perpendicular'
    return 'Other'


def load_final_df():
    cols = ["flow_angle", "far_isa", "near_isa", "far_building", "near_building",
            "error_node", "sample_geometry_category", "far_ratio", "near_ratio"]
    import pyarrow.parquet as pq
    schema = pq.read_schema(f"{PROJ}/3_analysis_results/swot_urban_nearfar_v6.5_results.parquet").names
    cols = [c for c in cols if c in schema]
    d = pq.read_table(f"{PROJ}/3_analysis_results/swot_urban_nearfar_v6.5_results.parquet",
                      columns=cols).to_pandas()
    d["abs_error_node"] = d.error_node.abs()
    for c in ("far_isa", "near_isa"):
        if d[c].max() <= 1.5:
            d[c] = d[c] * 100
    gmap = {"na": "Neither", "near": "Near", "far": "Far", "near/far": "Both"}
    d["geometry_category"] = d.sample_geometry_category.astype(str).str.lower().map(gmap)
    for c in ("far_ratio", "near_ratio"):
        if c not in d.columns:
            d[c] = 1.0
    return d


# ═══════════════════════════════════════════════════════ Fig 1 (cell 23)
CLS_COLORS = ['maroon', 'gold', 'skyblue', 'mediumblue', 'dimgray', 'cyan', 'darkturquoise']
CLS_LABELS = ['Land', 'Land (near water)', 'Water (near land)', 'Open Water',
              'Dark Water', 'Low Coh. (near land)', 'Low Coh. (open)']
NLCD_COLORS = {11: '#476BA1', 12: '#D1DDF9', 21: '#DDC9C9', 22: '#D89382', 23: '#ED0000',
               24: '#AA0000', 31: '#B2ADA3', 41: '#68AA63', 42: '#1C6330', 43: '#B5C98E',
               52: '#CCBA7C', 71: '#E2E2C1', 81: '#DDD83E', 82: '#AE7229', 90: '#BBD7ED',
               95: '#71A4C1'}
NLCD_LABELS = {11: 'Open Water', 12: 'Ice/Snow', 21: 'Developed (Open)', 22: 'Developed (Low)',
               23: 'Developed (Med)', 24: 'Developed (High)', 31: 'Barren', 41: 'Deciduous Forest',
               42: 'Evergreen Forest', 43: 'Mixed Forest', 52: 'Shrub/Scrub', 71: 'Grassland',
               81: 'Pasture/Hay', 82: 'Cultivated', 90: 'Woody Wetlands', 95: 'Herbaceous Wetlands'}


def _best(files):
    fd = {}
    for f in files:
        m = re.match(r'(.+)_(P[IG]D\d)_(\d+)\.nc', f.name)
        if not m:
            continue
        b, pr, c = m.group(1), m.group(2), int(m.group(3))
        if b not in fd or (pr == 'PGD0' and fd[b][1] == 'PID0') or \
           (pr == fd[b][1] and c > fd[b][2]):
            fd[b] = (f, pr, c)
    return [v[0] for v in fd.values()]


def _extract(pixc, pixcvec, polygon_gdf):
    ds = xr.open_dataset(pixc, group='pixel_cloud')
    lon, lat = ds['longitude'].values, ds['latitude'].values
    b = polygon_gdf.total_bounds
    tvp = xr.open_dataset(pixc, group='tvp')
    heading = np.nanmedian(tvp['velocity_heading'].values)
    tvp.close()
    bbox = (lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3])
    if not bbox.sum():
        ds.close(); return None
    cls = ds['classification'].values[bbox]
    wat = cls > 2
    if not wat.sum():
        ds.close(); return None
    lo, la = lon[bbox][wat], lat[bbox][wat]
    poly = polygon_gdf.geometry.iloc[0]
    import shapely
    ins = shapely.contains_xy(poly, lo, la)
    if not ins.sum():
        ds.close(); return None
    idx = np.where(bbox)[0][wat][ins]
    ct = np.nanmedian(ds['cross_track'].values[idx])
    dv = xr.open_dataset(pixcvec)
    rid, nid = dv['reach_id'].values[idx], dv['node_id'].values[idx]
    oid, lid = dv['obs_id'].values[idx], dv['lake_id'].values[idx]

    def valid(a):
        return np.array([len(x) > 0 and x != b'' for x in a]) if a.dtype.kind == 'S' else a > 0
    sel = valid(rid) & valid(nid) & ~valid(oid) & ~valid(lid)
    out = dict(cls=ds['classification'].values[idx], az=ds['azimuth_index'].values[idx],
               rg=ds['range_index'].values[idx], sel=sel, heading=heading, cross=ct)
    ds.close(); dv.close()
    return out


def _sat_side(heading, cross):
    n = (0 <= heading < 90) or (270 <= heading < 360)
    return ('left' if cross >= 0 else 'right') if n else ('right' if cross >= 0 else 'left')


def _band_lonlat(ax, side, heading, cyc, pas, tile, x0, x1, y0, y1, bw, fs):
    north = (0 <= heading < 90) or (270 <= heading < 360)
    cx = x0 + bw / 2 if side == 'left' else x1 - bw / 2
    ys = y1 - y0
    off = ys * 0.15 * 0.3
    pts = ([(cx - bw/2 - off, y0), (cx + bw/2 - off, y0), (cx + bw/2 + off, y1), (cx - bw/2 + off, y1)]
           if north else
           [(cx - bw/2 + off, y0), (cx + bw/2 + off, y0), (cx + bw/2 - off, y1), (cx - bw/2 - off, y1)])
    ax.add_patch(MplPolygon(pts, facecolor='#505050', edgecolor='#404040', alpha=0.75,
                            linewidth=0.6, zorder=3))
    tx = cx + bw * 0.15 if side == 'left' else cx - bw * 0.15
    tyc = (y0 + y1) / 2
    sp = ys * 0.10
    for dy, txt in ((sp, f'Tile: {tile}'), (0, f'Pass: {pas}'), (-sp, f'Cycle: {cyc}')):
        ax.text(tx, tyc + dy, txt, fontsize=fs, fontweight='bold', color='white',
                ha='center', va='center', zorder=5)
    if north:
        ax.text(tx, y1 - ys * 0.05, f'{heading:.0f}°', fontsize=fs + 0.5, fontweight='bold',
                color='white', ha='center', va='top', zorder=5)
    else:
        ax.text(tx, y0 + ys * 0.05, f'{heading:.0f}°', fontsize=fs + 0.5, fontweight='bold',
                color='white', ha='center', va='bottom', zorder=5)


def fig01_orig():
    FS = 8
    sites = gpd.read_file(f"{WORK}/2_urban_reaches/selected_12_vD.shp")
    col = "fctimp_mea" if "fctimp_mea" in sites.columns else "fctimp_mean"
    sample = sites[np.isclose(sites[col], 0.06, atol=0.01)].to_crs("EPSG:4326")
    poly = sample
    bounds = sample.total_bounds
    obs = pd.read_parquet(f"{WORK}/1_data_processing/fctimp_mean_0.06_sample.parquet")
    obs["error"] = obs.wse_relative_m - obs.wse_usgs_relative_m
    obs["t"] = pd.to_datetime(obs.time_str).dt.tz_localize(None)
    pixc = _best(list(Path(f"{PROJ}/0_data_collection/SWOT_PIXC/fctimp_0.06").glob("SWOT_L2_HR_PIXC_*.nc")))
    pvec = {f.name: f for f in _best(list(Path(f"{PROJ}/0_data_collection/SWOT_PIXCVec/fctimp_0.06").glob("SWOT_L2_HR_PIXCVec_*.nc")))}

    picks = {}
    for f in pixc:
        parts = f.name.split('_')
        cyc, pas, tile = parts[4], parts[5], parts[6]
        if (cyc, pas, tile) in [("039", "272", "082R"), ("039", "315", "227R")]:
            v = pvec.get(f.name.replace('PIXC_', 'PIXCVec_'))
            if v is not None:
                picks[pas] = (f, v, cyc, tile)
    assert set(picks) == {"272", "315"}, picks

    nlcd_path = f"{WORK}/0_data_collection/1_nlcd/Annual_NLCD_LndCov_2024_CU_C1V1_EPSG4326.tif"
    LON_EXTEND, BW = 0.06, 0.035

    fig = plt.figure(figsize=(7.16, 6.15))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.52], hspace=0.20, wspace=0.24,
                  left=0.075, right=0.985, top=0.95, bottom=0.012)
    unique_vals = None
    for row, pas in enumerate(["272", "315"]):
        f, v, cyc, tile = picks[pas]
        t0 = pd.to_datetime(f.name.split('_')[7], format="%Y%m%dT%H%M%S")
        near = obs.loc[(obs.t - t0).abs().dt.total_seconds().idxmin()]
        err = float(near.error)
        data = _extract(f, v, sample)
        heading = data['heading']
        fc = 'green' if row == 0 else 'red'
        side = _sat_side(heading, data['cross'])
        sel = data['sel']
        rmin, rmax = np.nanmin(data['rg'][sel]), np.nanmax(data['rg'][sel])
        amin, amax = np.nanmin(data['az'][sel]), np.nanmax(data['az'][sel])

        ax1 = fig.add_subplot(gs[row, 0])
        # 2026-10-05: one filled cell per PIXC pixel (range x azimuth grid) instead of tiny translucent dots, so the
        # displayed colours are the legend colours (Reviewer 4 comment 14); classes drawn in increasing order as before.
        rgi = np.rint(data['rg'][sel]).astype(int); azi = np.rint(data['az'][sel]).astype(int)
        cli = np.asarray(data['cls'][sel]).astype(int)
        r0, a0 = rgi.min(), azi.min()
        grid = np.zeros((azi.max() - a0 + 1, rgi.max() - r0 + 1), dtype=np.int16)
        order = np.argsort(cli, kind='stable')
        grid[azi[order] - a0, rgi[order] - r0] = cli[order]
        ax1.imshow(np.ma.masked_equal(grid, 0), cmap=ListedColormap(CLS_COLORS), vmin=0.5, vmax=7.5,
                   origin='lower', interpolation='nearest', aspect='auto',
                   extent=(r0 - 0.5, rgi.max() + 0.5, a0 - 0.5, azi.max() + 0.5), zorder=1)
        print(f"fig01 pass {pas}: {len(cli):,} pixels, grid {grid.shape}, classes "
              f"{dict(zip(*np.unique(cli, return_counts=True)))}")
        rm, am = (rmax - rmin) * 0.02, (amax - amin) * 0.02
        ax1.set_xlim(rmin - rm, rmax + rm); ax1.set_ylim(amin - am, amax + am)
        if row == 1:
            ax1.set_xlabel('Range', fontsize=FS, fontweight='bold')
        ax1.set_ylabel('Azimuth', fontsize=FS, fontweight='bold')
        ax1.tick_params(labelsize=FS - 1.5, labelbottom=(row == 1))
        ax1.set_title(f'SWOT L2_HR_PIXC classification (error {abs(err):.2f} m)',
                      fontsize=FS + 0.5, fontweight='bold', color=fc, pad=4)
        ax1.grid(True, alpha=0.3)
        for sp_ in ax1.spines.values():
            sp_.set_linewidth(1.3); sp_.set_color(fc)
        ax1.text(-0.155, 1.045, '(a)' if row == 0 else '(b)', transform=ax1.transAxes,
                 fontsize=FS + 3, fontweight='bold')

        ax2 = fig.add_subplot(gs[row, 1])
        buf = 0.008
        lon0, lon1 = bounds[0] - buf, bounds[2] + buf
        lat0, lat1 = bounds[1] - buf, bounds[3] + buf
        lspan = lat1 - lat0
        plon0 = lon0 - LON_EXTEND if side == 'left' else lon0
        plon1 = lon1 + LON_EXTEND if side == 'right' else lon1
        plat0, plat1 = lat0 - lspan * 0.08, lat1 + lspan * 0.08
        with rasterio.open(nlcd_path) as src:
            win = src.window(plon0, plat0, plon1, plat1)
            arr = src.read(1, window=win)
            unique_vals = np.unique(arr[arr > 0]) if unique_vals is None else \
                np.union1d(unique_vals, np.unique(arr[arr > 0]))
            cmap_cols = ['#FFFFFF'] * 256
            for val, clr in NLCD_COLORS.items():
                cmap_cols[val] = clr
            ax2.imshow(arr, extent=[plon0, plon1, plat0, plat1],
                       cmap=ListedColormap(cmap_cols), vmin=0, vmax=255,
                       interpolation='nearest', rasterized=True)
        poly.boundary.plot(ax=ax2, color='#404040', linewidth=1.3)
        ax2.set_xlim(plon0, plon1); ax2.set_ylim(plat0, plat1)
        ax2.set_aspect('equal')
        ax2.xaxis.set_major_formatter(FuncFormatter(
            lambda x, p: f'{abs(x):.2f}°W' if x < 0 else f'{x:.2f}°E'))
        ax2.yaxis.set_major_formatter(FuncFormatter(
            lambda y, p: f'{y:.2f}°N' if y > 0 else f'{abs(y):.2f}°S'))
        if row == 1:
            ax2.set_xlabel('Longitude', fontsize=FS, fontweight='bold')
        ax2.set_ylabel('Latitude', fontsize=FS, fontweight='bold', labelpad=1)
        ax2.tick_params(labelsize=FS - 1.5, labelbottom=(row == 1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
        ax2.set_title(f'NLCD land cover (error {abs(err):.2f} m)',
                      fontsize=FS + 0.5, fontweight='bold', color=fc, pad=4)
        ax2.grid(True, alpha=0.3)
        for sp_ in ax2.spines.values():
            sp_.set_linewidth(1.3); sp_.set_color(fc)
        _band_lonlat(ax2, side, heading, cyc, pas, tile, plon0, plon1, plat0, plat1,
                     BW, fs=FS - 1.8)
        print(f"  pass {pas}: err={err:+.3f} n_pix={sel.sum()}")

    axb = fig.add_subplot(gs[2, :]); axb.axis('off')
    lg1 = axb.legend(handles=[Patch(facecolor=CLS_COLORS[i], edgecolor='black', linewidth=0.4,
                               label=CLS_LABELS[i]) for i in range(7)],
                loc='center', bbox_to_anchor=(0.0, 0.0, 0.43, 1.0), mode='expand',
                ncol=2, fontsize=6.5,
                frameon=True, fancybox=True, edgecolor='gray', framealpha=1.0,
                title='PIXC Classification', title_fontsize=7.2,
                columnspacing=0.7, handletextpad=0.4, handlelength=1.0,
                borderpad=0.4, labelspacing=0.45)
    axb.add_artist(lg1)
    axb.legend(handles=[Patch(facecolor=NLCD_COLORS[v], edgecolor='black', linewidth=0.4,
                              label=NLCD_LABELS[v])
                        for v in sorted(unique_vals) if v in NLCD_COLORS],
               loc='center', bbox_to_anchor=(0.46, 0.0, 0.54, 1.0), mode='expand',
               ncol=3, fontsize=6.5,
               frameon=True, fancybox=True, edgecolor='gray', framealpha=1.0,
               title='NLCD Land Cover', title_fontsize=7.2,
               columnspacing=0.6, handletextpad=0.35, handlelength=0.9,
               borderpad=0.4, labelspacing=0.45)
    fig.savefig(f"{OUT}/fig01_orig_print_1005.png", dpi=600, facecolor='white')
    fig.savefig(f"{OUT}/fig01_orig_print_1005.pdf", facecolor='white')
    plt.close(fig)
    print("fig01_orig saved")


# ═══════════════════════════════════════════════════════ Fig 4 (cell 84)
def fig04_orig():
    import pyarrow.parquet as pq
    ob = pq.read_table(f"{WORK}/2_urban_reaches/all_final_shape.parquet").to_pandas()
    if "fctimp_mean" not in ob.columns:
        isa = pd.read_parquet(f"{PROJ}/3_analysis_results/reviewer_checks/obstruction_network_distance.parquet",
                              columns=["reach_id", "isa"])
        isa["reach_id"] = isa.reach_id.astype("int64")
        ob["reach_id"] = ob.reach_id.astype("int64")
        ob = ob.merge(isa.rename(columns={"isa": "fctimp_mean"}), on="reach_id", how="left")
        if ob.fctimp_mean.max() > 1.5:
            ob["fctimp_mean"] = ob.fctimp_mean / 100
    df = ob.copy()
    df['error_relative_m'] = (df['wse_relative_m'] - df['wse_usgs_relative_m']).abs()
    df['isa_pct'] = df['fctimp_mean'] * 100
    isa_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    isa_labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40',
                  '40-45', '45-50']
    df['isa_bin'] = pd.cut(df['isa_pct'], bins=isa_bins, labels=isa_labels,
                           include_lowest=True, right=True).astype(str)
    df.loc[df['isa_pct'] >= 50, 'isa_bin'] = '50+'
    labels = isa_labels + ['50+']
    df = df.dropna(subset=['error_relative_m'])
    df = df[df['isa_bin'] != 'nan']

    fig, ax = plt.subplots(figsize=(3.6, 2.85))
    box_data, pos, ns, p68s = [], [], [], []
    for i, lb in enumerate(labels):
        s = df[df.isa_bin == lb]['error_relative_m'].values
        if len(s):
            box_data.append(s); pos.append(i); ns.append(len(s))
            p68s.append(np.percentile(s, 68))
    bp = ax.boxplot(box_data, positions=pos, widths=0.6, whis=[5, 95], showfliers=True,
                    patch_artist=True,
                    medianprops=dict(color='white', linewidth=1.4),
                    boxprops=dict(facecolor='olivedrab', edgecolor='black', linewidth=0.9),
                    whiskerprops=dict(color='black', linewidth=0.9),
                    capprops=dict(color='black', linewidth=0.9),
                    flierprops=dict(marker='o', markersize=1.6, markerfacecolor='lightgray',
                                    markeredgecolor='lightgray', alpha=0.5))
    ax.scatter(pos, p68s, marker='x', s=26, color='black', linewidth=1.4, zorder=5,
               label='68th Percentile Error')
    for i, (p, n) in enumerate(zip(pos, ns)):
        y_top = bp['whiskers'][i * 2 + 1].get_ydata().max()
        ax.text(p, min(y_top * 1.08, 4.55), f'n={n}', fontsize=5.6, fontweight='bold',
                rotation=90, ha='center', va='bottom', color='black')
    ax.axhline(T, color='black', linestyle='--', linewidth=1.1, alpha=0.9,
               label='Target Error (T = 0.18 m)')
    ax.set_ylim(0, 5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6.2, fontweight='bold')
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.tick_params(axis='y', labelsize=6.5)
    ax.set_xlabel('Impervious Surface Area (%)', fontsize=7.5, fontweight='bold')
    ax.set_ylabel(r'$|Relative\ WSE\ Error|$ (m)', fontsize=7.5, fontweight='bold')
    ax.set_title('SWOT Relative WSE Error Distribution\nby Impervious Surface Area',
                 fontsize=7.8, fontweight='bold', pad=4)
    ax.legend(loc='upper left', fontsize=6.6, framealpha=0.95, edgecolor='gray',
              borderpad=0.4, handlelength=1.4)
    ax.grid(True, axis='y', alpha=0.3)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0); sp.set_color('black')
    fig.subplots_adjust(left=0.13, right=0.985, top=0.87, bottom=0.19)
    fig.savefig(f"{OUT}/fig04_orig_print.png", dpi=600, facecolor='white')
    fig.savefig(f"{OUT}/fig04_orig_print.pdf", facecolor='white')
    plt.close(fig)
    print(f"fig04_orig saved (bins n: {ns})")


# ═══════════════════════════════════════════════════════ Fig 6 (cell 172)
def _resolve(vals, mind):
    p = np.array([v[0] for v in vals], float)
    for _ in range(3):
        for i in range(1, len(p)):
            if p[i] - p[i-1] < mind:
                p[i] += mind - (p[i] - p[i-1])
    return p


def _cdf_single(ax, data, y_col, geom_col, cats, title, zone_key):
    colors = {'Neither': 'dimgray', 'Near': 'teal', 'Far': 'coral', 'Both': 'olivedrab'}
    lm = ({'Neither': 'Neither', 'Near': 'Near', 'Far': 'Far', 'Both': 'Both'}
          if zone_key == 'Perpendicular' else
          {'Neither': 'Neither', 'Near': 'Near', 'Far': 'Far', 'Both': 'Both'})
    stats, p68i, cdfi = {}, [], []
    for cat in cats:
        cd = data[data[geom_col] == cat][y_col].dropna()
        if not len(cd):
            continue
        sd = np.sort(cd); cdf = np.arange(1, len(sd) + 1) / len(sd)
        c = colors[cat]
        ax.plot(sd, cdf, color=c, label=f'{lm[cat]} (n={len(cd)})', linewidth=1.2)
        p68 = np.percentile(cd, 68)
        cT = (cd <= T).sum() / len(cd)
        stats[cat] = dict(n=len(cd), p68=p68, pct_above_T=100 * (cd > T).mean(),
                          cdf_at_T=cT)
        ax.scatter([p68], [0.68], color=c, s=9, zorder=5)
        ax.scatter([T], [cT], color=c, s=12, zorder=5, marker='s')
        ax.vlines(p68, 0, 0.68, colors=c, linestyles=':', linewidth=0.6, alpha=0.6)
        ax.hlines(cT, 0, T, colors=c, linestyles=':', linewidth=0.6, alpha=0.6)
        p68i.append(dict(val=p68, color=c, text=f'{p68:.2f}'))
        cdfi.append(dict(val=cT, color=c, text=f'{cT:.2f}'.lstrip('0')))
    ax.axvline(T, color='black', linestyle='--', linewidth=0.9, alpha=0.8)
    ax.axhline(0.68, color='black', linewidth=0.9, alpha=0.8)
    ax.text(0.01, 0.695, '68%', transform=ax.get_yaxis_transform(), color='black',
            fontsize=6.5, fontweight='bold', ha='left', va='bottom')
    ax.text(T, 1.012, 'T=0.18m', transform=ax.get_xaxis_transform(), color='black',
            fontsize=6.2, fontweight='bold', ha='center', va='bottom')
    p68i.sort(key=lambda x: x['val'])
    for i, it in enumerate(p68i):
        nx = _resolve([(x['val'], j) for j, x in enumerate(p68i)], 0.16)[i]
        ax.text(nx, 0.012, it['text'], transform=ax.get_xaxis_transform(),
                color=it['color'], fontsize=5.9, fontweight='bold', ha='center', va='bottom')
    cdfi.sort(key=lambda x: x['val'])
    for i, it in enumerate(cdfi):
        ny = _resolve([(x['val'], j) for j, x in enumerate(cdfi)], 0.065)[i]
        ax.text(0.012, ny, it['text'], transform=ax.get_yaxis_transform(),
                color=it['color'], fontsize=5.9, fontweight='bold', ha='left', va='center')
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$|Relative\ WSE\ Error|$ (m)', fontsize=7.2)
    ax.set_ylabel('CDF', fontsize=7.2)
    ax.set_title(title, fontsize=7.8, fontweight='bold', pad=4)
    ax.tick_params(labelsize=6.3)
    ax.legend(loc='center right', fontsize=6.5, edgecolor='gray', framealpha=0.9,
              borderpad=0.35, handlelength=1.2)
    ax.grid(True, alpha=0.3)
    for sp in ax.spines.values():
        sp.set_color('black')
    return stats


def _stats_panel(ax, all_stats):
    ax.axis('off')
    colors = {'Neither': 'dimgray', 'Near': 'teal', 'Far': 'coral', 'Both': 'olivedrab'}
    cats = ['Neither', 'Near', 'Far', 'Both']
    zones = ['Parallel', 'Perpendicular', 'Other']
    ax.text(0.5, 1, 'Statistics Summary', transform=ax.transAxes, fontsize=8.5,
            fontweight='bold', ha='center', va='top')
    ax.text(0.5, 0.92, '68th Percentile |Error| (m)', transform=ax.transAxes,
            fontsize=7.2, fontweight='bold', ha='center', va='top')
    header = f"{'Category':<10}{'Parallel':>14}{'Perpendicular':>18}{'Other':>12}"
    ax.text(0.5, 0.84, header, transform=ax.transAxes, fontsize=6.4, ha='center',
            va='top', family='monospace')
    ax.text(0.5, 0.79, "─" * 54, transform=ax.transAxes, fontsize=6.4, color='gray',
            ha='center', va='top', family='monospace')
    y = 0.74
    for cat in cats:
        row = f"{cat:<10}"
        for z in zones:
            if z in all_stats and cat in all_stats[z]:
                v = all_stats[z][cat]['p68']
                row += f"{v:>18.3f}" if z == 'Perpendicular' else f"{v:>14.3f}" if z == 'Parallel' else f"{v:>12.3f}"
            else:
                row += f"{'N/A':>18}" if z == 'Perpendicular' else f"{'N/A':>14}" if z == 'Parallel' else f"{'N/A':>12}"
        ax.text(0.5, y, row, transform=ax.transAxes, fontsize=6.4, color=colors[cat],
                ha='center', va='top', family='monospace')
        y -= 0.07
    y -= 0.06
    ax.text(0.5, y, '% with |Error| > T (T=0.18m)', transform=ax.transAxes,
            fontsize=7.2, fontweight='bold', ha='center', va='top')
    y -= 0.08
    ax.text(0.5, y, header, transform=ax.transAxes, fontsize=6.4, ha='center', va='top',
            family='monospace')
    y -= 0.05
    ax.text(0.5, y, "─" * 54, transform=ax.transAxes, fontsize=6.4, color='gray',
            ha='center', va='top', family='monospace')
    y -= 0.05
    for cat in cats:
        row = f"{cat:<10}"
        for z in zones:
            if z in all_stats and cat in all_stats[z]:
                v = all_stats[z][cat]['pct_above_T']
                row += f"{v:>17.1f}%" if z == 'Perpendicular' else f"{v:>13.1f}%" if z == 'Parallel' else f"{v:>11.1f}%"
            else:
                row += f"{'N/A':>18}" if z == 'Perpendicular' else f"{'N/A':>14}" if z == 'Parallel' else f"{'N/A':>12}"
        ax.text(0.5, y, row, transform=ax.transAxes, fontsize=6.4, color=colors[cat],
                ha='center', va='top', family='monospace')
        y -= 0.07


def fig06_orig(final_df):
    df = final_df.dropna(subset=['abs_error_node', 'flow_angle', 'geometry_category']).copy()
    df['flow_zone'] = df['flow_angle'].apply(get_flow_angle_zone)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.9), facecolor='white')
    zones = [('Parallel', 'Parallel (350-10°, 170-190°)', axes[0, 0]),
             ('Perpendicular', 'Perpendicular (80-100°, 260-280°)', axes[0, 1]),
             ('Other', 'Other Angles', axes[1, 0])]
    all_stats = {}
    for zk, zt, ax in zones:
        dz = df[df.flow_zone == zk]
        st = _cdf_single(ax, dz, 'abs_error_node', 'geometry_category',
                         ['Neither', 'Near', 'Far', 'Both'], f'{zt}\n(n={len(dz):,})', zk)
        ax.set_xlim(0, 2)
        all_stats[zk] = st
    _stats_panel(axes[1, 1], all_stats)
    fig.suptitle(r'CDF of $|Relative\ WSE\ Error|$ by ISA Location and Flow Angle',
                 fontsize=9.5, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(f"{OUT}/fig06_orig_print.png", dpi=600, facecolor='white')
    fig.savefig(f"{OUT}/fig06_orig_print.pdf", facecolor='white')
    plt.close(fig)
    print("fig06_orig saved | 표값:", {z: {c: round(v['p68'], 3) for c, v in s.items()}
                                       for z, s in all_stats.items()})


# ═══════════════════════════════════════════════════════ Fig 7 (cell 182)
def _isa_bin(v):
    for hi, lab in ((5, '0-5'), (10, '5-10'), (20, '10-20'), (30, '20-30'), (40, '30-40'),
                    (50, '40-50'), (60, '50-60'), (70, '60-70'), (80, '70-80')):
        if v <= hi:
            return lab
    return '80-100'


BIN_ORDER = ['0-5', '5-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70',
             '70-80', '80-100']


def _bin_colors(isa_col):
    cmap = cm.get_cmap('Oranges' if isa_col == 'far_isa' else 'Greens', 12)
    return {BIN_ORDER[i]: cmap(i + 2) for i in range(10)}


def _cdf_bins(ax, data, y_col, isa_col, title, offset=1, xlim_max=20):
    bc = _bin_colors(isa_col)
    d = data.copy()
    d['isa_bin'] = d[isa_col].apply(_isa_bin)
    plotted = []
    for lb in BIN_ORDER:
        bd = d[d.isa_bin == lb][y_col].dropna()
        if not len(bd):
            continue
        sd = np.sort(bd)
        ax.plot(sd + offset, np.arange(1, len(sd) + 1) / len(sd), color=bc[lb], linewidth=1.2)
        p68 = np.percentile(bd, 68)
        cT = (bd <= T).sum() / len(bd)
        plotted.append((lb, p68, cT, bc[lb], len(bd)))
        ax.scatter([p68 + offset], [0.68], color=bc[lb], s=13, zorder=5)
    ax.axhline(0.68, color='black', linewidth=1.1, alpha=0.8)
    ax.text(0.02, 0.70, '68%', transform=ax.transAxes, fontsize=6.5, fontweight='bold')
    ax.axvline(T + offset, color='black', linestyle='--', linewidth=1.1, alpha=0.8)
    ax.axvline(offset, color='black', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.text(T + offset, 1.012, 'T=0.18m', transform=ax.get_xaxis_transform(),
            fontsize=6.2, fontweight='bold', ha='center', va='bottom')
    if plotted:
        ax.text(0.98, 0.555, f"{'ISA':>9}  {'68%tile':>9}  {'<T':>6}  {'n':>7}",
                transform=ax.transAxes, fontsize=6.4, va='top', ha='right',
                family='monospace', fontweight='bold')
        ax.text(0.98, 0.515, "─" * 40, transform=ax.transAxes, fontsize=4.6, color='gray',
                va='top', ha='right', family='monospace')
        y = 0.485
        for lb, p68, cT, c, n in plotted:
            ax.text(0.98, y, f"{lb + '%':>8}  {p68:>7.2f}m  {cT * 100:>5.0f}%  {n:>7,}",
                    transform=ax.transAxes, fontsize=6.4,
                    color=tuple(0.62 * x for x in __import__('matplotlib').colors.to_rgb(c)),
                    va='top', ha='right',
                    family='monospace')
            y -= 0.047
    ax.set_xscale('log')
    ax.set_xlim(offset * 0.9, xlim_max + offset)
    ax.set_ylim(0, 1)

    def fmt(v, p):
        o = v - offset
        return '0' if abs(o) < 0.01 else (f'{o:.0f}' if o >= 10 else f'{o:.1f}' if o >= 1 else f'{o:.2f}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))
    ax.set_xticks([v + offset for v in [0, 0.2, 0.5, 1, 2, 5, 10, 20] if v <= xlim_max])
    ax.set_xlabel(r'$|Relative\ WSE\ Error|$ (m)', fontsize=7.2)
    ax.set_ylabel('CDF', fontsize=7.2)
    ax.set_title(title, fontsize=7.8, fontweight='bold', pad=4)
    ax.tick_params(labelsize=6.3)
    ax.grid(True, alpha=0.3)
    for sp in ax.spines.values():
        sp.set_color('black')


def _box_bins(ax, data, y_col, isa_col, title, xlabel):
    bc = _bin_colors(isa_col)
    d = data.copy()
    d['isa_bin'] = d[isa_col].apply(_isa_bin)
    ex = [b for b in BIN_ORDER if b in d.isa_bin.values]
    box, poss, cols = [], [], []
    for i, lb in enumerate(ex):
        v = d[d.isa_bin == lb][y_col].dropna().values
        if len(v):
            box.append(v); poss.append(len(poss)); cols.append(bc[lb])
    bp = ax.boxplot(box, positions=poss, widths=0.6, patch_artist=True, whis=1.5,
                    medianprops=dict(color='black', linewidth=1.1),
                    whiskerprops=dict(color='black', linewidth=0.7),
                    capprops=dict(color='black', linewidth=0.7),
                    flierprops=dict(marker='o', markersize=1.2, alpha=0.5))
    for patch, c in zip(bp['boxes'], cols):
        patch.set_facecolor(c); patch.set_alpha(0.7); patch.set_edgecolor('black')
        patch.set_linewidth(0.7)
    for fl, c in zip(bp['fliers'], cols):
        fl.set_markerfacecolor(c); fl.set_markeredgecolor(c)
    ax.set_yscale('symlog', linthresh=1)
    ticks = [0.1, 0.5, 1, 2, 5, 10, 20]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=6.3)
    ax.set_ylim(0.05, 30)
    ax.set_xticks(poss)
    ax.set_xticklabels(ex, rotation=45, ha='right', fontsize=6.0)
    ax.set_xlabel(xlabel, fontsize=7.2)
    ax.set_ylabel(r'$|Relative\ WSE\ Error|$ (m)', fontsize=7.2)
    ax.set_title(title, fontsize=7.8, fontweight='bold', pad=4)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, len(ex) - 0.5)
    ax.tick_params(labelsize=6.3)
    for sp in ax.spines.values():
        sp.set_color('black')


def fig07_orig(final_df):
    need = ['abs_error_node', 'flow_angle', 'far_isa', 'far_ratio', 'near_isa', 'near_ratio']
    df = final_df.dropna(subset=[c for c in need if c in final_df.columns]).copy()
    df['flow_zone'] = df['flow_angle'].apply(get_flow_angle_zone)
    par = df[df.flow_zone == 'Parallel']
    far, nearp = par[par.far_ratio >= 0], par[par.near_ratio >= 0]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.9), facecolor='white')
    _cdf_bins(axes[0, 0], far, 'abs_error_node', 'far_isa', f'CDF by Far ISA (n={len(far):,})')
    _box_bins(axes[0, 1], far, 'abs_error_node', 'far_isa', 'Boxplot by Far ISA', 'Far ISA (%)')
    _cdf_bins(axes[1, 0], nearp, 'abs_error_node', 'near_isa', f'CDF by Near ISA (n={len(nearp):,})')
    _box_bins(axes[1, 1], nearp, 'abs_error_node', 'near_isa', 'Boxplot by Near ISA', 'Near ISA (%)')
    fig.suptitle('Parallel Angle Analysis - Far & Near ISA', fontsize=9.5,
                 fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(f"{OUT}/fig07_orig_print.png", dpi=600, facecolor='white')
    fig.savefig(f"{OUT}/fig07_orig_print.pdf", facecolor='white')
    plt.close(fig)
    print(f"fig07_orig saved (far n={len(far):,} near n={len(nearp):,})")


# ═══════════════════════════════════════════════════════ Fig 8 (cell 213)
def _isa_bin8(v):
    for hi, lab in ((5, '0-5'), (10, '5-10'), (20, '10-20'), (30, '20-30'), (45, '30-45'),
                    (60, '45-60')):
        if v <= hi:
            return lab
    return '60+'


def _grouped_orig(final_df, isa_col, bld_col, side_label, out_name, expect_n=None):
    ISA_ORDER = ['0-5', '5-10', '10-20', '20-30', '30-45', '45-60', '60+']
    df = final_df[final_df['flow_angle'].apply(get_flow_angle_zone) == 'Parallel'].copy()
    data = df[[isa_col, bld_col, 'abs_error_node']].dropna()
    data.columns = ['far_isa', 'far_building', 'abs_error_node']
    data['isa_bin'] = data.far_isa.apply(_isa_bin8)
    vals = data.far_building[data.far_building >= 5]
    edges = [5]
    for e in range(7, int(np.ceil(vals.max())) + 2, 2):
        if (vals >= e).sum() >= 500:
            edges.append(e)
        else:
            break
    edges.append(np.inf)
    blabels = [f'{edges[i]}-{edges[i+1]}' if np.isfinite(edges[i+1]) else f'{edges[i]}+'
               for i in range(len(edges) - 1)]
    data['building_bin'] = pd.cut(data.far_building, bins=edges, labels=blabels, right=False)
    data = data.dropna(subset=['building_bin'])
    nI, nB = len(ISA_ORDER), len(blabels)
    bcnt = data.groupby('building_bin', observed=True).size().to_dict()
    icnt = data.groupby('isa_bin', observed=True).size().to_dict()
    bl_n = [f'{l}\n(n={bcnt.get(l, 0):,})' for l in blabels]
    ex_isa = [b for b in ISA_ORDER if b in data.isa_bin.values]
    il_n = [f'{l}\n(n={icnt.get(l, 0):,})' for l in ex_isa]
    colors_isa = [cm.get_cmap('Oranges', nI + 2)(i + 2) for i in range(nI)]
    colors_b = [cm.get_cmap('Purples', nB + 2)(i + 2) for i in range(nB)]
    OFF = 0.1
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 6.85), facecolor='white')

    def one_row(ax, outer_labels, inner_labels, outer_col, inner_col, cols, xlab, title,
                x_n_labels):
        gw = 0.8
        bw = gw / (len(inner_labels) + 1)
        box, poss, bcols, ns = [], [], [], []
        for i, ol in enumerate(outer_labels):
            for j, il in enumerate(inner_labels):
                m = (data[outer_col] == ol) & (data[inner_col] == il)
                s = data[m]['abs_error_node'].values
                box.append(s + OFF if len(s) else [np.nan])
                ns.append(len(s))
                poss.append(i + 1 + (j - (len(inner_labels) - 1) / 2) * bw)
                bcols.append(cols[j])
        bp = ax.boxplot(box, positions=poss, widths=bw * 0.85, patch_artist=True,
                        showfliers=True,
                        flierprops=dict(marker='o', markersize=0.8, alpha=0.3,
                                        markerfacecolor='gray', markeredgecolor='none'))
        for patch, c in zip(bp['boxes'], bcols):
            patch.set_facecolor(c); patch.set_edgecolor('black')
            patch.set_linewidth(0.45); patch.set_alpha(0.85)
        for w in bp['whiskers']:
            w.set_color('black'); w.set_linewidth(0.45)
        for cp in bp['caps']:
            cp.set_color('black'); cp.set_linewidth(0.45)
        for md in bp['medians']:
            md.set_color('black'); md.set_linewidth(0.8)
        for i, (p, n) in enumerate(zip(poss, ns)):
            if n > 0:
                yt = bp['whiskers'][i * 2 + 1].get_ydata().max()
                ax.text(p, yt * 1.06, f'n={n}', rotation=90, fontsize=6.5, ha='center',
                        va='bottom', color='black')
        ax.set_yscale('log')
        ax.set_ylim(OFF * 0.8, 85 + OFF)
        tv = [0, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
        ax.set_yticks([v + OFF for v in tv])
        ax.set_yticklabels([str(v) for v in tv], fontsize=7.0)
        ax.set_ylabel(r'$|Relative\ WSE\ Error|$ (m)', fontsize=7.8)
        ax.set_xlabel(xlab, fontsize=7.8, fontweight='bold')
        ax.set_xticks(range(1, len(outer_labels) + 1))
        ax.set_xticklabels(x_n_labels, fontsize=6.8)
        ax.set_xlim(0.3, len(outer_labels) + 0.7)
        ax.tick_params(labelsize=7.0)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.4)
        ax.set_title(title, fontsize=8.4, fontweight='bold', pad=3)
        ax.margins(x=0.01)
        for sp in ax.spines.values():
            sp.set_color('black')

    one_row(axes[0], blabels, ISA_ORDER, 'building_bin', 'isa_bin', colors_isa,
            f'{side_label} Building Height (m)',
            f'Bin: {side_label} Building Height & Color: {side_label} ISA (n={len(data):,})', bl_n)
    axes[0].legend(handles=[Patch(facecolor=colors_isa[i], edgecolor='black',
                                  label=ISA_ORDER[i] + '%', alpha=0.85, linewidth=0.5)
                            for i in range(nI)],
                   title=f'{side_label} ISA', loc='upper center',
                   bbox_to_anchor=(0.5, -0.33), ncol=nI,
                   framealpha=0.95, title_fontsize=7.0, fontsize=6.5, borderpad=0.4,
                   handlelength=1.1, columnspacing=0.9, handletextpad=0.4)
    one_row(axes[1], ex_isa, blabels, 'isa_bin', 'building_bin', colors_b,
            f'{side_label} ISA (%)',
            f'Bin: {side_label} ISA & Color: {side_label} Building Height (n={len(data):,})', il_n)
    axes[1].legend(handles=[Patch(facecolor=colors_b[i], edgecolor='black',
                                  label=blabels[i] + 'm', alpha=0.85, linewidth=0.5)
                            for i in range(nB)],
                   title=f'{side_label} Building Height', loc='upper center',
                   bbox_to_anchor=(0.5, -0.33), ncol=nB,
                   framealpha=0.95, title_fontsize=7.8, fontsize=7.2, borderpad=0.4,
                   handlelength=1.1, columnspacing=0.9, handletextpad=0.4)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.975, bottom=0.165, hspace=0.68)
    fig.savefig(f"{OUT}/{out_name}.png", dpi=600, facecolor='white')
    fig.savefig(f"{OUT}/{out_name}.pdf", facecolor='white')
    plt.close(fig)
    if expect_n is not None:
        print(f"  모집단 n={len(data):,} (기대 {expect_n:,})")
    # 리뷰어 정합 확인용
    g05 = data[data.building_bin == blabels[0]]
    lo = np.percentile(g05[g05.isa_bin == '0-5'].abs_error_node, 68) if len(g05[g05.isa_bin == '0-5']) else np.nan
    hi = np.percentile(g05[g05.isa_bin == '30-45'].abs_error_node, 68) if len(g05[g05.isa_bin == '30-45']) else np.nan
    print(f"{out_name} saved | 최저높이빈 ISA 0-5→30-45 p68: {lo:.2f}→{hi:.2f}")


def fig08_orig(final_df):
    _grouped_orig(final_df, 'far_isa', 'far_building', 'Far', 'fig08_orig_print')


def figS3_orig(final_df):
    _grouped_orig(final_df, 'near_isa', 'near_building', 'Near', 'figS3_orig_print',
                  expect_n=17368)





# ═══════════════════════════════════════════════════════ Fig 3 (money map + schematic)
def _sat_glyph(ax, x, y, scale, angle_deg):
    """작은 위성 글리프(본체+패널 2)."""
    import matplotlib.transforms as mtr
    tr = mtr.Affine2D().rotate_deg(angle_deg).translate(x, y) + ax.transData
    body = plt.Rectangle((-0.35 * scale, -0.5 * scale), 0.7 * scale, 1.0 * scale,
                         facecolor='white', edgecolor='black', linewidth=0.6,
                         zorder=9, transform=tr)
    p1 = plt.Rectangle((-1.7 * scale, -0.3 * scale), 1.1 * scale, 0.6 * scale,
                       facecolor='#8f9bd6', edgecolor='black', linewidth=0.5,
                       zorder=9, transform=tr)
    p2 = plt.Rectangle((0.6 * scale, -0.3 * scale), 1.1 * scale, 0.6 * scale,
                       facecolor='#8f9bd6', edgecolor='black', linewidth=0.5,
                       zorder=9, transform=tr)
    for a in (p1, p2, body):
        ax.add_patch(a)


def _money_map(ax):
    from matplotlib.lines import Line2D
    sites = gpd.read_file(f"{WORK}/2_urban_reaches/selected_12_vD.shp")
    col = "fctimp_mea" if "fctimp_mea" in sites.columns else "fctimp_mean"
    sample = sites[np.isclose(sites[col], 0.06, atol=0.01)].copy()
    s4326 = sample.to_crs('EPSG:4326')
    b = s4326.total_bounds
    ext = [b[0] - 0.1, b[1] - 0.005, b[2] + 0.1, b[3] + 0.005]

    nlcd_path = f"{WORK}/0_data_collection/1_nlcd/Annual_NLCD_LndCov_2024_CU_C1V1_EPSG4326.tif"
    with rasterio.open(nlcd_path) as src:
        win = src.window(*ext)
        nl = src.read(1, window=win)
        nb = rasterio.windows.bounds(win, src.transform)
    ov = np.zeros((*nl.shape, 4), dtype=np.float32)
    ov[np.isin(nl, [22, 23, 24])] = [1.0, 0.0, 0.0, 0.30]
    ov[nl == 11] = [0.0, 0.4, 1.0, 0.70]
    ov[(nl > 0) & ~np.isin(nl, [11, 22, 23, 24])] = [0.5, 0.5, 0.5, 0.55]

    basemap_ok = False
    try:
        import contextily as cx
        s3857 = sample.to_crs('EPSG:3857')
        import geopandas as _g
        from shapely.geometry import box as _box
        eb = _g.GeoDataFrame(geometry=[_box(*ext)], crs='EPSG:4326').to_crs('EPSG:3857').total_bounds
        ax.set_xlim(eb[0], eb[2]); ax.set_ylim(eb[1], eb[3])
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zoom=13,
                       attribution_size=3)
        nbb = _g.GeoDataFrame(geometry=[_box(nb[0], nb[1], nb[2], nb[3])],
                              crs='EPSG:4326').to_crs('EPSG:3857').total_bounds
        ax.imshow(ov, extent=[nbb[0], nbb[2], nbb[1], nbb[3]], origin='upper',
                  zorder=2, interpolation='nearest')
        s3857.boundary.plot(ax=ax, color='black', linewidth=1.6, zorder=5)
        from pyproj import Transformer
        t34 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        t43 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, p: f'{abs(t34.transform(x, 0)[0]):.2f}°W'))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda y, p: f'{t34.transform(0, y)[1]:.2f}°N'))
        def to_ax(lon, lat):
            return t43.transform(lon, lat)
        basemap_ok = True
    except Exception as e:
        print("  basemap 실패, NLCD 배경으로 대체:", e)
        ax.set_xlim(ext[0], ext[2]); ax.set_ylim(ext[1], ext[3])
        ax.imshow(ov, extent=[nb[0], nb[2], nb[1], nb[3]], origin='upper',
                  zorder=2, interpolation='nearest')
        ax.set_facecolor('#d9d9cf')
        s4326.boundary.plot(ax=ax, color='black', linewidth=1.6, zorder=5)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{abs(x):.2f}°W'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{y:.2f}°N'))
        def to_ax(lon, lat):
            return lon, lat

    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xs, ys = x1 - x0, y1 - y0

    def band(pos, colr, heading, cyc, pas, tile):
        north = (0 <= heading < 90) or (270 <= heading < 360)
        bw = xs * 0.055
        cx_ = x0 + xs * 0.13 if pos == 'left' else x1 - xs * 0.13
        off = ys * 0.09
        pts = ([(cx_-bw/2-off, y0), (cx_+bw/2-off, y0), (cx_+bw/2+off, y1), (cx_-bw/2+off, y1)]
               if north else
               [(cx_-bw/2+off, y0), (cx_+bw/2+off, y0), (cx_+bw/2-off, y1), (cx_-bw/2-off, y1)])
        ax.add_patch(MplPolygon(pts, facecolor=colr, edgecolor='black', alpha=0.5,
                                linewidth=0.8, zorder=6))
        rot = 76 if north else -76
        ax.text(cx_, y0 + ys * 0.42, f'Satellite Path (Cycle: {cyc}, Pass: {pas}, Tile: {tile})',
                fontsize=5.6, fontweight='bold', color='white', ha='center', va='center',
                rotation=rot, zorder=7)
        hx = cx_ + (off if north else -off) * 0.9
        ax.text(hx, y1 - ys * 0.035, f'{heading:.0f}°', fontsize=6.2, fontweight='bold',
                color='white', ha='center', va='top', rotation=rot, zorder=7)
        a0 = (cx_ - off * 0.75, y0 + ys * 0.06) if north else (cx_ + off * 0.75, y0 + ys * 0.06)
        a1 = (cx_ + off * 0.75, y1 - ys * 0.06) if north else (cx_ - off * 0.75, y1 - ys * 0.06)
        ax.annotate('', xy=a1, xytext=a0,
                    arrowprops=dict(arrowstyle='-|>', color='black', lw=1.0), zorder=7)
        sat = (cx_ + (off if north else -off) * 0.35, y0 + ys * 0.62)
        _sat_glyph(ax, sat[0], sat[1], xs * 0.008, rot)
        return sat

    sat_l = band('left', 'coral', 14, '039', '315', '227R')
    sat_r = band('right', 'teal', 166, '039', '272', '082R')

    node = to_ax(-96.503, 41.410)
    far_lab = to_ax(-96.492, 41.424)
    near_lab = to_ax(-96.462, 41.404)
    ax.plot([sat_l[0], far_lab[0]], [sat_l[1], far_lab[1]], color='orange',
            linewidth=1.1, linestyle='--', zorder=8,
            marker='>', markevery=[1], markersize=3.5)
    ax.plot([sat_r[0], near_lab[0]], [sat_r[1], near_lab[1]], color='mediumturquoise',
            linewidth=1.1, linestyle='--', zorder=8,
            marker='<', markevery=[1], markersize=3.5)
    ax.scatter(*node, s=26, facecolor='white', edgecolor='black', linewidth=1.0, zorder=9)
    ax.text(far_lab[0], far_lab[1] + ys * 0.012, 'Far ISA', fontsize=7, fontweight='bold',
            color='orange', ha='center', va='bottom', zorder=9,
            path_effects=None)
    ax.text(near_lab[0], near_lab[1] - ys * 0.012, 'Near ISA', fontsize=7, fontweight='bold',
            color='mediumturquoise', ha='center', va='top', zorder=9)

    ax.tick_params(labelsize=7.2)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=12, ha='right')
    ax.set_xlabel('Longitude', fontsize=7.5, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=7.5, fontweight='bold')
    for sp in ax.spines.values():
        sp.set_color('black')

    handles = [
        Line2D([0], [0], color='orange', linestyle='--', linewidth=1.4, marker='>',
               markersize=4, label='Measurement path (pass 315)'),
        Line2D([0], [0], color='mediumturquoise', linestyle='--', linewidth=1.4,
               marker='<', markersize=4, label='Measurement path (pass 272)'),
        Patch(facecolor='none', edgecolor='black', linewidth=1.2, label='Reach buffer zone'),
        Line2D([0], [0], color='black', linewidth=1.2, marker='>', markersize=4,
               label='Satellite path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=5, label='SWOT node'),
        Patch(facecolor='red', alpha=0.35, edgecolor='black', label='Developed (ISA)'),
        Patch(facecolor='blue', alpha=0.4, edgecolor='black', label='Open water'),
        Patch(facecolor='gray', alpha=0.5, edgecolor='black', label='Other land cover'),
    ]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.175, 0.03),
              fontsize=6.2, framealpha=0.92, edgecolor='gray', borderpad=0.45,
              handlelength=1.6, labelspacing=0.4)


def _schematic_panels(axes, final_df=None):
    """기존 zone-flip 개략도 2패널 (fig_print_final.fig03 이식)."""
    from shapely.geometry import LineString
    import matplotlib.patches as mpatches
    import pyarrow.parquet as pq
    cols = ["reach_id", "time_str", "flow_angle", "far_isa", "near_isa", "far_building",
            "near_building", "error_node", "node_id", "lat", "lon", "sat_lat", "sat_lon"]
    d = pq.read_table(f"{PROJ}/3_analysis_results/swot_urban_nearfar_v6.5_results.parquet",
                      columns=cols).to_pandas()
    d["reach_id"] = d.reach_id.astype("int64")
    d["e"] = d.error_node.abs()
    for c in ("far_isa", "near_isa"):
        d[c + "_pct"] = d[c] * 100 if d[c].max() <= 1 else d[c]
    d["visit"] = pd.to_datetime(d.time_str, errors="coerce", utc=True).dt.floor("h")
    d = d.dropna(subset=["e", "lat", "lon", "sat_lat", "sat_lon"])
    g = (d.groupby(["reach_id", "visit"])
         .agg(fi=("far_isa_pct", "mean"), ni=("near_isa_pct", "mean"),
              fb=("far_building", "mean"), nb=("near_building", "mean"),
              n=("e", "size"), slat=("sat_lat", "median"), slon=("sat_lon", "median"))
         .reset_index())
    best = None
    for rid, gr in g.groupby("reach_id"):
        A = gr[(gr.fi >= 25) & (gr.ni <= 8)]
        B = gr[(gr.ni >= 25) & (gr.fi <= 8)]
        if len(A) and len(B):
            score = min(A.fi.max(), B.ni.max()) + 0.01 * min(A.n.max(), B.n.max())
            if best is None or score > best[0]:
                best = (score, rid, A.sort_values("fi").iloc[-1], B.sort_values("ni").iloc[-1])
    _, rid, va, vb = best
    for ax, vis, tag, pl in ((axes[0], va, "far", "b"), (axes[1], vb, "near", "c")):
        nod = d[(d.reach_id == rid) & (d.visit == vis.visit)].sort_values("node_id")
        lon0, lat0 = nod.lon.mean(), nod.lat.mean()
        kx = np.cos(np.radians(lat0)) * 111.32
        x = ((nod.lon - lon0) * kx).values
        y = ((nod.lat - lat0) * 111.32).values
        sx = (vis.slon - lon0) * kx
        sy = (vis.slat - lat0) * 111.32
        sd = np.hypot(sx, sy)
        ux, uy = sx / sd, sy / sd
        line = LineString(np.column_stack([x, y]))
        offA = line.offset_curve(0.85)
        offB = line.offset_curve(-0.85)
        cA = np.array(offA.coords).mean(axis=0)
        near_off, far_off = (offA, offB) if np.dot(cA - [x.mean(), y.mean()], [ux, uy]) > 0 else (offB, offA)
        urban_off = far_off if tag == "far" else near_off
        for off, colr in ((near_off, "#74add1"), (far_off, "#f4a582")):
            oc = np.array(off.coords)
            poly = np.vstack([np.column_stack([x, y]), oc[::-1]])
            ax.add_patch(mpatches.Polygon(poly, closed=True, facecolor=colr,
                                          alpha=0.45, edgecolor="none"))
        ouc = np.array(urban_off.coords)
        upoly = np.vstack([np.column_stack([x, y]), ouc[::-1]])
        ax.add_patch(mpatches.Polygon(upoly, closed=True, facecolor="none",
                                      edgecolor="#7f0000", hatch="////", lw=0.0, alpha=0.85))
        ax.plot(x, y, color="#9ecae1", lw=4.2, solid_capstyle="round", zorder=3)
        ax.plot(x, y, color="#08306b", lw=1.5, solid_capstyle="round", zorder=4)
        non_urban_off = near_off if tag == "far" else far_off
        noc_r = np.array(non_urban_off.coords)
        ri = int(len(x) * 0.12); rin = int(len(noc_r) * 0.12)
        dvec = noc_r[rin] - [x[ri], y[ri]]
        dvec = dvec / np.hypot(*dvec)
        ax.annotate("river", xy=(x[ri], y[ri]),
                    xytext=(x[ri] + dvec[0] * 1.45, y[ri] + dvec[1] * 1.45),
                    fontsize=6.6, color="#08306b", fontweight="bold", style="italic",
                    ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.7, alpha=0.78),
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#08306b"))
        ax.annotate("", xytext=(ux * 1.35, uy * 1.35), xy=(ux * 2.2, uy * 2.2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="k"))
        if ux >= 0:
            tx, ty = ux * 2.85, uy * 2.85
        else:
            tx, ty = ux * 2.05, uy * 2.05 + 0.78
        ax.text(tx, ty, "to\nsatellite", ha="center", va="center", fontsize=6.6,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.75))
        mid = len(x) // 2
        for off, lab, colr in ((near_off, "near-nadir", "#08519c"),
                               (far_off, "far-nadir", "#a50f15")):
            oc = np.array(off.coords)
            om = oc[len(oc) // 2]
            away = om - [x[mid], y[mid]]
            away = away / np.hypot(*away)
            ax.text(om[0] + away[0] * 0.72, om[1] + away[1] * 0.72, lab, fontsize=6.5,
                    color=colr, ha="center", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.7))
        zisa = vis.fi if tag == "far" else vis.ni
        zbld = vis.fb if tag == "far" else vis.nb
        om2 = ouc[int(len(ouc) * 0.22)]
        aw2 = om2 - [x[int(len(x) * 0.22)], y[int(len(y) * 0.22)]]
        aw2 = aw2 / np.hypot(*aw2)
        ax.annotate(f"urban corridor\nISA {zisa:.0f}%, {zbld:.1f} m",
                    xy=(om2[0], om2[1]),
                    xytext=(om2[0] + aw2[0] * 1.15, om2[1] + aw2[1] * 1.15 - 0.6),
                    fontsize=6.3, ha="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.75),
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#7f0000"))
        ax.set_title(f"({pl}) same urban corridor, {tag}-nadir zone ({vis.visit.date()})",
                     fontsize=7.4)
        ax.set_aspect("equal")
        ax.set_xlim(-3.1, 3.1); ax.set_ylim(-3.4, 3.4)
        ax.set_xlabel("east-west (km)", fontsize=7.8)
        ax.tick_params(labelsize=7.2)
    axes[0].set_ylabel("north-south (km)", fontsize=7.8)


def fig03_orig():
    fig = plt.figure(figsize=(7.16, 6.75))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.52, 1.0], hspace=0.30, wspace=0.18,
                  left=0.108, right=0.985, top=0.965, bottom=0.06)
    axm = fig.add_subplot(gs[0, :])
    img = plt.imread(str(Path(__file__).parent / "fig03_panel_a_core.png"))
    import geopandas as gpd
    from shapely.geometry import box as _box
    from pyproj import Transformer
    from matplotlib.ticker import FuncFormatter
    sel = gpd.read_file("/work/pi_kandread_umass_edu/swot-urban/2_urban_reaches/selected_12_vD.shp")
    smp = sel[np.isclose(sel["fctimp_mea"], 0.06, atol=0.01)]
    b4 = smp.to_crs("EPSG:4326").total_bounds
    eb = [b4[0] - 0.1, b4[1] - 0.005, b4[2] + 0.1, b4[3] + 0.005]
    e3 = gpd.GeoDataFrame(geometry=[_box(*eb)], crs="EPSG:4326").to_crs("EPSG:3857").total_bounds
    axm.imshow(img, extent=[e3[0], e3[2], e3[1], e3[3]], aspect='auto')
    _tr = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    axm.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{abs(_tr.transform(x, 0)[0]):.2f}°W"))
    axm.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f"{_tr.transform(0, y)[1]:.2f}°N"))
    axm.tick_params(labelsize=7.2)
    plt.setp(axm.xaxis.get_majorticklabels(), rotation=15, ha='right')
    axm.set_xlabel("Longitude", fontsize=8, fontweight='bold', labelpad=1.5)
    axm.set_ylabel("Latitude", fontsize=8, fontweight='bold', labelpad=1.5)
    axm.set_title("(a) two observing geometries over the study corridor "
                  "(cycle 039, passes 315 and 272)", fontsize=8, fontweight='bold', pad=4)
    axs = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    _schematic_panels(axs)
    fig.savefig(f"{OUT}/fig03_map_schematic.png", dpi=600, facecolor='white')
    fig.savefig(f"{OUT}/fig03_map_schematic.pdf", facecolor='white')
    plt.close(fig)
    print("fig03_orig saved")


if __name__ == "__main__":
    import sys
    NEED_DF = ("fig06_orig", "fig07_orig", "fig08_orig", "figS3_orig")
    todo = sys.argv[1:] or ["fig01_orig", "fig03_orig", "fig04_orig", "fig06_orig",
                            "fig07_orig", "fig08_orig", "figS3_orig"]
    fd = None
    for name in todo:
        if name in NEED_DF and fd is None:
            fd = load_final_df()
        fn = globals()[name]
        fn(fd) if name in NEED_DF else fn()
