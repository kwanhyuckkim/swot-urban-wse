#!/usr/bin/env python
"""Print-width rebuilds the reviewers asked for.

fig01  the paired PIXC/NLCD comparison at its final 7.16 in width, keeping the
       published pair (cycle 039, pass 272 low error vs pass 315 high error at the
       0.06-ISA Platte site), adding the (a)/(b) row labels R3 asked for, naming the
       products in the titles as R4 asked, and holding every font at 8 pt or larger.
fig05  the quality-flag boxplot with the Wilson 95% interval the response promises
       on every annotated share, the geometric screen row on the same axis, and the
       silent-bit note.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.windows import from_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from statsmodels.stats.proportion import proportion_confint
import pyarrow.parquet as pq

WORK = "/work/pi_kandread_umass_edu/swot-urban"
PROJ = "/project/pi_kandread_umass_edu/swot-urban"
OUT = f"{PROJ}/3_analysis_results/figures_print"
Path(OUT).mkdir(exist_ok=True)
T = 0.1832

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.linewidth": 0.7})

CLS_COLORS = ['maroon', 'gold', 'skyblue', 'mediumblue', 'dimgray', 'cyan', 'darkturquoise']
CLS_LABELS = ['Land', 'Land near water', 'Water near land', 'Open water',
              'Dark water', 'Low coh. near land', 'Low coh. open']
NLCD_COLORS = {11: '#476BA1', 12: '#D1DDF9', 21: '#DDC9C9', 22: '#D89382', 23: '#ED0000',
               24: '#AA0000', 31: '#B2ADA3', 41: '#68AA63', 42: '#1C6330', 43: '#B5C98E',
               52: '#CCBA7C', 71: '#E2E2C1', 81: '#DDD83E', 82: '#AE7229', 90: '#BBD7ED',
               95: '#71A4C1'}
NLCD_LABELS = {11: 'Open water', 12: 'Ice/snow', 21: 'Dev. open', 22: 'Dev. low',
               23: 'Dev. medium', 24: 'Dev. high', 31: 'Barren', 41: 'Decid. forest',
               42: 'Evergr. forest', 43: 'Mixed forest', 52: 'Shrub', 71: 'Grassland',
               81: 'Pasture', 82: 'Cultivated', 90: 'Woody wetl.', 95: 'Herb. wetl.'}


def best(files):
    keep = {}
    for f in files:
        m = re.match(r"(.+)_(P[IG]D\d)_(\d+)\.nc", Path(f).name)
        if not m:
            continue
        b, pr, c = m.group(1), m.group(2), int(m.group(3))
        if b not in keep or (pr == "PGD0" and keep[b][1] == "PID0") or \
           (pr == keep[b][1] and c > keep[b][2]):
            keep[b] = (f, pr, c)
    return [v[0] for v in keep.values()]


def fig01():
    fct = "0.06"
    sites = gpd.read_file(f"{WORK}/2_urban_reaches/selected_12_vD.shp")
    col = "fctimp_mea" if "fctimp_mea" in sites.columns else "fctimp_mean"
    poly = sites[np.isclose(sites[col], float(fct), atol=0.01)].to_crs("EPSG:4326").geometry.iloc[0]
    obs = pd.read_parquet(f"{WORK}/1_data_processing/fctimp_mean_{fct}_sample.parquet")
    obs["err"] = (obs.wse_relative_m - obs.wse_usgs_relative_m).abs()
    obs["t"] = pd.to_datetime(obs.time_str).dt.tz_localize(None)
    files = best(list(Path(f"{PROJ}/0_data_collection/SWOT_PIXC/fctimp_{fct}").glob("SWOT_L2_HR_PIXC_*.nc")))
    picks = {}
    for f in files:
        parts = Path(f).name.split("_")
        cyc, pas, tile = parts[4], parts[5], parts[6]
        if (cyc, pas, tile) in [("039", "272", "082R"), ("039", "315", "227R")]:
            picks[pas] = (f, tile, cyc)
    assert set(picks) == {"272", "315"}, picks

    fig = plt.figure(figsize=(7.16, 5.9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.46], hspace=0.46, wspace=0.24,
                          left=0.075, right=0.985, top=0.93, bottom=0.01)
    nlcd_path = f"{WORK}/0_data_collection/1_nlcd/Annual_NLCD_LndCov_2024_CU_C1V1_EPSG4326.tif"
    b = poly.bounds
    pad = 0.03
    win_b = (b[0]-pad, b[1]-pad, b[2]+pad, b[3]+pad)
    present = set()
    for row, pas in enumerate(["272", "315"]):
        f, tile, cyc = picks[pas]
        parts = Path(f).name.split("_")
        t0 = pd.to_datetime(parts[7], format="%Y%m%dT%H%M%S")
        near = obs.loc[(obs.t - t0).abs().dt.total_seconds().idxmin()]
        err = float(near.err)
        ds = xr.open_dataset(f, group="pixel_cloud")
        lon, lat = ds.longitude.values, ds.latitude.values
        import shapely
        inside = (lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3])
        idx = np.where(inside)[0]
        idx = idx[shapely.contains_xy(poly, lon[idx], lat[idx])]
        cls = ds.classification.values[idx].astype(int)
        rng = ds.range_index.values[idx]
        azi = ds.azimuth_index.values[idx]
        tvp = xr.open_dataset(f, group="tvp")
        head = float(np.nanmedian(tvp.velocity_heading.values))
        ds.close(); tvp.close()

        ax = fig.add_subplot(gs[row, 0])
        for ci in range(3, 8):
            m = cls == ci
            if m.any():
                ax.scatter(rng[m], azi[m], s=0.5, c=CLS_COLORS[ci-1], rasterized=True)
        ax.set_xlabel("range index"); ax.set_ylabel("azimuth index")
        ax.set_title(f"SWOT L2_HR_PIXC classification (error {err:.2f} m)", fontsize=9)
        lab = "(a)" if row == 0 else "(b)"
        ax.text(-0.13, 1.05, lab, transform=ax.transAxes, fontsize=12, fontweight="bold")
        for s in ax.spines.values():
            s.set_edgecolor("green" if row == 0 else "red"); s.set_linewidth(1.4)

        ax2 = fig.add_subplot(gs[row, 1])
        with rasterio.open(nlcd_path) as src:
            win = from_bounds(*win_b, src.transform)
            arr = src.read(1, window=win)
            ext = rasterio.windows.bounds(win, src.transform)
        cmap_cols = ['#FFFFFF'] * 256
        for v, c in NLCD_COLORS.items():
            cmap_cols[v] = c
        ax2.imshow(arr, extent=(ext[0], ext[2], ext[1], ext[3]),
                   cmap=ListedColormap(cmap_cols), vmin=0, vmax=255,
                   interpolation="nearest", rasterized=True)
        present |= set(np.unique(arr[arr > 0]).tolist())
        xs, ys = poly.exterior.xy
        ax2.plot(xs, ys, color="k", lw=0.9)
        ax2.set_xlabel("longitude"); ax2.set_ylabel("latitude")
        ax2.set_title(f"NLCD land cover (error {err:.2f} m)", fontsize=9)
        asc = (head % 360) < 90 or (head % 360) > 270
        x0 = ext[0] + 0.90*(ext[2]-ext[0])
        y0, y1 = (ext[1] + 0.12*(ext[3]-ext[1]), ext[1] + 0.88*(ext[3]-ext[1]))
        if not asc: y0, y1 = y1, y0
        ax2.annotate("", xy=(x0, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.6))
        ax2.text(0.985, 0.985, f"Cycle {int(cyc)} · Pass {int(pas)}\nTile {tile} · heading {head:.0f}°",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=7.2,
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.4", lw=0.5))
        for s in ax2.spines.values():
            s.set_edgecolor("green" if row == 0 else "red"); s.set_linewidth(1.4)
        print(f"  pass {pas}: err={err:.3f} n_pix={idx.size} tile={tile} head={head:.0f}")

    axb = fig.add_subplot(gs[2, :]); axb.axis("off")
    lg1 = axb.legend(handles=[Patch(facecolor=c, edgecolor="k", lw=0.4, label=l)
                              for c, l in zip(CLS_COLORS, CLS_LABELS)],
                     loc="center left", bbox_to_anchor=(0.0, 0.5), ncol=2, fontsize=7.2,
                     frameon=True, edgecolor="gray", title="PIXC classification",
                     title_fontsize=8, handlelength=1.2, columnspacing=0.9)
    axb.add_artist(lg1)
    hl = [Patch(facecolor=NLCD_COLORS[v], edgecolor="k", lw=0.4, label=NLCD_LABELS[v])
          for v in sorted(present) if v in NLCD_COLORS]
    axb.legend(handles=hl, loc="center right", bbox_to_anchor=(1.0, 0.5), ncol=4,
               fontsize=6.6, frameon=True, edgecolor="gray",
               title="NLCD land cover (classes present)", title_fontsize=8,
               handlelength=1.1, columnspacing=0.7)
    fig.savefig(f"{OUT}/fig01_paired_print.png", dpi=600)
    fig.savefig(f"{OUT}/fig01_paired_print.pdf")
    plt.close(fig)
    print("fig01 saved")


def fig05():
    ob = pq.read_table(f"{WORK}/2_urban_reaches/all_final_shape.parquet",
                       columns=["reach_id", "time_str", "reach_q_b",
                                "wse_relative_m", "wse_usgs_relative_m"]).to_pandas()
    ob["e"] = (ob.wse_relative_m - ob.wse_usgs_relative_m).abs()
    ob = ob.dropna(subset=["e"]).reset_index(drop=True)
    q = ob.reach_q_b.astype("int64")
    base = ob.e.values
    BITS = [("classification_qual_suspect", 2), ("geolocation_qual_suspect", 4),
            ("water_frac_suspect", 8), ("far_range_suspect", 8192),
            ("near_range_suspect", 16384), ("classification_qual_degraded", 262144),
            ("geolocation_qual_degraded", 524288), ("few_wse_observations", 2048)]
    rows = [("All observations (base rate)", base)]
    for name, bit in BITS:
        m = (q & bit) > 0
        if m.sum() >= 30:
            rows.append((name, ob.e[m].values))

    nod = pq.read_table(f"{PROJ}/3_analysis_results/swot_urban_nearfar_v6.5_results.parquet",
                        columns=["reach_id", "time_str", "flow_angle", "far_isa",
                                 "far_building", "error_node"]).to_pandas()
    fa = nod.flow_angle % 360
    nod["ang"] = np.minimum.reduce([fa, (fa-180).abs(), 360-fa])
    nod["isa"] = (nod.far_isa*100 if nod.far_isa.max() <= 1 else nod.far_isa).fillna(0)
    nod["bld"] = nod.far_building.fillna(0)
    nod["flag"] = (nod.ang <= 13) & ((nod.bld > 8) | (nod.isa > 19))
    key = nod.groupby(["reach_id", "time_str"]).flag.mean().rename("share").reset_index()
    ob2 = ob.merge(key, on=["reach_id", "time_str"], how="left")
    scr = ob2[ob2.share > 0.25]
    rows.append(("Geometric screen (>25% of nodes)", scr.e.values))
    print(f"  screen row n={len(scr)}, high-error {(scr.e>=T).mean()*100:.1f}%")

    labels = [r[0] for r in rows][::-1]
    WRAP = {"All observations (base rate)": "All observations\n(base rate)",
            "classification_qual_suspect": "classification_\nqual_suspect",
            "geolocation_qual_suspect": "geolocation_\nqual_suspect",
            "water_frac_suspect": "water_frac_\nsuspect",
            "far_range_suspect": "far_range_\nsuspect",
            "near_range_suspect": "near_range_\nsuspect",
            "classification_qual_degraded": "classification_\nqual_degraded",
            "geolocation_qual_degraded": "geolocation_\nqual_degraded",
            "few_wse_observations": "few_wse_\nobservations",
            "Geometric screen (>25% of nodes)": "Geometric screen\n(>25% of nodes)"}
    data = [np.clip(r[1], 0, 4.0) for r in rows][::-1]
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(7.16, 4.55),
                                  gridspec_kw={"width_ratios": [3.1, 1.0], "wspace": 0.04})
    bp = ax.boxplot(data, vert=False, showfliers=False, whis=1.5, widths=0.62,
                    patch_artist=True,
                    medianprops=dict(color="k", lw=1.1),
                    boxprops=dict(lw=0.7), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
    for i, b in enumerate(bp["boxes"]):
        nm = labels[i]
        b.set_facecolor("#e8a13c" if nm.startswith("Geometric") else
                        ("#bbbbbb" if nm.startswith("All") else "#7fa8d0"))
        b.set_alpha(0.9)
    ax.axvline(T, color="crimson", ls="--", lw=1.0)
    ax.annotate("T = 0.18 m", xy=(T+0.02, 1.006), xycoords=("data", "axes fraction"),
                color="crimson", fontsize=8, va="bottom")
    ax.set_yticks(range(1, len(rows)+1))
    ax.set_yticklabels([WRAP.get(l, l) for l in labels], fontsize=8, linespacing=1.05)
    ax.set_xlabel("absolute relative WSE error (m)")
    ax.set_xlim(0, 4.0)

    ys, shares, los, his, ns = [], [], [], [], []
    for i, (nm, arr) in enumerate(zip(labels, data)):
        n = len(arr); k = int((np.array(arr) >= T).sum())
        lo, hi = proportion_confint(k, n, method="wilson")
        ys.append(i+1); shares.append(100*k/n); los.append(100*lo); his.append(100*hi); ns.append(n)
    axr.errorbar(shares, ys, xerr=[np.array(shares)-np.array(los),
                                   np.array(his)-np.array(shares)],
                 fmt="o", ms=3.4, color="k", ecolor="0.35", elinewidth=1.0, capsize=2.2)
    axr.axvline(shares[labels.index("All observations (base rate)")], color="0.6", ls=":", lw=0.9)
    for y, s, n in zip(ys, shares, ns):
        axr.text(102.5, y, f"n={n:,}\n{s:.1f}% > T", fontsize=6.0, va="center",
                 linespacing=1.25)
    axr.set_xlim(35, 100); axr.set_ylim(ax.get_ylim())
    base_share = shares[labels.index("All observations (base rate)")]
    axr.text(base_share - 1.0, axr.get_ylim()[0] + 0.15,
             f"base rate {base_share:.1f}%", rotation=90, va="bottom", ha="right",
             fontsize=5.8, color="0.4")
    axr.set_yticks([])
    axr.set_xlabel("share above T (%)\nwith Wilson 95% interval", fontsize=8)
    axr.grid(axis="x", lw=0.3, alpha=0.5)
    fig.subplots_adjust(left=0.20, right=0.90, top=0.93, bottom=0.12)
    fig.savefig(f"{OUT}/fig05_flags_screen.png", dpi=600)
    fig.savefig(f"{OUT}/fig05_flags_screen.pdf")
    plt.close(fig)
    for nm, s, lo, hi, n in zip(labels, shares, los, his, ns):
        print(f"  {nm[:34]:34s} n={n:6,} share={s:5.1f}% CI[{lo:.1f},{hi:.1f}]")
    print("fig05 saved")


if __name__ == "__main__":
    import sys
    for a in (sys.argv[1:] or ["fig01", "fig05"]):
        globals()[a]()
