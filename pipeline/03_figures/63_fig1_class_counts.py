#!/usr/bin/env python
"""2026-10-05: PIXC class counts inside the Figure 1 reach polygon for the two plotted passes (cycle 039, pass 272 tile 082R
and pass 315 tile 227R), with the same file and polygon selection as fig01() in 30_fig01_fig05_print.py. Checks the caption
claim that panel (b) shows low-coherence open water."""
import re
from pathlib import Path
import numpy as np, geopandas as gpd, xarray as xr, shapely
WORK = "/work/pi_kandread_umass_edu/swot-urban"
PROJ = "/project/pi_kandread_umass_edu/swot-urban"
NAMES = {1: 'land', 2: 'land_near_water', 3: 'water_near_land', 4: 'open_water', 5: 'dark_water',
         6: 'low_coh_water_near_land', 7: 'open_low_coh_water'}


def best(files):
    keep = {}
    for f in files:
        m = re.match(r"(.+)_(P[IG]D\d)_(\d+)\.nc", Path(f).name)
        if not m:
            continue
        b, pr, c = m.group(1), m.group(2), int(m.group(3))
        if b not in keep or (pr == "PGD0" and keep[b][1] == "PID0") or (pr == keep[b][1] and c > keep[b][2]):
            keep[b] = (f, pr, c)
    return [v[0] for v in keep.values()]


fct = "0.06"
sites = gpd.read_file(f"{WORK}/2_urban_reaches/selected_12_vD.shp")
col = "fctimp_mea" if "fctimp_mea" in sites.columns else "fctimp_mean"
poly = sites[np.isclose(sites[col], float(fct), atol=0.01)].to_crs("EPSG:4326").geometry.iloc[0]
b = poly.bounds
files = best(list(Path(f"{PROJ}/0_data_collection/SWOT_PIXC/fctimp_{fct}").glob("SWOT_L2_HR_PIXC_*.nc")))
for f in files:
    parts = Path(f).name.split("_")
    cyc, pas, tile = parts[4], parts[5], parts[6]
    if (cyc, pas, tile) not in [("039", "272", "082R"), ("039", "315", "227R")]:
        continue
    ds = xr.open_dataset(f, group="pixel_cloud")
    lon, lat = ds.longitude.values, ds.latitude.values
    idx = np.where((lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3]))[0]
    idx = idx[shapely.contains_xy(poly, lon[idx], lat[idx])]
    cls = ds.classification.values[idx].astype(int)
    ds.close()
    n = len(cls); plotted = (cls >= 3) & (cls <= 7)
    print(f"pass {pas} tile {tile} cycle {cyc}: {n} pixels in polygon, {plotted.sum()} plotted (classes 3-7)  [{Path(f).name}]")
    for k in range(1, 8):
        c = int((cls == k).sum())
        print(f"   {k} {NAMES[k]:<24} {c:>7}  {100*c/max(n,1):5.1f}% of all  {100*c/max(plotted.sum(),1):5.1f}% of plotted" if k >= 3 else f"   {k} {NAMES[k]:<24} {c:>7}  {100*c/max(n,1):5.1f}% of all")
