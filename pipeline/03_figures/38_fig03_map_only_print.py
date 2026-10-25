#!/usr/bin/env python
"""2026-10-04: Figure 3 with the map panel only (user decision: drop schematic panels (b), (c)).

Same drawing as fig03_orig() in 32_originals_print.py for the former panel (a): the
pre-rendered core image fig03_panel_a_core.png placed on EPSG:3857 extents derived from the
Figure 1 corridor polygon (selected_12_vD.shp, fctimp_mea == 0.06), with lon/lat tick labels.
The panel letter and the panel title are dropped because the figure now has one panel.
Writes fig03_map_only.png (600 dpi) and .pdf into FIG_OUT (default: figures_print).
"""
import os
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer
from shapely.geometry import box

WORK = "/work/pi_kandread_umass_edu/swot-urban"
PROJ = "/project/pi_kandread_umass_edu/swot-urban"
CORE = f"{WORK}/code/03_figures/fig03_panel_a_core.png"
OUT = os.environ.get("FIG_OUT", f"{PROJ}/3_analysis_results/figures_print")
Path(OUT).mkdir(exist_ok=True)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

fig, ax = plt.subplots(figsize=(7.16, 4.35))
fig.subplots_adjust(left=0.108, right=0.985, top=0.975, bottom=0.145)
img = plt.imread(CORE)
sel = gpd.read_file(f"{WORK}/2_urban_reaches/selected_12_vD.shp")
smp = sel[np.isclose(sel["fctimp_mea"], 0.06, atol=0.01)]
b4 = smp.to_crs("EPSG:4326").total_bounds
eb = [b4[0] - 0.1, b4[1] - 0.005, b4[2] + 0.1, b4[3] + 0.005]
e3 = gpd.GeoDataFrame(geometry=[box(*eb)], crs="EPSG:4326").to_crs("EPSG:3857").total_bounds
ax.imshow(img, extent=[e3[0], e3[2], e3[1], e3[3]], aspect="auto")
tr = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{abs(tr.transform(x, 0)[0]):.2f}°W"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f"{tr.transform(0, y)[1]:.2f}°N"))
ax.tick_params(labelsize=7.2)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
ax.set_xlabel("Longitude", fontsize=8, fontweight="bold", labelpad=1.5)
ax.set_ylabel("Latitude", fontsize=8, fontweight="bold", labelpad=1.5)
fig.savefig(f"{OUT}/fig03_map_only.png", dpi=600, facecolor="white")
fig.savefig(f"{OUT}/fig03_map_only.pdf", facecolor="white")
plt.close(fig)
print(f"saved {OUT}/fig03_map_only.png and .pdf; image {img.shape}, extent {e3}")
