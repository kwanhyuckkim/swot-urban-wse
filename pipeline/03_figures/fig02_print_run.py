#!/usr/bin/env python
"""Print-size rerun of Figure 2 (study map): fonts raised so every label prints
at 8 pt or larger at double-column width."""
import geopandas as gpd
from fig02_study_map_revision import create_paper_figure

W = '/work/pi_kandread_umass_edu/swot-urban'
OUT = '/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/figures_print'
all_reaches = gpd.read_parquet(f'{W}/2_urban_reaches/all_reaches_vD.parquet')
selected_12 = gpd.read_parquet(f'{W}/2_urban_reaches/selected_12_vD.parquet')
create_paper_figure(
    all_reaches, selected_12,
    sword_shp_path=f'{W}/1_data_processing/reaches_v1.shp',
    usgs_csv_path=f'{W}/0_data_collection/5_sword_usgs/paired_reach_SWOT_gage.csv',
    nlcd_path=f'{W}/0_data_collection/1_nlcd/Annual_NLCD_LndCov_2024_CU_C1V1_EPSG4326.tif',
    save_path=f'{OUT}/fig02_study_map_print.png', overview_bg='nlcd',
    figsize=(18, 19), hspace=0.24, wspace=0.12,
    fs_title=27, fs_legend=27, fs_tick=26, fs_site_num=22,
    fs_nlcd_legend=20, fs_site_title=20, fs_data_legend=26, dpi=170)
