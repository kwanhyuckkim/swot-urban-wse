"""2026-10-05: verify the reach-selection chain behind Section III.A (independent check of review round 9).
(a) gauges with >= 90,000 valid stage records; (b) loss in the reach_id join between the v16 pairing file and the v17b buffer
polygons; (c) observations per reach in the final reach table; (d) |wse - p_wse| in the QC output vs the Table I "< 10 m" rule."""
import pickle, pandas as pd, pyarrow.parquet as pq
W = "/work/pi_kandread_umass_edu/swot-urban"
p = pd.read_csv(f"{W}/0_data_collection/5_sword_usgs/paired_reach_SWOT_gage.csv", dtype={"gage_id": object},
                usecols=lambda c: c in ("gage_agency", "gage_id", "reach_id", "sword_version"), low_memory=False)
u = p[p.gage_agency == "USGS"]; print("pairing USGS gauges", u.gage_id.nunique(), "reaches", u.reach_id.nunique(),
                                       "| sword_version", p.sword_version.value_counts().to_dict() if "sword_version" in p else "no column")
s90 = pickle.load(open("/work/kwanhyuckkim_umass_edu/_4_swot-urban/swot-urban/sword-usgsgage/sites_over_90k.pkl", "rb"))
k = u[u.gage_id.isin(s90)]; print("sites_over_90k", len(s90), "| pairing USGS ∩ 90k: gauges", k.gage_id.nunique(), "reaches", k.reach_id.nunique())
b = pq.read_table(f"{W}/1_data_processing/buffered_polygons_with_nlcd.parquet", columns=["reach_id"]).to_pandas()
b_ids = set(b.reach_id.astype("int64")); k_ids = set(k.reach_id.astype("int64"))
j = k[k.reach_id.astype("int64").isin(b_ids)]
print("after join with v17b buffers: reaches", j.reach_id.nunique(), "gauges", j.gage_id.nunique(), "| lost reaches", len(k_ids - b_ids))
q = pq.read_table(f"{W}/2_urban_reaches/swotall_qc_filtered_vD.parquet").to_pandas()
q["reach_id"] = q.reach_id.astype("int64"); print("QC reaches", q.reach_id.nunique(), "| join ∩ QC", len(set(j.reach_id.astype("int64")) & set(q.reach_id)))
if {"wse", "p_wse"} <= set(q.columns):
    dq = (q.wse - q.p_wse).abs(); print("QC |wse - p_wse| max %.2f m; obs >= 10 m: %d" % (dq.max(), int((dq >= 10).sum())))
    keep = set(q.reach_id[dq < 10]); print("reaches left in join ∩ QC if < 10 m:", len(set(j.reach_id.astype('int64')) & keep))
f = pq.read_table(f"{W}/2_urban_reaches/all_final_shape.parquet", columns=["reach_id"]).to_pandas()
c = f.reach_id.astype("int64").value_counts()
print("final reach table rows", len(f), "reaches", len(c), "| obs per reach min", int(c.min()), "median", float(c.median()), "| reaches < 5 obs", int((c < 5).sum()), "| 72582400501:", int(c.get(72582400501, 0)))
