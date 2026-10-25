"""2026-10-05: count reaches in the gauge-reach matchup file (does [108]'s 1,723 come from it?). Reads two columns only."""
import pyarrow.parquet as pq
W = "/work/pi_kandread_umass_edu/swot-urban"
f = f"{W}/1_data_processing/usgs_sword_matchups_merritt_v2.parquet"
names = pq.read_schema(f).names; print("columns:", len(names), [n for n in names if n in ("reach_id", "site_no", "gage_id")])
cols = [c for c in ("reach_id", "site_no") if c in names]
m = pq.read_table(f, columns=cols).to_pandas()
print("rows", len(m), "unique reaches", m.reach_id.nunique(), "unique sites", m.site_no.nunique() if "site_no" in m else "n/a")
ar = pq.read_table(f"{W}/2_urban_reaches/all_reaches_vD.parquet", columns=["reach_id"]).to_pandas()
print("409 in matchup:", int(ar.reach_id.astype("int64").drop_duplicates().isin(m.reach_id.astype("int64")).sum()))
l = pq.read_table(f"{W}/1_data_processing/reaches_nlcd_merritt_v2.parquet", columns=["reach_id"]).to_pandas() if True else None
print("reaches_nlcd_merritt_v2 unique reaches", l.reach_id.nunique())
q = pq.read_table(f"{W}/2_urban_reaches/swotall_qc_filtered_vD.parquet", columns=["reach_id"]).to_pandas()
print("swotall_qc_filtered_vD unique reaches", q.reach_id.nunique(), "| matchup ∩ QC", len(set(m.reach_id.astype('int64')) & set(q.reach_id.astype('int64'))))
