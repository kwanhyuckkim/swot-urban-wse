"""2026-10-04: (1) which of the 12 selected sites supply the 166 passes (11 vs 12); (2) what the 12-site
pass tables contain; (3) whether the 409-reach streamed PIXC extractions keep per-pass pixel heights and flags."""
import glob, os
import pandas as pd, pyarrow.parquet as pq
W = "/work/pi_kandread_umass_edu/swot-urban"; P = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
RC = f"{P}/reviewer_checks"
s12 = pq.read_table(f"{W}/2_urban_reaches/selected_12_vD.parquet").to_pandas()
cols = [c for c in s12.columns if c in ("site_number", "reach_id", "river_name", "site_no", "fctimp_mean")]
print("selected_12:", len(s12)); print(s12[cols].to_string())
for name in ("pixc_layover_sweep", "cross_level_pixc_vs_riversp", "pixc_partial_correlation"):
    for f in glob.glob(f"{RC}/{name}*.parquet") + glob.glob(f"{P}/round2/{name}*.parquet"):
        d = pd.read_parquet(f)
        rid = "reach_id" if "reach_id" in d else None
        print(f"\n{f}: rows {len(d):,}; cols {list(d.columns)[:14]}")
        if rid:
            print("  reaches:", d[rid].nunique(), "| per reach:", d.groupby(rid).size().to_dict())
            if "reach_id" in s12:
                print("  selected_12 reaches with no rows:", sorted(set(s12.reach_id.astype('int64')) - set(d[rid].astype('int64'))))
print("\n--- streamed 409-reach extractions")
for d in ("pixdiag", "coh_all", "coh_matrix2"):
    for f in sorted(glob.glob(f"{RC}/{d}/*.parquet"))[:3]:
        sch = pq.read_schema(f)
        print(f"{f} ({os.path.getsize(f)/1e6:.0f} MB): {sch.names[:40]}")
    print(f"  {d}: {len(glob.glob(f'{RC}/{d}/*.parquet'))} parquet files")
