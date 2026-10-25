#!/usr/bin/env python
"""Version rule (PGD0 > PID0, then highest counter) for the pixel-diagnostic tables: tiles keyed by (tile, start), reach rows by (tile, start, reach_id)."""
import re, json, numpy as np, pandas as pd
PD = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/reviewer_checks/pixdiag"; R2 = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/round2"
PAT = re.compile(r"SWOT_L2_HR_PIXC_(\d{3})_(\d{3})_(\d{3}[LR])_(\d{8}T\d{6})_(\d{8}T\d{6})_(P\w{3})_(\d+)"); RANK = {"PGD0": 0, "PID0": 1}
rep = {}
for name, extra in [("pixdiag_tiles", []), ("pixdiag_reaches", ["reach_id"])]:
    d = pd.read_parquet(f"{PD}/{name}.parquet"); pp = d.granule.str.extract(PAT)
    d["_tile"], d["_t0"], d["_crank"], d["_cnt"] = pp[2], pp[3], pp[5].map(RANK).fillna(9), pd.to_numeric(pp[6], errors="coerce")
    key = ["_tile", "_t0"] + extra; d["_orig"] = np.arange(len(d))
    old = d.drop_duplicates(key, keep="first"); new = d.sort_values(key + ["_crank", "_cnt", "n_water"], ascending=[True] * len(key) + [True, False, False], kind="mergesort").drop_duplicates(key, keep="first")
    changed = int((old.set_index(key)["_orig"] != new.set_index(key)["_orig"].reindex(old.set_index(key).index)).sum())
    rep[name] = {"rows_in": int(len(d)), "rows_out": int(len(new)), "dup_keys": int(d[d.duplicated(key, keep=False)].groupby(key).ngroups), "choice_changed_vs_first": changed, "crid_kept": pp[5][new.index].value_counts().to_dict()}
    new.drop(columns=["_tile", "_t0", "_crank", "_cnt", "_orig"]).to_parquet(f"{PD}/{name}_dedup.parquet", index=False); print(name, rep[name], flush=True)
json.dump(rep, open(f"{R2}/dedup_pixdiag_report.json", "w"), indent=1)
