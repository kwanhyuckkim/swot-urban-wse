#!/usr/bin/env python
"""Apply the 12-site version rule to the three 409-reach PIXC scan outputs: per (node_id, t) keep the granule with CRID PGD0 before PID0,
then the highest product counter, then the most pixels. Writes *_dedup.parquet next to each original (originals untouched) and a JSON report."""
import re, json, numpy as np, pandas as pd, pyarrow.parquet as pq
RC = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/reviewer_checks"; R2 = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/round2"
PAT = re.compile(r"SWOT_L2_HR_PIXC_(\d{3})_(\d{3})_(\d{3}[LR])_(\d{8}T\d{6})_(\d{8}T\d{6})_(P\w{3})_(\d+)")
FILES = {"coh_all": f"{RC}/coh_all/coherence_all_reaches.parquet", "coh_matrix": f"{RC}/coh_matrix/coherence_matrix_all_reaches.parquet", "coh_matrix2": f"{RC}/coh_matrix2/matrix2_all_reaches.parquet"}
CRID_RANK = {"PGD0": 0, "PID0": 1}
report = {}
for name, path in FILES.items():
    d = pd.read_parquet(path); n0 = len(d)
    pp = d.granule.str.extract(PAT); d["_crid"] = pp[5]; d["_cnt"] = pd.to_numeric(pp[6], errors="coerce"); d["_tile"] = pp[2]
    d["_crank"] = d._crid.map(CRID_RANK).fillna(9)
    npix = next((c for c in ["n_pix_A", "n_pix"] if c in d.columns), None); d["_npix"] = d[npix] if npix else 0
    d = d[d.node_id > 0].copy(); d["_orig"] = np.arange(len(d))
    key = ["node_id", "t"]
    old = d.drop_duplicates(key, keep="first")                       # what every analysis script did so far
    srt = d.sort_values(key + ["_crank", "_cnt", "_npix"], ascending=[True, True, True, False, False], kind="mergesort")
    new = srt.drop_duplicates(key, keep="first")
    dup_keys = d.duplicated(key, keep=False)
    changed = old.set_index(key)["_orig"] != new.set_index(key)["_orig"].reindex(old.set_index(key).index)
    combos = d[dup_keys].groupby(key).apply(lambda s: "+".join(sorted(s._crid + "_" + s._cnt.astype("Int64").astype(str)))).value_counts()
    rep = {"rows_in": int(n0), "rows_node_gt0": int(len(d)), "rows_out": int(len(new)), "keys_with_duplicates": int(d[dup_keys].groupby(key).ngroups),
           "rows_dropped": int(len(d) - len(new)), "keys_where_choice_changed_vs_first_occurrence": int(changed.sum()),
           "crid_kept": new._crid.value_counts().to_dict(), "crid_kept_old_rule": old._crid.value_counts().to_dict(),
           "duplicate_combos": {str(k): int(v) for k, v in combos.head(12).items()}, "unparsed_granules": int(pp[5].isna().sum())}
    report[name] = rep; print(name, json.dumps(rep, indent=1), flush=True)
    out = path.replace(".parquet", "_dedup.parquet"); new.drop(columns=["_crid", "_cnt", "_tile", "_crank", "_npix", "_orig"]).to_parquet(out, index=False); print("wrote", out, flush=True)
json.dump(report, open(f"{R2}/dedup_versions_report.json", "w"), indent=1); print("saved report")
