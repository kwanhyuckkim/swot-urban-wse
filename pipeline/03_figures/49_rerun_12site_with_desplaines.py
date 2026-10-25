"""2026-10-05: rerun the 12-site cross-level / sweep / partial-correlation checks with Des Plaines included.
Same code as 06_reviewer_product_checks.py (run_cross_level_check, max_per_site=None; run_partial_check), output
redirected to reviewer_checks/rerun_20261005/ so the 08-05 tables are not overwritten. Then: does the rerun
reproduce the 166 stored rows, and how do the V.C 'B' numbers change with Des Plaines (site 12)?"""
import os, sys, importlib
import numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
rpc = importlib.import_module("06_reviewer_product_checks")  # importable by name so worker processes can unpickle _cross_level_one
OLD = rpc.OUT_DIR
rpc.OUT_DIR = f"{OLD}/rerun_20261005"; os.makedirs(rpc.OUT_DIR, exist_ok=True)
import multiprocessing as mp
mp.set_start_method("fork", force=True)  # workers inherit the loaded module; no re-import of this script
new = rpc.run_cross_level_check(max_per_site=None, workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), verbose=True)
old = pd.read_parquet(f"{OLD}/cross_level_pixc_vs_riversp.parquet")
k = ["reach_id", "time_str"]
j = old.merge(new, on=k, how="outer", suffixes=("_old", "_new"), indicator=True)
print("\nrow match:", j._merge.value_counts().to_dict())
both = j[j._merge == "both"]
for c in ("abs_error_m", "h_all", "delta_layover_m", "delta_bright_m", "layover_pixel_bias_m", "layover_significant_pct", "bright_land_fire_pct"):
    if f"{c}_old" in both:
        dif = (both[f"{c}_old"] - both[f"{c}_new"]).abs()
        print(f"  {c:26s} max |old-new| {dif.max():.3g}  (n={dif.notna().sum()})")
print("new rows by site isa:", j[j._merge == "right_only"].groupby("isa_new").size().to_dict() if "isa_new" in j else "-")


def bstats(o, lab):
    out = {}
    for tag, dcol in (("layover", "delta_layover_m"), ("bright", "delta_bright_m")):
        s = o[["error_m", "abs_error_m", dcol]].dropna()
        cor = (s.error_m + s[dcol]).abs()
        out[tag] = (len(s), int((cor < s.abs_error_m).sum()), s.abs_error_m.mean(), cor.mean())
    sb = o.delta_bright_m.dropna(); sl = o.layover_pixel_bias_m.dropna()
    print(f"\n[{lab}] passes {len(o)}, sites {o.isa.nunique()}")
    for tag, (n, better, before, after) in out.items():
        print(f"  exclude {tag:7s}: improves {better}/{n}; mean |e| {before:.3f} -> {after:.3f} m")
    print(f"  bright removal lowers pass median in {int((sb < 0).sum())}/{len(sb)}, median shift {sb.median():+.3f} m")
    print(f"  layover-pixel bias positive in {int((sl > 0).sum())}/{len(sl)}")


bstats(new, "rerun, all sites")
bstats(new[~np.isclose(new.isa, 0.61)], "rerun, without Des Plaines (should reproduce manuscript)")
bstats(new[np.isclose(new.isa, 0.61)], "Des Plaines only")
bstats(old, "stored 08-05 table")
print("\n--- partial correlation on the rerun sweep table")
pc = rpc.run_partial_check(verbose=True)
