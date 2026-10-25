"""2026-10-05: does 1,723 equal the number of SWORD reaches paired with a USGS gauge before the 90,000-record gauge filter?"""
import pandas as pd
p = pd.read_csv("/work/pi_kandread_umass_edu/swot-urban/0_data_collection/5_sword_usgs/paired_reach_SWOT_gage.csv", dtype={"gage_id": object}, low_memory=False)
print("columns:", list(p.columns)[:12])
print("rows", len(p), "unique reaches", p.reach_id.nunique(), "| agencies", p.gage_agency.value_counts().to_dict() if "gage_agency" in p else "n/a")
u = p[p.gage_agency == "USGS"] if "gage_agency" in p else p
print("USGS rows", len(u), "unique reaches", u.reach_id.nunique(), "unique gauges", u.gage_id.nunique())
if "gage_latitude" in u: print("USGS reaches with gauge lat > 50:", u[u.gage_latitude > 50].reach_id.nunique())
