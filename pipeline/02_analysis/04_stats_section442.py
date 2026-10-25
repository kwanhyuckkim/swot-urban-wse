"""
Compute all statistics needed for Section 4.4.2 rewrite.
Uses the same data structure and bin definitions as the Figure 8 plotting code.
"""
import pandas as pd
import numpy as np
from scipy import stats

# ─── Load data ───
DATA_PATH = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/swot_urban_nearfar_v6.5_results.parquet"
df = pd.read_parquet(DATA_PATH)
df['abs_error_node'] = df['error_node'].abs()
THRESHOLD = 0.1832

# ─── Flow angle zone classification ───
def classify_flow_angle(angle):
    a = angle % 180
    if a <= 10 or a >= 170:
        return 'Parallel'
    elif 80 <= a <= 100:
        return 'Perpendicular'
    else:
        return 'Other'

df['flow_zone'] = df['flow_angle'].apply(classify_flow_angle)

# ─── ISA bin function (same as plotting code) ───
def get_isa_bin(isa):
    if isa < 5:
        return '0-5'
    elif isa < 10:
        return '5-10'
    elif isa < 20:
        return '10-20'
    elif isa < 30:
        return '20-30'
    elif isa < 45:
        return '30-45'
    elif isa < 60:
        return '45-60'
    else:
        return '60+'

ISA_ORDER = ['0-5', '5-10', '10-20', '20-30', '30-45', '45-60', '60+']

# ─── Building height bin function (same as plotting code) ───
def get_building_bins_dynamic(series, interval=2, min_samples=500, start_from=5):
    bins = [0, start_from]
    current = start_from
    while current < series.max():
        bins.append(current + interval)
        current += interval

    labels = []
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            labels.append(f'{bins[i]}+')
        else:
            labels.append(f'{bins[i]}-{bins[i+1]}')

    result = pd.cut(series, bins=bins + [np.inf] if bins[-1] < series.max() else bins,
                     labels=labels[:len(bins)-1] if bins[-1] >= series.max() else labels,
                     right=False)

    # Merge small bins from right
    value_counts = result.value_counts()
    merge_from = None
    for label in reversed(labels):
        if label in value_counts.index and value_counts[label] < min_samples:
            merge_from = label
        else:
            break

    if merge_from is not None:
        merge_idx = labels.index(merge_from)
        merge_label = f'{labels[merge_idx].split("-")[0]}+'
        for label in labels[merge_idx:]:
            result = result.cat.rename_categories({label: merge_label}) if label in result.cat.categories else result
        # Remove duplicate categories
        result = result.astype(str)
    else:
        result = result.astype(str)

    return result

# ─── Filter to PARALLEL zone (same as Figure 8) ───
df_par = df[df['flow_zone'] == 'Parallel'].copy()
print(f"Total samples: {len(df)}")
print(f"Parallel zone samples: {len(df_par)}")
print()

# ─── Apply bins ───
df_par['isa_bin'] = df_par['far_isa'].apply(get_isa_bin)
df_par['bldg_bin'] = get_building_bins_dynamic(df_par['far_building'])

# Get the actual building height bin order
bldg_bins_unique = df_par['bldg_bin'].unique()
def sort_key(x):
    return float(x.replace('+', '').split('-')[0])
BLDG_ORDER = sorted(bldg_bins_unique, key=sort_key)

print(f"Building height bins: {BLDG_ORDER}")
print(f"ISA bins: {ISA_ORDER}")
print()

# ═══════════════════════════════════════════════════════════
# 1. BUILDING HEIGHT: Overall 68th percentile per bin
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("1. BUILDING HEIGHT — 68th percentile error per bin (Parallel zone)")
print("=" * 70)
for b in BLDG_ORDER:
    subset = df_par[df_par['bldg_bin'] == b]['abs_error_node']
    p68 = subset.quantile(0.68)
    med = subset.median()
    p25 = subset.quantile(0.25)
    p75 = subset.quantile(0.75)
    exceed = (subset > THRESHOLD).mean() * 100
    print(f"  {b:>8s}:  n={len(subset):>6d}  median={med:.3f}m  68th={p68:.3f}m  "
          f"IQR=[{p25:.3f}, {p75:.3f}]  exceed>{THRESHOLD}m: {exceed:.1f}%")
print()

# ═══════════════════════════════════════════════════════════
# 2. ISA: Overall 68th percentile per bin (Parallel zone)
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("2. ISA — 68th percentile error per bin (Parallel zone)")
print("=" * 70)
for isa in ISA_ORDER:
    subset = df_par[df_par['isa_bin'] == isa]['abs_error_node']
    if len(subset) == 0:
        continue
    p68 = subset.quantile(0.68)
    med = subset.median()
    p25 = subset.quantile(0.25)
    p75 = subset.quantile(0.75)
    exceed = (subset > THRESHOLD).mean() * 100
    print(f"  {isa:>8s}:  n={len(subset):>6d}  median={med:.3f}m  68th={p68:.3f}m  "
          f"IQR=[{p25:.3f}, {p75:.3f}]  exceed>{THRESHOLD}m: {exceed:.1f}%")
print()

# ═══════════════════════════════════════════════════════════
# 3. CROSS-TAB: Building height x ISA, 68th percentile
#    (This is the key Figure 8 top panel data)
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("3. CROSS-TAB: Building Height × ISA — 68th percentile (Parallel)")
print("=" * 70)
print(f"{'Bldg\\ISA':>10s}", end="")
for isa in ISA_ORDER:
    print(f"  {isa:>8s}", end="")
print("      n_total")
print("-" * 90)

for b in BLDG_ORDER:
    print(f"{b:>10s}", end="")
    row_n = 0
    for isa in ISA_ORDER:
        subset = df_par[(df_par['bldg_bin'] == b) & (df_par['isa_bin'] == isa)]['abs_error_node']
        row_n += len(subset)
        if len(subset) >= 30:
            p68 = subset.quantile(0.68)
            print(f"  {p68:>7.3f}m", end="")
        else:
            print(f"  {'n<30':>8s}", end="")
    print(f"  {row_n:>8d}")
print()

# ─── Same table but with sample sizes ───
print("=" * 70)
print("3b. CROSS-TAB: Building Height × ISA — SAMPLE SIZES")
print("=" * 70)
print(f"{'Bldg\\ISA':>10s}", end="")
for isa in ISA_ORDER:
    print(f"  {isa:>8s}", end="")
print()
print("-" * 80)

for b in BLDG_ORDER:
    print(f"{b:>10s}", end="")
    for isa in ISA_ORDER:
        subset = df_par[(df_par['bldg_bin'] == b) & (df_par['isa_bin'] == isa)]
        print(f"  {len(subset):>8d}", end="")
    print()
print()

# ═══════════════════════════════════════════════════════════
# 4. CROSS-TAB: ISA x Building height, 68th percentile
#    (This is the key Figure 8 bottom panel data)
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("4. CROSS-TAB: ISA × Building Height — 68th percentile (Parallel)")
print("=" * 70)
print(f"{'ISA\\Bldg':>10s}", end="")
for b in BLDG_ORDER:
    print(f"  {b:>8s}", end="")
print("      n_total")
print("-" * 90)

for isa in ISA_ORDER:
    print(f"{isa:>10s}", end="")
    row_n = 0
    for b in BLDG_ORDER:
        subset = df_par[(df_par['isa_bin'] == isa) & (df_par['bldg_bin'] == b)]['abs_error_node']
        row_n += len(subset)
        if len(subset) >= 30:
            p68 = subset.quantile(0.68)
            print(f"  {p68:>7.3f}m", end="")
        else:
            print(f"  {'n<30':>8s}", end="")
    print(f"  {row_n:>8d}")
print()

# ═══════════════════════════════════════════════════════════
# 5. NON-MONOTONIC ANALYSIS
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("5. NON-MONOTONIC BUILDING HEIGHT ANALYSIS")
print("=" * 70)

# Find peak bin and reversal
bldg_p68 = {}
for b in BLDG_ORDER:
    subset = df_par[df_par['bldg_bin'] == b]['abs_error_node']
    bldg_p68[b] = subset.quantile(0.68)

peak_bin = max(bldg_p68, key=bldg_p68.get)
peak_val = bldg_p68[peak_bin]
print(f"  Peak error bin: {peak_bin} → 68th percentile = {peak_val:.3f}m")
print()

# Compare consecutive bins for reversal
for i in range(len(BLDG_ORDER) - 1):
    b1, b2 = BLDG_ORDER[i], BLDG_ORDER[i + 1]
    v1, v2 = bldg_p68[b1], bldg_p68[b2]
    direction = "↑" if v2 > v1 else "↓" if v2 < v1 else "="
    pct_change = ((v2 - v1) / v1) * 100
    print(f"  {b1} → {b2}: {v1:.3f}m → {v2:.3f}m  {direction}  ({pct_change:+.1f}%)")

print()

# KS test between peak bin and next higher bin
peak_idx = BLDG_ORDER.index(peak_bin)
if peak_idx < len(BLDG_ORDER) - 1:
    next_bin = BLDG_ORDER[peak_idx + 1]
    s1 = df_par[df_par['bldg_bin'] == peak_bin]['abs_error_node']
    s2 = df_par[df_par['bldg_bin'] == next_bin]['abs_error_node']
    ks_stat, ks_p = stats.ks_2samp(s1, s2)
    print(f"  KS test ({peak_bin} vs {next_bin}): statistic={ks_stat:.4f}, p-value={ks_p:.2e}")

# Also check if last bin drops below peak
last_bin = BLDG_ORDER[-1]
if last_bin != peak_bin:
    s1 = df_par[df_par['bldg_bin'] == peak_bin]['abs_error_node']
    s2 = df_par[df_par['bldg_bin'] == last_bin]['abs_error_node']
    ks_stat, ks_p = stats.ks_2samp(s1, s2)
    print(f"  KS test ({peak_bin} vs {last_bin}): statistic={ks_stat:.4f}, p-value={ks_p:.2e}")
print()

# ═══════════════════════════════════════════════════════════
# 6. MULTIPLICATIVE INTERACTION
#    Within specific building height bins, how much does ISA matter?
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("6. MULTIPLICATIVE INTERACTION: ISA effect within each height bin")
print("=" * 70)
for b in BLDG_ORDER:
    isa_vals = {}
    for isa in ISA_ORDER:
        subset = df_par[(df_par['bldg_bin'] == b) & (df_par['isa_bin'] == isa)]['abs_error_node']
        if len(subset) >= 30:
            isa_vals[isa] = subset.quantile(0.68)

    if len(isa_vals) >= 2:
        min_isa = min(isa_vals, key=isa_vals.get)
        max_isa = max(isa_vals, key=isa_vals.get)
        ratio = isa_vals[max_isa] / isa_vals[min_isa] if isa_vals[min_isa] > 0 else float('inf')
        print(f"  {b:>8s}: lowest ISA={min_isa} ({isa_vals[min_isa]:.3f}m) → "
              f"highest ISA={max_isa} ({isa_vals[max_isa]:.3f}m)  ratio={ratio:.2f}x")
    else:
        print(f"  {b:>8s}: insufficient data")
print()

# ═══════════════════════════════════════════════════════════
# 7. SPEARMAN CORRELATIONS
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("7. SPEARMAN CORRELATIONS (Parallel zone)")
print("=" * 70)

# Overall: building height vs error
rho_bldg, p_bldg = stats.spearmanr(df_par['far_building'], df_par['abs_error_node'])
print(f"  Building height vs error (all): ρ={rho_bldg:.4f}, p={p_bldg:.2e}")

# Overall: ISA vs error
rho_isa, p_isa = stats.spearmanr(df_par['far_isa'], df_par['abs_error_node'])
print(f"  ISA vs error (all):             ρ={rho_isa:.4f}, p={p_isa:.2e}")

# Partial: ISA vs error conditioned on height bins
print()
print("  ISA vs error, conditioned on building height bin:")
for b in BLDG_ORDER:
    subset = df_par[df_par['bldg_bin'] == b]
    if len(subset) >= 30:
        rho, p = stats.spearmanr(subset['far_isa'], subset['abs_error_node'])
        print(f"    {b:>8s}: ρ={rho:.4f}, p={p:.2e}, n={len(subset)}")

# Partial: Building vs error conditioned on ISA bins
print()
print("  Building height vs error, conditioned on ISA bin:")
for isa in ISA_ORDER:
    subset = df_par[df_par['isa_bin'] == isa]
    if len(subset) >= 30:
        rho, p = stats.spearmanr(subset['far_building'], subset['abs_error_node'])
        print(f"    {isa:>8s}: ρ={rho:.4f}, p={p:.2e}, n={len(subset)}")

print()

# ═══════════════════════════════════════════════════════════
# 8. THRESHOLD EXCEEDANCE RATES (cross-tab)
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("8. THRESHOLD EXCEEDANCE RATES (>{:.4f}m) — Building × ISA".format(THRESHOLD))
print("=" * 70)
print(f"{'Bldg\\ISA':>10s}", end="")
for isa in ISA_ORDER:
    print(f"  {isa:>8s}", end="")
print()
print("-" * 80)

for b in BLDG_ORDER:
    print(f"{b:>10s}", end="")
    for isa in ISA_ORDER:
        subset = df_par[(df_par['bldg_bin'] == b) & (df_par['isa_bin'] == isa)]['abs_error_node']
        if len(subset) >= 30:
            exceed = (subset > THRESHOLD).mean() * 100
            print(f"  {exceed:>7.1f}%", end="")
        else:
            print(f"  {'n<30':>8s}", end="")
    print()
print()

# ═══════════════════════════════════════════════════════════
# 9. SUMMARY: Key numbers for the narrative
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("9. KEY NARRATIVE NUMBERS")
print("=" * 70)

# Lowest height bin error
lowest_b = BLDG_ORDER[0]
lowest_err = bldg_p68[lowest_b]
print(f"  Lowest building bin ({lowest_b}): 68th = {lowest_err:.3f}m")

# Peak bin
print(f"  Peak building bin ({peak_bin}): 68th = {peak_val:.3f}m")
print(f"  Peak / Lowest ratio: {peak_val/lowest_err:.2f}x")

# Reversal
if peak_idx < len(BLDG_ORDER) - 1:
    last_val = bldg_p68[BLDG_ORDER[-1]]
    print(f"  Last building bin ({BLDG_ORDER[-1]}): 68th = {last_val:.3f}m")
    print(f"  Reversal from peak: {((peak_val - last_val)/peak_val)*100:.1f}% reduction")

# ISA range in peak building bin
peak_isa_vals = {}
for isa in ISA_ORDER:
    subset = df_par[(df_par['bldg_bin'] == peak_bin) & (df_par['isa_bin'] == isa)]['abs_error_node']
    if len(subset) >= 30:
        peak_isa_vals[isa] = subset.quantile(0.68)

if peak_isa_vals:
    min_isa_k = min(peak_isa_vals, key=peak_isa_vals.get)
    max_isa_k = max(peak_isa_vals, key=peak_isa_vals.get)
    print(f"  In peak bin ({peak_bin}), ISA range: {peak_isa_vals[min_isa_k]:.3f}m ({min_isa_k}) "
          f"→ {peak_isa_vals[max_isa_k]:.3f}m ({max_isa_k}), "
          f"ratio={peak_isa_vals[max_isa_k]/peak_isa_vals[min_isa_k]:.2f}x")

print()
print("DONE. Copy everything above and paste into the chat.")
