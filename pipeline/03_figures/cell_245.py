"""
Masking optimization v2: geometry-split analysis
======================================================================
(a) Flow angle: All/Far/Near/Both/Neither error reduction + All removed
    Optimal threshold = max sum of slopes across categories -> 뒤에점
(b) Building height: parallel < opt_angle, far_building >= 5 m
(c) ISA: parallel < opt_angle, far_isa > 0
    Optimal threshold = max marginal efficiency (Δred / Δrm) -> 뒤에점

y1 = Cumulative 68th %ile error reduction (%)
y2 = Observations removed (%)

Run on server:  python optimize_mask_individual_v2.py
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# ═══════════════════════════════════════════════════════════════
# Publication style
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════
DATA_PATH = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/swot_urban_nearfar_v6.5_results.parquet"
COLS_NEEDED = ['flow_angle', 'far_building', 'far_isa', 'error_node',
               'near_isa', 'near_ratio', 'far_ratio']
MIN_RATIO = 0.50

print("Loading data ...")
df = pd.read_parquet(DATA_PATH, columns=COLS_NEEDED, engine='pyarrow')
df['abs_error'] = df['error_node'].abs()
df['flow_angle_parallel'] = df['flow_angle'].apply(
    lambda a: min(a % 180, 180 - a % 180))

if df['far_isa'].max() <= 1.0:
    df['far_isa_pct'] = df['far_isa'] * 100
else:
    df['far_isa_pct'] = df['far_isa']

# ─── Geometry categorisation (mutually exclusive) ───
has_far  = (df['far_isa'].fillna(0) > 0) & (df['far_ratio'].fillna(0) >= MIN_RATIO)
has_near = (df['near_isa'].fillna(0) > 0) & (df['near_ratio'].fillna(0) >= MIN_RATIO)

abs_err  = df['abs_error'].values
flow_par = df['flow_angle_parallel'].values
far_bldg = df['far_building'].values
far_isa  = df['far_isa_pct'].values

is_far_only  = (has_far & ~has_near).values
is_near_only = (has_near & ~has_far).values
is_both      = (has_far & has_near).values
is_neither   = (~has_far & ~has_near).values
is_all       = np.ones(len(df), dtype=bool)

n_total = len(df)
p68_total = np.quantile(abs_err, 0.68)

CAT_ORDER = ['All', 'Far', 'Near', 'Both', 'Neither']
CAT = {
    'All': is_all, 'Far': is_far_only, 'Near': is_near_only,
    'Both': is_both, 'Neither': is_neither,
}
COLORS = {
    'All': '#000000', 'Far': '#D62728', 'Near': '#2CA02C',
    'Both': '#1F77B4', 'Neither': '#7F7F7F',
}
LABELS = {
    'All': 'All Observations', 'Far': 'Far-Nadir Only',
    'Near': 'Near-Nadir Only', 'Both': 'Both Geometries',
    'Neither': 'Neither Geometry',
}

# Baselines per category
baselines = {}
print(f"\nTotal: {n_total:,}   68th %ile = {p68_total:.4f} m")
for name in CAT_ORDER:
    n = CAT[name].sum()
    p = np.quantile(abs_err[CAT[name]], 0.68) if n > 0 else np.nan
    baselines[name] = p
    print(f"  {name:<10s}: {n:>9,} ({n/n_total*100:>5.1f}%)  68th %ile = {p:.4f} m")
print()

# ═══════════════════════════════════════════════════════════════
# 1. FLOW ANGLE SWEEP
# ═══════════════════════════════════════════════════════════════
print("=" * 80)
print("1. FLOW ANGLE SWEEP")
print("=" * 80)

angle_thresholds = np.arange(1, 32, 2)
results_angle = {name: [] for name in CAT_ORDER}

for angle_t in angle_thresholds:
    remove = flow_par < angle_t
    for name in CAT_ORDER:
        mask = CAT[name]
        retained = mask & ~remove
        n_rm  = (mask & remove).sum()
        n_cat = mask.sum()
        if retained.sum() > 0:
            p68_ret = np.quantile(abs_err[retained], 0.68)
            pct_red = (baselines[name] - p68_ret) / baselines[name] * 100
        else:
            pct_red = np.nan
        results_angle[name].append({
            'threshold': angle_t,
            'pct_reduction': pct_red,
            'pct_removed': n_rm / n_cat * 100 if n_cat > 0 else 0,
        })
    vals = "  ".join(f"{name}={results_angle[name][-1]['pct_reduction']:>+5.1f}%"
                     for name in CAT_ORDER)
    print(f"  <{angle_t:>2.0f}°: {vals}")

print()

# ─── Optimal angle threshold (needed for sections 2 & 3) ───
slopes_a = []
for i in range(len(angle_thresholds) - 1):
    dx = float(angle_thresholds[i+1] - angle_thresholds[i])
    s = sum((results_angle[n][i+1]['pct_reduction'] - results_angle[n][i]['pct_reduction']) / dx
            for n in CAT_ORDER)
    slopes_a.append(s)
opt_step_a = int(np.argmax(slopes_a))
opt_idx_a  = opt_step_a + 1          # 뒤에점
opt_thresh_a = float(angle_thresholds[opt_idx_a])
print(f">>> Optimal angle threshold: {opt_thresh_a:.0f}°  "
      f"(max slope sum at {angle_thresholds[opt_step_a]:.0f}°→{opt_thresh_a:.0f}°)\n")

# ═══════════════════════════════════════════════════════════════
# 2. BUILDING HEIGHT SWEEP (parallel < opt_angle, far_bldg >= 5 m)
# ═══════════════════════════════════════════════════════════════
print("=" * 80)
print(f"2. BUILDING HEIGHT SWEEP  (parallel < {opt_thresh_a:.0f}°, bldg ≥ 5 m)")
print("=" * 80)

has_valid_bldg = far_bldg >= 0
is_par_bldg    = (flow_par < opt_thresh_a) & has_valid_bldg
n_sub_bldg     = is_par_bldg.sum()
p68_sub_bldg   = np.quantile(abs_err[is_par_bldg], 0.68)
print(f"   Subset: {n_sub_bldg:,},  68th %ile = {p68_sub_bldg:.4f} m\n")

bldg_thresholds = np.arange(20, 4.5, -0.5)
results_bldg = []

for h_t in bldg_thresholds:
    remove = is_par_bldg & (far_bldg > h_t)
    n_rm   = remove.sum()
    ret_sub = is_par_bldg & ~remove
    if ret_sub.sum() > 0:
        p68s = np.quantile(abs_err[ret_sub], 0.68)
        red_sub = (p68_sub_bldg - p68s) / p68_sub_bldg * 100
    else:
        red_sub = np.nan
    rm_sub = n_rm / n_sub_bldg * 100
    p68t = np.quantile(abs_err[~remove], 0.68)
    red_tot = (p68_total - p68t) / p68_total * 100
    rm_tot  = n_rm / n_total * 100

    results_bldg.append(dict(threshold=h_t, n_removed=n_rm,
                             red_sub=red_sub, rm_sub=rm_sub,
                             red_tot=red_tot, rm_tot=rm_tot))
    print(f"  >{h_t:>5.1f}m: rm={n_rm:>6,d}  sub[{red_sub:>+5.1f}% | {rm_sub:>5.1f}%]  "
          f"tot[{red_tot:>+5.2f}% | {rm_tot:>5.2f}%]")

print()

# ═══════════════════════════════════════════════════════════════
# 3. ISA SWEEP (parallel < opt_angle, far_isa > 0)
# ═══════════════════════════════════════════════════════════════
print("=" * 80)
print(f"3. FAR ISA SWEEP  (parallel < {opt_thresh_a:.0f}°, ISA > 0)")
print("=" * 80)

has_valid_isa = far_isa > 0
is_par_isa    = (flow_par < opt_thresh_a) & has_valid_isa
n_sub_isa     = is_par_isa.sum()
p68_sub_isa   = np.quantile(abs_err[is_par_isa], 0.68)
print(f"   Subset: {n_sub_isa:,},  68th %ile = {p68_sub_isa:.4f} m\n")

isa_thresholds = np.arange(63, 4, -2)
results_isa = []

for isa_t in isa_thresholds:
    remove = is_par_isa & (far_isa > isa_t)
    n_rm   = remove.sum()
    ret_sub = is_par_isa & ~remove
    if ret_sub.sum() > 0:
        p68s = np.quantile(abs_err[ret_sub], 0.68)
        red_sub = (p68_sub_isa - p68s) / p68_sub_isa * 100
    else:
        red_sub = np.nan
    rm_sub = n_rm / n_sub_isa * 100
    p68t    = np.quantile(abs_err[~remove], 0.68)
    red_tot = (p68_total - p68t) / p68_total * 100
    rm_tot  = n_rm / n_total * 100

    results_isa.append(dict(threshold=isa_t, n_removed=n_rm,
                            red_sub=red_sub, rm_sub=rm_sub,
                            red_tot=red_tot, rm_tot=rm_tot))
    print(f"  >{isa_t:>4.0f}%: rm={n_rm:>6,d}  sub[{red_sub:>+5.1f}% | {rm_sub:>5.1f}%]  "
          f"tot[{red_tot:>+5.2f}% | {rm_tot:>5.2f}%]")

print()

# ═══════════════════════════════════════════════════════════════
# 4. COMBINED MASKS
# ═══════════════════════════════════════════════════════════════
print("=" * 80)
print("4. COMBINED MASKS")
print("=" * 80)

combos = [
    ("No masking",           np.zeros(n_total, dtype=bool)),
    ("Bldg > 13 m",         is_par_bldg & (far_bldg > 13)),
    ("Bldg > 8 m",          is_par_bldg & (far_bldg > 8)),
    ("ISA > 40%",            is_par_isa  & (far_isa > 40)),
    ("ISA > 30%",            is_par_isa  & (far_isa > 30)),
    ("Bldg>8 OR ISA>30%",   (is_par_bldg & (far_bldg > 8)) | (is_par_isa & (far_isa > 30))),
    ("Bldg>13 OR ISA>30%",  (is_par_bldg & (far_bldg > 13)) | (is_par_isa & (far_isa > 30))),
    (f"All parallel <{opt_thresh_a:.0f}°", flow_par < opt_thresh_a),
]

print(f"\n{'Mask':<28s} {'Removed':>8s} {'%Tot':>7s} {'68th%ile':>9s} {'Red%':>7s}")
print("-" * 65)
for label, mask in combos:
    n_rm = mask.sum()
    p68r = np.quantile(abs_err[~mask], 0.68)
    red  = (p68_total - p68r) / p68_total * 100
    print(f"{label:<28s} {n_rm:>8,d} {n_rm/n_total*100:>6.2f}% {p68r:>9.4f} {red:>+6.2f}%")

# ═══════════════════════════════════════════════════════════════
# OPTIMAL THRESHOLDS (b, c)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("OPTIMAL THRESHOLDS")
print("=" * 80)
print(f"  (a) Flow angle : {opt_thresh_a:.0f}°")

# (b) Knee point: normalize red_sub & rm_sub to [0,1], find max(red_norm - rm_norm)
#     = point where error reduction leads sample removal the most (sweet spot)
rm_b  = np.array([r['rm_sub']  for r in results_bldg])
red_b = np.array([r['red_sub'] for r in results_bldg])
rm_b_norm  = rm_b  / rm_b.max()       if rm_b.max()        > 0 else rm_b
red_b_norm = red_b / np.nanmax(red_b)  if np.nanmax(red_b) > 0 else red_b
knee_b = red_b_norm - rm_b_norm
opt_idx_b = int(np.argmax(knee_b))
opt_thresh_b = results_bldg[opt_idx_b]['threshold']
print(f"  (b) Building ht: {opt_thresh_b:.1f} m  (knee score={knee_b[opt_idx_b]:.3f},  "
      f"red_sub={results_bldg[opt_idx_b]['red_sub']:.1f}% / "
      f"rm_sub={results_bldg[opt_idx_b]['rm_sub']:.1f}%)")

# (c) Knee point: same approach for ISA
rm_c  = np.array([r['rm_sub']  for r in results_isa])
red_c = np.array([r['red_sub'] for r in results_isa])
rm_c_norm  = rm_c  / rm_c.max()       if rm_c.max()        > 0 else rm_c
red_c_norm = red_c / np.nanmax(red_c)  if np.nanmax(red_c) > 0 else red_c
knee_c = red_c_norm - rm_c_norm
opt_idx_c = int(np.argmax(knee_c))
opt_thresh_c = results_isa[opt_idx_c]['threshold']
print(f"  (c) ISA        : {opt_thresh_c:.0f}%  (knee score={knee_c[opt_idx_c]:.3f},  "
      f"red_sub={results_isa[opt_idx_c]['red_sub']:.1f}% / "
      f"rm_sub={results_isa[opt_idx_c]['rm_sub']:.1f}%)")

# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════
print("\nMaking plots ...")

ANNO_FS = 13  # match tick label size

fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))

# ──────────────────────────────────────────────────────────────
# (a) Flow Angle
# ──────────────────────────────────────────────────────────────
ax = axes[0]
xa = [r['threshold'] for r in results_angle['All']]

for name in CAT_ORDER:
    vals = [r['pct_reduction'] for r in results_angle[name]]
    ax.plot(xa, vals, color=COLORS[name], linestyle='-', marker='o',
            markersize=4, linewidth=1.8, label=LABELS[name])

ax.set_xlabel('Flow Angle Threshold (°)')
ax.set_ylabel('Cumul. 68th %ile Error Reduction (%)')
ax.set_title('(a) Flow Angle Masking')
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)

ax2 = ax.twinx()
vals_rm = [r['pct_removed'] for r in results_angle['All']]
ax2.plot(xa, vals_rm, color='black', linestyle='--', linewidth=1.2,
         alpha=0.5, label='All Obs. Removed (%)')
ax2.set_ylabel('Observations Removed (%)', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

# Optimal threshold annotation
ax.axvline(x=opt_thresh_a, color='#444444', linestyle='--',
           linewidth=1.5, alpha=0.7, zorder=3)

for name in CAT_ORDER:
    val = results_angle[name][opt_idx_a]['pct_reduction']
    ax.plot(opt_thresh_a, val, 'o', color=COLORS[name], markersize=8,
            markeredgecolor='white', markeredgewidth=1.0, zorder=5)

rm_all = results_angle['All'][opt_idx_a]['pct_removed']
ax2.plot(opt_thresh_a, rm_all, 's', color='black', markersize=7,
         markeredgecolor='white', markeredgewidth=1.0, zorder=5)

# Text box
anno = [f'θ* = {opt_thresh_a:.0f}°']
for name in CAT_ORDER:
    v = results_angle[name][opt_idx_a]['pct_reduction']
    abs_cm = baselines[name] * v
    anno.append(f'  {LABELS[name]}: {v:.1f}% ({abs_cm:.1f} cm)')
anno.append(f'  Removed (All): {rm_all:.1f}%')
bbox_a = dict(boxstyle='round,pad=0.35', facecolor='#FFFFF0',
              edgecolor='gray', alpha=0.92)
trans_a = blended_transform_factory(ax.transData, ax.transAxes)
ax.text(opt_thresh_a + 1.2, 0.62, '\n'.join(anno), transform=trans_a,
        fontsize=9, va='top', family='monospace', bbox=bbox_a, zorder=6)

lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2,
          loc='upper left', fontsize=14, ncol=1,
          framealpha=0.9, edgecolor='gray')

# ──────────────────────────────────────────────────────────────
# Helper: annotate optimal on (b)/(c)
# ──────────────────────────────────────────────────────────────
def annotate_bc(ax_main, ax_right, opt_x, r, panel_label, p68_sub, p68_tot):
    """Draw vertical line, θ* label inside plot, and annotate intersections."""
    ax_main.axvline(x=opt_x, color='#444444', linestyle='--',
                    linewidth=1.5, alpha=0.7, zorder=3)

    abs_sub_cm = p68_sub * r['red_sub']
    abs_tot_cm = p68_tot * r['red_tot']

    # Intersection dots
    ax_main.plot(opt_x, r['red_sub'], 'o', color='blue', markersize=8,
                 markeredgecolor='white', markeredgewidth=1.0, zorder=5)
    ax_main.plot(opt_x, r['red_tot'], 'o', color='blue', markersize=6,
                 markeredgecolor='white', markeredgewidth=0.8, alpha=0.6, zorder=5)
    ax_right.plot(opt_x, r['rm_sub'], 's', color='red', markersize=7,
                  markeredgecolor='white', markeredgewidth=1.0, zorder=5)
    ax_right.plot(opt_x, r['rm_tot'], 's', color='red', markersize=5,
                  markeredgecolor='white', markeredgewidth=0.8, alpha=0.6, zorder=5)

    # Horizontal reference lines
    xlims = ax_main.get_xlim()
    ax_main.plot([xlims[0], opt_x], [r['red_sub'], r['red_sub']],
                 color='blue', linestyle=':', linewidth=0.8, alpha=0.4, zorder=2)
    ax_main.plot([xlims[0], opt_x], [r['red_tot'], r['red_tot']],
                 color='blue', linestyle=':', linewidth=0.6, alpha=0.3, zorder=2)
    ax_right.plot([opt_x, xlims[1]], [r['rm_sub'], r['rm_sub']],
                  color='red', linestyle=':', linewidth=0.8, alpha=0.4, zorder=2)
    ax_right.plot([opt_x, xlims[1]], [r['rm_tot'], r['rm_tot']],
                  color='red', linestyle=':', linewidth=0.6, alpha=0.3, zorder=2)

    # Value labels on y-axes
    ax_main.text(xlims[0], r['red_sub'],
                 f' {r["red_sub"]:.1f}%\n ({abs_sub_cm:.1f} cm)',
                 fontsize=ANNO_FS, color='blue', fontweight='bold', va='center')
    ax_main.text(xlims[0]-15, r['red_tot'],
                 f' {r["red_tot"]:.2f}%\n ({abs_tot_cm:.1f} cm)',
                 fontsize=ANNO_FS - 1, color='blue', alpha=0.7, va='center')
    ax_right.text(xlims[1], r['rm_sub'], f'{r["rm_sub"]:.1f}% ',
                  fontsize=ANNO_FS, color='red', fontweight='bold',
                  va='center', ha='right')
    ax_right.text(xlims[1], r['rm_tot'], f'{r["rm_tot"]:.2f}% ',
                  fontsize=ANNO_FS - 1, color='red', alpha=0.7,
                  va='center', ha='right')

    print(xlims[0] + 10)

    # θ* label inside plot, right of dashed line
    unit = 'm' if 'uilding' in panel_label else '%'
    x_range = abs(xlims[1] - xlims[0])
    text_x = opt_x - 0.03 * x_range   # inverted axis: lower x = right
    trans = blended_transform_factory(ax_main.transData, ax_main.transAxes)
    ax_main.text(text_x, 0.95, f'θ* = {opt_x:.0f} {unit}', fontsize=ANNO_FS,
                 fontweight='bold', color='#444444', ha='right', va='top',
                 transform=trans, zorder=6)

# ──────────────────────────────────────────────────────────────
# (b) Building Height
# ──────────────────────────────────────────────────────────────
ax = axes[1]
xb = [r['threshold'] for r in results_bldg]

ax.plot(xb, [r['red_sub'] for r in results_bldg],
        'b-o', markersize=4, linewidth=1.8,
        label=f'Parallel subset (n={n_sub_bldg:,})')
ax.plot(xb, [r['red_tot'] for r in results_bldg],
        color='blue', linestyle='--', linewidth=1.2, alpha=0.5,
        label=f'Total dataset (n={n_total:,})')
ax.set_xlabel('Far-Nadir Building Height Threshold (m)')
ax.set_ylabel('Cumul. 68th %ile Error Reduction (%)', color='blue')
ax.tick_params(axis='y', labelcolor='blue')
ax.set_title(f'(b) Building Height Masking (Parallel < {opt_thresh_a:.0f}°)')
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
ax.invert_xaxis()

ax2 = ax.twinx()
ax2.plot(xb, [r['rm_sub'] for r in results_bldg],
         'r-s', markersize=3, linewidth=1.4, label='Subset removed (%)')
ax2.plot(xb, [r['rm_tot'] for r in results_bldg],
         color='red', linestyle='--', linewidth=1.0, alpha=0.5,
         label='Total removed (%)')
ax2.set_ylabel('Observations Removed (%)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2,
          loc='upper left', fontsize=13, framealpha=0.9, edgecolor='gray')

annotate_bc(ax, ax2, opt_thresh_b, results_bldg[opt_idx_b],
            'Building', p68_sub_bldg, p68_total)

# ──────────────────────────────────────────────────────────────
# (c) ISA
# ──────────────────────────────────────────────────────────────
ax = axes[2]
xc = [r['threshold'] for r in results_isa]

ax.plot(xc, [r['red_sub'] for r in results_isa],
        'b-o', markersize=4, linewidth=1.8,
        label=f'Parallel subset (n={n_sub_isa:,})')
ax.plot(xc, [r['red_tot'] for r in results_isa],
        color='blue', linestyle='--', linewidth=1.2, alpha=0.5,
        label=f'Total dataset (n={n_total:,})')
ax.set_xlabel('Far-Nadir ISA Threshold (%)')
ax.set_ylabel('Cumul. 68th %ile Error Reduction (%)', color='blue')
ax.tick_params(axis='y', labelcolor='blue')
ax.set_title(f'(c) ISA Masking (Parallel < {opt_thresh_a:.0f}°)')
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
ax.invert_xaxis()

ax2 = ax.twinx()
ax2.plot(xc, [r['rm_sub'] for r in results_isa],
         'r-s', markersize=3, linewidth=1.4, label='Subset removed (%)')
ax2.plot(xc, [r['rm_tot'] for r in results_isa],
         color='red', linestyle='--', linewidth=1.0, alpha=0.5,
         label='Total removed (%)')
ax2.set_ylabel('Observations Removed (%)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2,
          loc='upper left', fontsize=14, framealpha=0.9, edgecolor='gray')

annotate_bc(ax, ax2, opt_thresh_c, results_isa[opt_idx_c],
            'ISA%', p68_sub_isa, p68_total)

# ──────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig('mask_individual_sweeps_v2.png', dpi=300, bbox_inches='tight')
plt.savefig('mask_individual_sweeps_v2.pdf', bbox_inches='tight')
print("Saved: mask_individual_sweeps_v2.png / .pdf")
print("\nDone!")

