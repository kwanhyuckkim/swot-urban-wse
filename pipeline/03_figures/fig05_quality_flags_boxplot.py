#!/usr/bin/env python
"""
Figure 5 (TGRS revision): Absolute relative WSE error by quality flag.

Boxplot keyed to the bitwise quality flags (replaces the log-ISA heatmap;
Reviewer 1, comments 2-4). Top box pools all observations as the dataset
base rate; each box below shows the error distribution of the observations
a single flag raises. Right margin reports the activation count (inside the
axes) and the high-error share (share of activations exceeding T) outside.
No in-plot title (IEEE captions carry the description).

Input : df_matched with columns wse_relative_m, wse_usgs_relative_m, reach_q_b
Output: figures/fig05_quality_flags_boxplot.(png|pdf)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BITS = {
    'class_suspect': 2, 'geo_suspect': 4, 'water_frac': 8, 'bright_land': 128,
    'few_area': 1024, 'few_wse': 2048, 'far_range': 8192, 'near_range': 16384,
    'partial_obs': 32768, 'class_degraded': 262144, 'geo_degraded': 524288,
    'lake_flag': 4194304, 'min_fit_pts': 33554432, 'no_area': 67108864,
    'no_wse': 134217728, 'no_pixels': 268435456,
}


def create_figure5_quality_flags_boxplot(
    df_matched, threshold=0.1832, figsize=(12, 9), save_path=None, dpi=300,
    xmax=3.5, flag_color='#4C72B0', base_color='#9e9e9e',
    fs_label=14, fs_tick=12, fs_annot=16,
    n_x=0.78, share_x=1.04,
):
    df = df_matched.copy()
    df['error_relative_m'] = (df['wse_relative_m'] - df['wse_usgs_relative_m']).abs()
    err_all = df['error_relative_m'].values
    base_rate = (err_all >= threshold).mean() * 100

    recs = []
    for disp, bit in BITS.items():
        m = (df['reach_q_b'].values & bit) > 0
        n = int(m.sum())
        err = err_all[m] if n > 0 else np.array([])
        hi = (err >= threshold).mean() * 100 if n > 0 else np.nan
        recs.append({'disp': disp, 'n': n, 'hi': hi, 'err': err})
    recs.sort(key=lambda r: r['n'], reverse=True)

    rows = [{'disp': 'All observations', 'n': len(df), 'hi': base_rate,
             'err': err_all, 'base': True}] + [{**r, 'base': False} for r in recs]
    ypos = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=figsize)

    bd, bp_pos, bcol = [], [], []
    for y, r in zip(ypos, rows):
        if r['err'].size > 0:
            bd.append(r['err']); bp_pos.append(y)
            bcol.append(base_color if r['base'] else flag_color)
    bp = ax.boxplot(bd, positions=bp_pos, vert=False, widths=0.62, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', linewidth=2.2),
                    whiskerprops=dict(color='black', linewidth=1.2),
                    capprops=dict(color='black', linewidth=1.2),
                    boxprops=dict(edgecolor='black', linewidth=1.2))
    for patch, col in zip(bp['boxes'], bcol):
        patch.set_facecolor(col); patch.set_alpha(0.9)

    ax.set_xlim(0, xmax)
    ax.axvline(threshold, color='#ca0020', linestyle='--', linewidth=2.2, zorder=5)
    ax.text(threshold, -0.45, f'  Target T = {threshold:.2f} m',
            color='#ca0020', fontsize=fs_label, fontweight='bold', va='top', ha='left')
    ax.axhline(ypos[0] - 0.5, color='gray', linestyle=':', linewidth=1.2)

    # n (inside, left-aligned column) / high-error share (outside margin)
    for y, r in zip(ypos, rows):
        fw = 'bold' if r['base'] else 'normal'
        if r['n'] > 0:
            ax.text(n_x, y, f"n = {r['n']:,}", transform=ax.get_yaxis_transform(),
                    va='center', ha='left', fontsize=fs_annot, fontweight=fw,
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                              alpha=0.75, edgecolor='none'))
        if not np.isnan(r['hi']):
            ax.text(share_x, y, f"≥T: {r['hi']:.0f}%",
                    transform=ax.get_yaxis_transform(),
                    va='center', ha='left', fontsize=fs_annot, fontweight=fw)

    ax.set_yticks(ypos)
    ax.set_yticklabels([r['disp'] for r in rows], fontsize=fs_label, fontweight='bold')
    ax.set_xlabel('|Relative WSE Error| (m)', fontsize=fs_label, fontweight='bold')
    ax.tick_params(axis='x', labelsize=fs_tick)
    for sp in ax.spines.values():
        sp.set_linewidth(1.6); sp.set_color('black')
    ax.grid(axis='x', linestyle=':', alpha=0.4)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        fig.savefig(save_path.replace('.png', '.pdf'), dpi=dpi,
                    bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    return fig


if __name__ == '__main__':
    df_matched = pd.read_parquet(
        '/work/pi_kandread_umass_edu/swot-urban/2_urban_reaches/all_reaches_vD.parquet')
    create_figure5_quality_flags_boxplot(
        df_matched, save_path='figures/fig05_quality_flags_boxplot.png')
