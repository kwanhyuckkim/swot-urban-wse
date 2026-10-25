"""
Parallel Zone Analysis: 2x2 Plot (Far & Near ISA) - Clean Version
=================================================================
- Legend에 68%tile과 <T 퍼센티지 표시
- x축, y축 직접 표기 제거
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from pathlib import Path

# 흰색 바탕
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_flow_angle_zone(flow_angle):
    fa = flow_angle % 360
    if (fa >= 350 or fa <= 10) or (170 <= fa <= 190):
        return 'Parallel'
    elif (80 <= fa <= 100) or (260 <= fa <= 280):
        return 'Perpendicular'
    else:
        return 'Other'


def get_isa_bin(isa_value):
    """80-100으로 합침"""
    if isa_value <= 5:
        return '0-5'
    elif isa_value <= 10:
        return '5-10'
    elif isa_value <= 20:
        return '10-20'
    elif isa_value <= 30:
        return '20-30'
    elif isa_value <= 40:
        return '30-40'
    elif isa_value <= 50:
        return '40-50'
    elif isa_value <= 60:
        return '50-60'
    elif isa_value <= 70:
        return '60-70'
    elif isa_value <= 80:
        return '70-80'
    else:
        return '80-100'


def plot_cdf_by_isa_bin(ax, data, y_col, isa_col, title, threshold=0.1832, offset=1, xlim_max=20):
    """CDF plot by ISA bins - Clean legend version"""
    ax.set_facecolor('white')
    
    bin_order = ['0-5', '5-10', '10-20', '20-30', '30-40', '40-50', 
                 '50-60', '60-70', '70-80', '80-100']
    
    # Far ISA -> Reds, Near ISA -> Blues
    if isa_col == 'far_isa':
        cmap = cm.get_cmap('Oranges', 12)
        bin_colors = {bin_order[i]: cmap(i + 2) for i in range(10)}
    else:
        cmap = cm.get_cmap('Greens', 12)
        bin_colors = {bin_order[i]: cmap(i + 2) for i in range(10)}
    
    data = data.copy()
    data['isa_bin'] = data[isa_col].apply(get_isa_bin)
    
    plotted_bins = []
    
    for bin_label in bin_order:
        bin_data = data[data['isa_bin'] == bin_label][y_col].dropna()
        if len(bin_data) > 0:
            sorted_data = np.sort(bin_data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            color = bin_colors[bin_label]
            
            ax.plot(sorted_data + offset, cdf, color=color, linewidth=2)
            
            p68 = np.percentile(bin_data, 68)
            cdf_at_T = (bin_data <= threshold).sum() / len(bin_data)
            
            plotted_bins.append((bin_label, p68, cdf_at_T, color, len(bin_data)))
            
            # 68th percentile marker
            ax.scatter([p68 + offset], [0.68], color=color, s=40, zorder=5)
    
    # 68% horizontal line
    ax.axhline(y=0.68, color='black', linestyle='-', linewidth=2, alpha=0.8)
    ax.text(0.02, 0.70, '68%', transform=ax.transAxes, color='black', fontsize=12, fontweight='bold')
    
    # Threshold line
    ax.axvline(x=threshold + offset, color='black', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(x=0 + offset, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # T label at top
    ax.text(threshold + offset, 1.02, f'T={threshold}m', transform=ax.get_xaxis_transform(),
            color='black', fontsize=11, fontweight='bold', ha='center', va='bottom')
    
    # Legend table with proper alignment
    if plotted_bins:
        # Header
        header = f"{'ISA':>9}  {'68%tile':>9}  {'<T':>6}  {'n':>7}"
        ax.text(0.98, 0.54, header, transform=ax.transAxes,
                fontsize=11, color='black', va='top', ha='right', family='monospace', fontweight='bold')
        
        # Separator line
        ax.text(0.98, 0.50, "─" * 45, transform=ax.transAxes,
                fontsize=8, color='gray', va='top', ha='right', family='monospace')
        
        y_pos = 0.47
        for bin_label, p68, cdf_at_T, color, n in plotted_bins:
            pct_below_T = cdf_at_T * 100
            row = f"{bin_label + '%':>8}  {p68:>7.2f}m  {pct_below_T:>5.0f}%  {n:>7,}"
            ax.text(0.98, y_pos, row, transform=ax.transAxes,
                    fontsize=11, color=color, va='top', ha='right', family='monospace')
            y_pos -= 0.042
    
    ax.set_xscale('log')
    ax.set_xlim(offset * 0.9, xlim_max + offset)
    ax.set_ylim(0, 1)
    
    def format_func(value, tick_number):
        original = value - offset
        if abs(original) < 0.01:
            return '0'
        elif abs(original) >= 10:
            return f'{original:.0f}'
        elif abs(original) >= 1:
            return f'{original:.1f}'
        else:
            return f'{original:.2f}'
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    
    tick_values_orig = [0, 0.2, 0.5, 1, 2, 5, 10, 20]
    tick_values = [v + offset for v in tick_values_orig if v <= xlim_max]
    ax.set_xticks(tick_values)
    
    ax.set_xlabel(r'$|Relative\ WSE\ Error|$ (m)', color='black', fontsize=13)
    ax.set_ylabel('CDF', color='black', fontsize=13)
    ax.set_title(title, color='black', fontsize=14, fontweight='bold')
    ax.tick_params(colors='black', labelsize=11)
    ax.grid(True, alpha=0.3, color='gray')
    
    for spine in ax.spines.values():
        spine.set_color('black')


def plot_boxplot_by_isa_bin(ax, data, y_col, isa_col, title, xlabel_label):
    """Boxplot by ISA bins"""
    ax.set_facecolor('white')
    
    bin_order = ['0-5', '5-10', '10-20', '20-30', '30-40', '40-50', 
                 '50-60', '60-70', '70-80', '80-100']
    
    # Far ISA -> Reds, Near ISA -> Blues
    if isa_col == 'far_isa':
        cmap = cm.get_cmap('Oranges', 12)
        bin_colors = {bin_order[i]: cmap(i + 2) for i in range(10)}
    else:
        cmap = cm.get_cmap('Greens', 12)
        bin_colors = {bin_order[i]: cmap(i + 2) for i in range(10)}
    
    data = data.copy()
    data['isa_bin'] = data[isa_col].apply(get_isa_bin)
    
    existing_bins = [b for b in bin_order if b in data['isa_bin'].values]
    
    box_data = []
    positions = []
    colors = []
    bin_labels = []
    
    for i, bin_label in enumerate(existing_bins):
        bin_df = data[data['isa_bin'] == bin_label]
        if len(bin_df) > 0:
            error_vals = bin_df[y_col].dropna().values
            if len(error_vals) > 0:
                box_data.append(error_vals)
                positions.append(len(positions))
                colors.append(bin_colors[bin_label])
                bin_labels.append(bin_label)
    
    if len(positions) == 0:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                ha='center', va='center', color='gray', fontsize=14)
        return
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    whis=1.5,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    
    for flier, color in zip(bp['fliers'], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
    
    ax.set_yscale('symlog', linthresh=1)
    
    error_ticks = [0.1, 0.5, 1, 2, 5, 10, 20]
    ax.set_yticks(error_ticks)
    ax.set_yticklabels([str(t) for t in error_ticks], fontsize=11, color='black')
    ax.set_ylim(0.05, 30)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', color='black', fontsize=11)
    ax.set_xlabel(xlabel_label, color='black', fontsize=13)
    ax.set_ylabel(r'$|Relative\ WSE\ Error|$ (m)', color='black', fontsize=13)
    ax.set_title(title, color='black', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_xlim(-0.5, len(bin_labels) - 0.5)
    
    for spine in ax.spines.values():
        spine.set_color('black')
    
    ax.tick_params(colors='black', labelsize=11)


def plot_parallel_zone_2x2(df, y_col='abs_error_node', 
                            isa_ratio=0,
                            threshold=0.1832, offset=1,
                            xlim_max=20,
                            figsize=(14, 12),
                            save=True, show=True):
    """
    2x2 Parallel Zone Analysis (Far & Near ISA)
    """
    
    print("=" * 70)
    print("Parallel Angle Analysis: 2x2 Plot (Far & Near ISA)")
    print("=" * 70)
    
    # Prepare data
    required_cols = [y_col, 'flow_angle', 'far_isa', 'far_ratio', 'near_isa', 'near_ratio']
    df_plot = df.dropna(subset=required_cols).copy()
    df_plot['flow_zone'] = df_plot['flow_angle'].apply(get_flow_angle_zone)
    df_parallel = df_plot[df_plot['flow_zone'] == 'Parallel'].copy()
    
    print(f"\n[Data] Total Parallel zone: {len(df_parallel):,}")
    
    # Filter by ratio
    df_far = df_parallel[df_parallel['far_ratio'] >= isa_ratio].copy()
    df_near = df_parallel[df_parallel['near_ratio'] >= isa_ratio].copy()
    
    print(f"[Data] Far ISA: {len(df_far):,}")
    print(f"[Data] Near ISA: {len(df_near):,}")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor='white')
    
    # Row 1: Far ISA
    plot_cdf_by_isa_bin(axes[0, 0], df_far, y_col, 'far_isa', 
                        f'CDF by Far ISA (n={len(df_far):,})',
                        threshold, offset, xlim_max)
    
    plot_boxplot_by_isa_bin(axes[0, 1], df_far, y_col, 'far_isa',
                            f'Boxplot by Far ISA', 'Far ISA (%)')
    
    # Row 2: Near ISA
    plot_cdf_by_isa_bin(axes[1, 0], df_near, y_col, 'near_isa', 
                        f'CDF by Near ISA (n={len(df_near):,})',
                        threshold, offset, xlim_max)
    
    plot_boxplot_by_isa_bin(axes[1, 1], df_near, y_col, 'near_isa',
                            f'Boxplot by Near ISA', 'Near ISA (%)')
    
    fig.suptitle('Parallel Angle Analysis - Far & Near ISA', 
                 fontsize=18, fontweight='bold', color='black', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save:
        filepath = OUTPUT_DIR / f"figure6_parallel_zone_2x2.png"
        fig.savefig(filepath, dpi=300, facecolor='white', bbox_inches='tight')
        filepath = OUTPUT_DIR / f"figure6_parallel_zone_2x2.pdf"
        fig.savefig(filepath, dpi=300, facecolor='white', bbox_inches='tight')

        print(f"\n✓ Saved: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig