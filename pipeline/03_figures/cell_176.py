"""
CDF Plot: Parallel Zone Analysis with ISA Bins
===============================================
Using log(x + offset) transformation
- Plot 1: Original Parallel zone CDF (by sample_geometry_category)
- Plot 2: Parallel zone - Near ISA bins (10% intervals)
- Plot 3: Parallel zone - Far ISA bins (10% intervals)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import matplotlib.cm as cm

plt.style.use('dark_background')

OUTPUT_DIR = Path("unit_level_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_flow_angle_zone(flow_angle):
    """
    Classify flow angle into zones
    """
    fa = flow_angle % 360
    if (fa >= 350 or fa <= 10) or (170 <= fa <= 190):
        return 'Parallel'
    elif (80 <= fa <= 100) or (260 <= fa <= 280):
        return 'Perpendicular'
    else:
        return 'Other'


def get_isa_bin(isa_value):
    """
    Classify ISA into 10% bins
    Assumes isa_value is percentage (0~100)
    """
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
    elif isa_value <= 90:
        return '80-90'
    else:
        return '90-100'


def plot_cdf_original(ax, data, y_col, threshold=0.1832, offset=1):
    """
    Original CDF plot by sample_geometry_category
    Using log(x + offset) transformation
    """
    ax.set_facecolor('black')
    
    colors = {
        'NA': 'white',
        'Near': '#3399ff',
        'Far': '#ff6666',
        'Near/Far': '#ffdd00'
    }
    
    geometry_cats = ['NA', 'Near', 'Far', 'Near/Far']
    pct_data = []
    p68_data = []
    
    all_errors = data[y_col].dropna()
    if len(all_errors) > 0:
        x_min_orig = all_errors.min()
        x_max_orig = all_errors.max()
    else:
        x_min_orig, x_max_orig = 0, 100
    
    for cat in geometry_cats:
        cat_data = data[data['sample_geometry_category'] == cat][y_col].dropna()
        if len(cat_data) > 0:
            sorted_data = np.sort(cat_data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            # +offset 변환
            ax.plot(sorted_data + offset, cdf, color=colors[cat], 
                   label=f'{cat} (n={len(cat_data)})', linewidth=2)
            
            p68 = np.percentile(cat_data, 68)
            p68_data.append((cat, p68, colors[cat]))
            
            pct_below = 100 * (cat_data < threshold).sum() / len(cat_data)
            pct_above = 100 * (cat_data >= threshold).sum() / len(cat_data)
            pct_data.append((cat, pct_below, pct_above, colors[cat]))
    
    # Reference lines (+offset)
    ax.axvline(x=threshold + offset, color='orange', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.axvline(x=0 + offset, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.68, color='lime', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(0.02, 0.70, '68%tile', transform=ax.transAxes, color='lime', fontsize=8, fontweight='bold')
    
    # 68th percentile markers
    y_offsets = [0.01, -0.04, 0.01, -0.04]
    va_aligns = ['bottom', 'top', 'bottom', 'top']
    
    for i, (cat, p68, color) in enumerate(p68_data):
        ax.axvline(x=p68 + offset, color=color, linestyle=':', linewidth=1, alpha=0.6)
        ax.scatter([p68 + offset], [0.68], color=color, s=30, zorder=5)
        
        y_pos = y_offsets[i % 4]
        va = va_aligns[i % 4]
        
        # 라벨은 원래 값
        ax.text(p68 + offset, y_pos, f'{p68:.2f}', transform=ax.get_xaxis_transform(),
                color=color, fontsize=9, fontweight='bold', ha='center', va=va)
    
    # Threshold percentage text
    if pct_data:
        ax.text(0.98, 0.62, f'{"T=0.1832m":8s} {"<T":>6s} {">T":>6s}', 
                transform=ax.transAxes, fontsize=8, color='lime', va='top', ha='right', 
                family='monospace')
        y_pos = 0.54
        for cat, below, above, color in pct_data:
            text = f'{cat:8s} {below:5.0f}% {above:5.0f}%'
            ax.text(0.98, y_pos, text, transform=ax.transAxes, 
                    fontsize=8, color=color, va='top', ha='right', family='monospace')
            y_pos -= 0.08
    
    # Log scale (symlog 대신)
    ax.set_xscale('log')
    ax.set_xlim((x_min_orig + offset) * 0.9, (x_max_orig + offset) * 1.1)
    ax.set_ylim(0, 1)
    
    # X축 라벨 (offset 빼서 원래 값으로 표시)
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
    
    tick_values_orig = [0, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    tick_values = [v + offset for v in tick_values_orig 
                   if (x_min_orig + offset) * 0.9 <= v + offset <= (x_max_orig + offset) * 1.1]
    ax.set_xticks(tick_values)
    
    ax.set_xlabel('Error (m)', color='white', fontsize=10)
    ax.set_ylabel('CDF', color='white', fontsize=10)
    ax.set_title(f'Parallel (350-10°, 170-190°)\n(n={len(data):,})', 
                 color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=9)
    ax.legend(loc='lower right', fontsize=8, facecolor='black', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.2, color='white')
    for spine in ax.spines.values():
        spine.set_color('white')


def plot_cdf_by_isa_bin(ax, data, y_col, isa_col, title, threshold=0.1832, offset=1):
    """
    CDF plot by ISA bins (10% intervals)
    Using log(x + offset) transformation
    """
    ax.set_facecolor('black')
    
    bin_order = ['0-5', '5-10', '10-20', '20-30', '30-40', '40-50', 
                 '50-60', '60-70', '70-80', '80-90', '90-100']
    
    cmap = cm.get_cmap('cool', 11)
    bin_colors = {bin_order[i]: cmap(i) for i in range(11)}
    
    data = data.copy()
    data['isa_bin'] = data[isa_col].apply(get_isa_bin)
    
    all_errors = data[y_col].dropna()
    if len(all_errors) > 0:
        x_min_orig = all_errors.min()
        x_max_orig = all_errors.max()
    else:
        x_min_orig, x_max_orig = 0, 100
    
    # Plot CDF for each bin
    plotted_bins = []
    for bin_label in bin_order:
        bin_data = data[data['isa_bin'] == bin_label][y_col].dropna()
        if len(bin_data) > 0:
            sorted_data = np.sort(bin_data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            color = bin_colors[bin_label]
            
            # +offset 변환
            ax.plot(sorted_data + offset, cdf, color=color, linewidth=1.5)
            
            p68 = np.percentile(bin_data, 68)
            plotted_bins.append((bin_label, p68, color, len(bin_data)))
    
    # 68th percentile line
    ax.axhline(y=0.68, color='lime', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(0.02, 0.70, '68%tile', transform=ax.transAxes, color='lime', fontsize=8, fontweight='bold')
    
    # Reference lines (+offset)
    ax.axvline(x=threshold + offset, color='orange', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.axvline(x=0 + offset, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Add text labels - alternate 4 positions
    y_offsets_text = [0.03, 0.08, -0.03, -0.08]
    va_aligns = ['bottom', 'bottom', 'top', 'top']
    
    for i, (bin_label, p68, color, n) in enumerate(plotted_bins):
        ax.scatter([p68 + offset], [0.68], color=color, s=30, zorder=5)
        ax.axvline(x=p68 + offset, color=color, linestyle=':', linewidth=0.8, alpha=0.4)
        
        y_offset = y_offsets_text[i % 4]
        va = va_aligns[i % 4]
        
        ax.text(p68 + offset, 0.68 + y_offset, f'{bin_label}%', color=color, fontsize=8, fontweight='bold',
                ha='center', va=va)
    
    # Add 68th percentile summary text
    if plotted_bins:
        ax.text(0.98, 0.48, '68%tile Error (m)', transform=ax.transAxes,
                fontsize=8, color='lime', va='top', ha='right', family='monospace', fontweight='bold')
        
        y_pos = 0.40
        for bin_label, p68, color, n in plotted_bins:
            text = f'{bin_label:>5}%: {p68:.2f}m (n={n})'
            ax.text(0.98, y_pos, text, transform=ax.transAxes,
                    fontsize=7, color=color, va='top', ha='right', family='monospace')
            y_pos -= 0.04
    
    # Log scale (symlog 대신)
    ax.set_xscale('log')
    ax.set_xlim((x_min_orig + offset) * 0.9, (x_max_orig + offset) * 1.1)
    ax.set_ylim(0, 1)
    
    # X축 라벨 (offset 빼서 원래 값으로 표시)
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
    
    tick_values_orig = [0, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    tick_values = [v + offset for v in tick_values_orig 
                   if (x_min_orig + offset) * 0.9 <= v + offset <= (x_max_orig + offset) * 1.1]
    ax.set_xticks(tick_values)
    
    ax.set_xlabel('Error (m)', color='white', fontsize=10)
    ax.set_ylabel('CDF', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=9)
    ax.grid(True, alpha=0.2, color='white')
    for spine in ax.spines.values():
        spine.set_color('white')


def plot_parallel_isa_analysis(df, y_col, threshold=0.1832, figsize=(18, 5), 
                                save=True, show=False, offset=1):
    """
    Create 1x3 plot for Parallel zone ISA analysis
    """
    
    df_plot = df.dropna(subset=[y_col, 'flow_angle', 'sample_geometry_category', 'near_isa', 'far_isa']).copy()
    df_plot['flow_zone'] = df_plot['flow_angle'].apply(get_flow_angle_zone)
    df_parallel = df_plot[df_plot['flow_zone'] == 'Parallel']
    
    if len(df_parallel) == 0:
        print("Warning: No data in Parallel zone")
        return None
    
    print(f"Parallel zone: {len(df_parallel):,} samples")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor='black')
    
    plot_cdf_original(axes[0], df_parallel, y_col, threshold, offset)
    plot_cdf_by_isa_bin(axes[1], df_parallel, y_col, 'near_isa', 'By Near ISA (%)', threshold, offset)
    plot_cdf_by_isa_bin(axes[2], df_parallel, y_col, 'far_isa', 'By Far ISA (%)', threshold, offset)
    
    fig.suptitle(f'Parallel Zone Analysis: {y_col}', 
                 fontsize=14, fontweight='bold', color='white', y=1.02)
    
    plt.tight_layout()
    
    if save:
        filename = f"parallel_isa_analysis_{y_col}_log.png"
        filepath = OUTPUT_DIR / filename
        fig.savefig(filepath, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig