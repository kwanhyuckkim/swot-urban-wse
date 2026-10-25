"""
Combined Grouped Boxplot: 2 Rows
================================
Row 1: X = Building Height, Color = Far ISA (Reds)
Row 2: X = Far ISA, Color = Building Height (Greens)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# White background style
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
    """ISA bins: 0-5, 5-10, 10-20, ..., 60-70, 70-100 (70-80, 80-100 합침)"""
    if isa_value <= 5:
        return '0-5'
    elif isa_value <= 10:
        return '5-10'
    elif isa_value <= 20:
        return '10-20'
    elif isa_value <= 30:
        return '20-30'
    elif isa_value <= 45:
        return '30-45'
    elif isa_value <= 60:
        return '45-60'        
    else:
        return '60+' 

def get_building_bins_dynamic(data, building_col, min_samples=100, start_from=5, interval=2):
    """
    Create building height bins with custom intervals.
    Stop when REMAINING samples (from edge to end) < min_samples.
    
    Parameters
    ----------
    interval : int
        Bin width in meters (1 = 1m intervals, 2 = 2m intervals)
    """
    values = data[building_col].dropna()
    values = values[values >= start_from]
    
    max_val = int(np.ceil(values.max()))
    
    print(f"[Building] Range (>={start_from}m): {values.min():.2f} ~ {values.max():.2f}m")
    print(f"[Building] Total samples: {len(values):,}")
    print(f"[Building] Interval: {interval}m")
    
    valid_edges = [start_from]
    
    for edge in range(start_from + interval, max_val + interval, interval):  # interval씩 증가
        remaining = (values >= edge).sum()
        
        print(f"  {edge}m 이상 남은 샘플: {remaining:,}")
        
        if remaining >= min_samples:
            valid_edges.append(edge)
        else:
            print(f"  -> Stopping: {edge}m 이상 남은 샘플 {remaining} < {min_samples}")
            break
    
    valid_edges.append(np.inf)
    
    # Create labels
    labels = []
    for i in range(len(valid_edges) - 1):
        if valid_edges[i + 1] == np.inf:
            labels.append(f'{valid_edges[i]}+')
        else:
            labels.append(f'{valid_edges[i]}-{valid_edges[i + 1]}')
    
    print(f"[Building] Final bins: {labels}")
    
    return valid_edges, labels


def plot_grouped_boxplot_combined(df,
                                   isa_col='far_isa',
                                   building_col='far_building',
                                   error_col='abs_error_node',
                                   min_samples=50,
                                   interval=2,
                                   offset=0.1,
                                   figsize=(18, 12),
                                   save=True,
                                   show=True):
    """
    Combined grouped boxplot with 2 rows.
    Row 1: X = Building Height, Color = ISA (Reds)
    Row 2: X = ISA, Color = Building Height (Greens)
    """
    
    print("COMBINED GROUPED BOXPLOT")
    
    # Filter to Parallel zone only
    df_parallel = df[df['flow_angle'].apply(get_flow_angle_zone) == 'Parallel'].copy()
    print(f"\n[Data] Parallel zone: {len(df_parallel):,} / {len(df):,}")
    
    # Clean data
    data = df_parallel[[isa_col, building_col, error_col]].dropna()
    print(f"[Data] After dropna: {len(data):,}")
    
    # Define bins
    
    isa_bin_order = ['0-5', '5-10', '10-20', '20-30', '30-45', '45-60', 
                     '60+']
    data['isa_bin'] = data[isa_col].apply(get_isa_bin)
    
    building_edges, building_labels = get_building_bins_dynamic(data, building_col, min_samples, interval=interval)
    data['building_bin'] = pd.cut(data[building_col], bins=building_edges,
                                   labels=building_labels, right=False)
    
    data = data.dropna(subset=['isa_bin', 'building_bin'])
    print(f"[Data] After binning: {len(data):,}")
    
    n_isa_bins = len(isa_bin_order)
    n_building_bins = len(building_labels)
    
    # Sample counts per bin (for x-axis labels)
    building_counts = data.groupby('building_bin', observed=True).size().to_dict()
    isa_counts = data.groupby('isa_bin', observed=True).size().to_dict()
    
    building_labels_with_n = [f'{lbl}\n(n={building_counts.get(lbl, 0):,})' for lbl in building_labels]
    
    existing_isa_bins = [b for b in isa_bin_order if b in data['isa_bin'].values]
    isa_labels_with_n = [f'{lbl}\n(n={isa_counts.get(lbl, 0):,})' for lbl in existing_isa_bins]
    
    # Colors
    import matplotlib.cm as cm
    
    cmap_isa = cm.get_cmap('Oranges', n_isa_bins + 2)
    colors_isa = [cmap_isa(i + 2) for i in range(n_isa_bins)]
    
    cmap_building = cm.get_cmap('Purples', n_building_bins + 2)
    colors_building = [cmap_building(i + 2) for i in range(n_building_bins)]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize, facecolor='white')
    
    # Row 1: X = Building Height, Color = ISA (Reds)
    ax1 = axes[0]
    ax1.set_facecolor('white')
    
    box_data1 = []
    positions1 = []
    box_colors1 = []
    sample_counts1 = []  # 각 boxplot별 sample 수
    
    group_width = 0.8
    box_width1 = group_width / (n_isa_bins + 1)
    
    for i, building_label in enumerate(building_labels):
        group_center = i + 1
        
        for j, isa_label in enumerate(isa_bin_order):
            mask = (data['building_bin'] == building_label) & (data['isa_bin'] == isa_label)
            subset = data[mask][error_col].values
            
            if len(subset) > 0:
                box_data1.append(subset + offset)  # offset 적용
                sample_counts1.append(len(subset))
            else:
                box_data1.append([np.nan])
                sample_counts1.append(0)
            
            pos_offset = (j - (n_isa_bins - 1) / 2) * box_width1
            positions1.append(group_center + pos_offset)
            box_colors1.append(colors_isa[j])
    
    bp1 = ax1.boxplot(box_data1,
                      positions=positions1,
                      widths=box_width1 * 0.85,
                      patch_artist=True,
                      showfliers=True,
                      flierprops=dict(marker='o', markersize=2, alpha=0.3,
                                     markerfacecolor='gray', markeredgecolor='none'))
    
    for patch, color in zip(bp1['boxes'], box_colors1):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
        patch.set_alpha(0.85)
    
    for whisker in bp1['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(0.8)
    for cap in bp1['caps']:
        cap.set_color('black')
        cap.set_linewidth(0.8)
    for median in bp1['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    
    # Whisker 위에 sample 수 표시 (90도 회전)
    for i, (pos, n) in enumerate(zip(positions1, sample_counts1)):
        if n > 0:
            # Upper whisker의 y값 가져오기 (whiskers는 [lower, upper] 순서로 2개씩)
            upper_whisker = bp1['whiskers'][i * 2 + 1]
            y_top = upper_whisker.get_ydata().max()
            
            ax1.text(pos, y_top * 1.05, f' n={n}', rotation=90, fontsize=11,
                    ha='center', va='bottom', color='black')
    
    # Log scale with offset
    ax1.set_yscale('log')
    ax1.set_ylim(offset * 0.8, 50 + offset)
    
    # Y축 tick labels (원래 값으로 표시)
    tick_values_orig = [0, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    tick_values = [v + offset for v in tick_values_orig]
    ax1.set_yticks(tick_values)
    ax1.set_yticklabels([str(v) for v in tick_values_orig])
    
    ax1.set_ylabel(r'$|Relative\ WSE\ Error|$ (m)', fontsize=13)
    ax1.set_xlabel('Far Building Height (m)', fontsize=13)
    ax1.set_xticks(range(1, n_building_bins + 1))
    ax1.set_xticklabels(building_labels_with_n, fontsize=10)
    ax1.set_xlim(0.3, n_building_bins + 0.7)
    ax1.tick_params(labelsize=11)
    ax1.yaxis.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.xaxis.grid(False)
    ax1.set_title(f'Bin: Far Building Height & Color: Far ISA (n={len(data):,})', 
                  fontsize=14, fontweight='bold')
    
    for spine in ax1.spines.values():
        spine.set_color('black')
    
    legend_patches1 = [Patch(facecolor=colors_isa[i], edgecolor='black',
                             label=isa_bin_order[i] + '%', alpha=0.85)
                       for i in range(n_isa_bins)]
    ax1.legend(handles=legend_patches1,
               title='Far ISA',
               loc='center left',
               bbox_to_anchor=(1.02, 0.5),
               framealpha=0.95,
               title_fontsize=12,
               fontsize=10)
    
    # Row 2: X = ISA, Color = Building Height (Greens)
    ax2 = axes[1]
    ax2.set_facecolor('white')
    
    box_data2 = []
    positions2 = []
    box_colors2 = []
    sample_counts2 = []
    
    box_width2 = group_width / (n_building_bins + 1)
    
    n_existing_isa = len(existing_isa_bins)
    
    for i, isa_label in enumerate(existing_isa_bins):
        group_center = i + 1
        
        for j, building_label in enumerate(building_labels):
            mask = (data['isa_bin'] == isa_label) & (data['building_bin'] == building_label)
            subset = data[mask][error_col].values
            
            if len(subset) > 0:
                box_data2.append(subset + offset)  # offset 적용
                sample_counts2.append(len(subset))
            else:
                box_data2.append([np.nan])
                sample_counts2.append(0)
            
            pos_offset = (j - (n_building_bins - 1) / 2) * box_width2
            positions2.append(group_center + pos_offset)
            box_colors2.append(colors_building[j])
    
    bp2 = ax2.boxplot(box_data2,
                      positions=positions2,
                      widths=box_width2 * 0.85,
                      patch_artist=True,
                      showfliers=True,
                      flierprops=dict(marker='o', markersize=2, alpha=0.3,
                                     markerfacecolor='gray', markeredgecolor='none'))
    
    for patch, color in zip(bp2['boxes'], box_colors2):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
        patch.set_alpha(0.85)
    
    for whisker in bp2['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(0.8)
    for cap in bp2['caps']:
        cap.set_color('black')
        cap.set_linewidth(0.8)
    for median in bp2['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    
    # Whisker 위에 sample 수 표시 (90도 회전)
    for i, (pos, n) in enumerate(zip(positions2, sample_counts2)):
        if n > 0:
            upper_whisker = bp2['whiskers'][i * 2 + 1]
            y_top = upper_whisker.get_ydata().max()
            
            ax2.text(pos, y_top * 1.05, f' n={n}', rotation=90, fontsize=11,
                    ha='center', va='bottom', color='black')
    
    # Log scale with offset
    ax2.set_yscale('log')
    ax2.set_ylim(offset * 0.8, 50 + offset)
    
    ax2.set_yticks(tick_values)
    ax2.set_yticklabels([str(v) for v in tick_values_orig])
    
    ax2.set_ylabel(r'$|Relative\ WSE\ Error|$ (m)', fontsize=13)
    ax2.set_xlabel('Far ISA (%)', fontsize=13)
    ax2.set_xticks(range(1, n_existing_isa + 1))
    ax2.set_xticklabels(isa_labels_with_n, fontsize=10)
    ax2.set_xlim(0.3, n_existing_isa + 0.7)
    ax2.tick_params(labelsize=11)
    ax2.yaxis.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.xaxis.grid(False)
    ax2.set_title(f'Bin: Far ISA & Color: Far Building Height (n={len(data):,})', 
                  fontsize=14, fontweight='bold')
    
    for spine in ax2.spines.values():
        spine.set_color('black')
    
    legend_patches2 = [Patch(facecolor=colors_building[i], edgecolor='black',
                             label=building_labels[i] + 'm', alpha=0.85)
                       for i in range(n_building_bins)]
    ax2.legend(handles=legend_patches2,
               title='Far Building Height',
               loc='center left',
               bbox_to_anchor=(1.02, 0.5),
               framealpha=0.95,
               title_fontsize=12,
               fontsize=10)
    
    # Main title and layout
    fig.suptitle('Parallel Angle X Far Zone Analysis - ISA & Building Height', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    
    # Save
    if save:
        filepath = OUTPUT_DIR / "figure7_grouped_boxplot_combined.png"
        fig.savefig(filepath, dpi=300, facecolor='white', bbox_inches='tight')
        print(f"\nSaved: {filepath}")
        
        filepath = OUTPUT_DIR / "figure7_grouped_boxplot_combined.pdf"
        fig.savefig(filepath, dpi=300, facecolor='white', bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig