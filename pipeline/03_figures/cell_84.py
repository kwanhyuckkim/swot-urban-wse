import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FIGURE 2: Boxplot - ISA Bins vs Error (Revised)
# =============================================================================
def create_figure2_isa_vs_error(
    df_matched,
    threshold=0.1832,
    figsize=(14, 9),
    save_path=None,
    dpi=300,
    use_log_scale=True,  # True: log scale, False: linear scale
    ylim_max=5           # Y축 최대값
):
    """
    Figure 2: Boxplot showing error distribution by ISA bins.
     
    ISA bins: 0-5, 5-10, ..., 45-50, 50+ (5% intervals)
    50% 이상은 50+ bin에 포함
     
    Parameters:
    -----------
    use_log_scale : bool
        True: log scale with +1 offset
        False: linear scale (no offset)
    ylim_max : float
        Y축 최대값 (default: 5)
    """
    
    df = df_matched.copy()
    
    # |Relative WSE Error| 계산
    df['error_relative_m'] = (df['wse_relative_m'] - df['wse_usgs_relative_m']).abs()
    
    df['isa_pct'] = df['fctimp_mean'] * 100
    
    max_isa = df['isa_pct'].max()
    
    isa_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    isa_labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', 
                  '30-35', '35-40', '40-45', '45-50']
    
    df['isa_bin'] = pd.cut(df['isa_pct'], bins=isa_bins, labels=isa_labels, 
                           include_lowest=True, right=True)
    
    df['isa_bin'] = df['isa_bin'].astype(str)
    df.loc[df['isa_pct'] >= 50, 'isa_bin'] = '50+'
    
    isa_labels_final = isa_labels + ['50+']
    
    df = df.dropna(subset=['isa_bin', 'error_relative_m'])
    df = df[df['isa_bin'] != 'nan']
    
    n_total_50plus = (df['isa_pct'] >= 50).sum()
    
    scale_type = "log" if use_log_scale else "linear"
    print(f"Figure 2: ISA bins 0-50% (5% intervals), 50+ (max: {max_isa:.1f}%)")
    print(f"  Scale: {scale_type}, Y-axis max: {ylim_max}")
    print(f"  Total '50+' bin: {n_total_50plus} samples")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Offset only for log scale
    offset = 1 if use_log_scale else 0
    
    box_data = []
    box_positions = []
    sample_sizes = []
    p68_values = [] # 각 bin의 68%tile 값을 저장할 리스트
    
    for i, bin_label in enumerate(isa_labels_final):
        subset = df[df['isa_bin'] == bin_label]['error_relative_m'].values
        if len(subset) > 0:
            box_data.append(subset + offset)
            box_positions.append(i)
            sample_sizes.append(len(subset))
            # 68%tile 계산 (offset 포함)
            p68 = np.percentile(subset, 68) + offset
            p68_values.append(p68)
        else:
            # 데이터가 없는 경우 처리 (필요시)
            p68_values.append(np.nan) 
    
    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=0.6,
        whis=[5, 95],
        showfliers=True,
        patch_artist=True,
        medianprops=dict(color='white', linewidth=3),
        boxprops=dict(facecolor='olivedrab', edgecolor='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=2),
        capprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='o', markersize=4, 
                        markerfacecolor='lightgray', markeredgecolor='lightgray',
                        alpha=0.5)
    )
    
    # 68%tile 표시 (녹색 X)
    # box_positions와 p68_values를 사용하여 scatter plot
    valid_indices = [i for i, val in enumerate(p68_values) if not np.isnan(val)]
    valid_positions = [box_positions[i] for i in valid_indices]
    valid_p68 = [p68_values[i] for i in valid_indices]
    
    ax.scatter(valid_positions, valid_p68, marker='x', s=100, color='black', 
               linewidth=3, zorder=5, label='68th Percentile Error')

    for i, (pos, n) in enumerate(zip(box_positions, sample_sizes)):
        upper_whisker = bp['whiskers'][i * 2 + 1]
        y_top = upper_whisker.get_ydata().max()
        
        ax.text(pos, y_top * 1.08, f'n={n}', 
                fontsize=22, fontweight='bold',
                rotation=90,
                ha='center', va='bottom', color='black')
    
    # Target Error line (주황색 실선)
    threshold_plot = threshold + offset
    ax.axhline(threshold_plot, color='black', linestyle='--', linewidth=3,
               alpha=0.9, label=f'Target Error ({threshold:.4f} m)')
    
    # =========================================================================
    # Scale 설정
    # =========================================================================
    if use_log_scale:
        ax.set_yscale('log')
        ax.set_ylim(offset * 0.9, ylim_max + offset)
        
        def log_formatter(y, pos):
            original = y - offset
            if original < 0.01:
                return ''
            elif original < 0.1:
                return f'{original:.2f}'
            elif original < 1:
                return f'{original:.1f}'
            else:
                return f'{int(original)}'
        
        ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
        
        # Log scale ticks (ylim_max까지)
        all_ticks = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
        tick_values_orig = [v for v in all_ticks if v <= ylim_max]
        tick_values = [v + offset for v in tick_values_orig]
        ax.set_yticks(tick_values)
    else:
        # Linear scale
        ax.set_ylim(0, ylim_max)
        ax.tick_params(axis='y', labelsize=14)
    
    # X-axis
    xticklabels = isa_labels + ['50-60'] # 50+ 라벨을 50-60으로 변경하신다면 이렇게, 아니면 isa_labels_final 그대로
    # 여기서는 isa_labels_final을 그대로 쓰는게 논리적일 수 있으나, 
    # 원하시는 라벨이 있다면 수정하세요. 일단 isa_labels_final 사용.
    # 만약 '50-60'으로 표기를 원하시면 아래 줄 주석 해제
    # xticklabels = isa_labels + ['50+'] 
    
    ax.set_xticks(range(len(isa_labels_final)))
    ax.set_xticklabels(isa_labels_final, rotation=45, ha='right', 
                       fontsize=22, fontweight='bold')
    ax.set_xlim(-0.5, len(isa_labels_final) - 0.5)
    
    # Labels and title
    ax.set_xlabel('Impervious Surface Area (%)', fontweight='bold', fontsize=22)
    ax.set_ylabel('|Relative WSE Error| (m)', fontweight='bold', fontsize=22) # Label 변경
    ax.set_title('SWOT Relative WSE Error Distribution by Impervious Surface Area',
                 fontweight='bold', fontsize=22, pad=15)
    
    ax.tick_params(axis='y', labelsize=22)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Legend 설정 (기존 boxplot 요소들은 자동 추가가 안되므로 수동 추가 필요할 수 있음)
    # 하지만 ax.scatter와 ax.axhline에 label을 넣었으므로 legend() 호출 시 이들은 자동 포함됨.
    # boxplot의 median 등을 추가하고 싶다면 커스텀 핸들 필요.
    # 여기서는 요청하신 68%tile과 Target Error 위주로 표시.
    
    # 커스텀 레전드 핸들 생성 (Boxplot 설명 추가를 원할 경우)
    # box_handle = mpatches.Patch(facecolor='steelblue', edgecolor='black', label='IQR (25-75%)')
    # median_handle = Line2D([0], [0], color='darkred', linewidth=3, label='Median')
    
    # 현재 자동 생성되는 핸들 가져오기
    handles, labels = ax.get_legend_handles_labels()
    
    # 필요하다면 순서 조정
    # handles = [handles[1], handles[0]] # 예: Target Error 먼저
    
    ax.legend(handles=handles, loc='upper left', fontsize=22, frameon=True, fancybox=True,
              edgecolor='black', framealpha=0.95, prop={'weight': 'bold', 'size': 22})
    
    ax.grid(True, alpha=0.3, axis='y')
    
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
        
        pdf_path = save_path.replace('.png', '.pdf')
        fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {pdf_path}")
    
    return fig