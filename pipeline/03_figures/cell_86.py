# FIGURE 3: Quality Flags Heatmap - Error 기반 (X축: ISA %)
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

def create_figure3_quality_flags(
    df_matched,
    threshold=0.1832,
    figsize=(16, 10),
    save_path=None,
    dpi=300
):
    """
    Figure 3: Quality flags heatmap colored by error.
    
    - X축: ISA (%) - 정렬된 샘플들
    - Y축: Quality flags
    - 색상: Blue (Low Error) -> White (threshold) -> Red (High Error)
    """
    
    df = df_matched.copy()
    
    # Data preparation
    df['error_relative_m'] = (df['wse_relative_m'] - df['wse_usgs_relative_m']).abs()
    df['isa_pct'] = df['fctimp_mean'] * 100
    
    if 'time_dt' not in df.columns:
        if 'time_str' in df.columns:
            df['time_dt'] = pd.to_datetime(df['time_str'])
    else:
        df['time_dt'] = pd.to_datetime(df['time_dt'])
    
    # Sort by ISA (정렬!)
    df = df.sort_values(['isa_pct']).reset_index(drop=True)
    n_samples = len(df)
    
    print(f"Figure 3: {n_samples} samples")
    print(f"  ISA range: {df['isa_pct'].min():.1f}% ~ {df['isa_pct'].max():.1f}%")
    
    # Decode quality flags
    has_quality_flags = 'reach_q_b' in df.columns
    if has_quality_flags:
        df['classification_qual_suspect'] = (df['reach_q_b'] & 2) > 0
        df['geolocation_qual_suspect'] = (df['reach_q_b'] & 4) > 0
        df['water_fraction_suspect'] = (df['reach_q_b'] & 8) > 0
        df['bright_land'] = (df['reach_q_b'] & 128) > 0
        df['few_area_observations'] = (df['reach_q_b'] & 1024) > 0
        df['few_wse_observations'] = (df['reach_q_b'] & 2048) > 0
        df['far_range_suspect'] = (df['reach_q_b'] & 8192) > 0
        df['near_range_suspect'] = (df['reach_q_b'] & 16384) > 0
        df['partially_observed'] = (df['reach_q_b'] & 32768) > 0
        df['classification_qual_degraded'] = (df['reach_q_b'] & 262144) > 0
        df['geolocation_qual_degraded'] = (df['reach_q_b'] & 524288) > 0
        df['lake_flagged'] = (df['reach_q_b'] & 4194304) > 0
        df['below_min_fit_points'] = (df['reach_q_b'] & 33554432) > 0
        df['no_area_observations'] = (df['reach_q_b'] & 67108864) > 0
        df['no_wse_observations'] = (df['reach_q_b'] & 134217728) > 0
        df['no_pixels'] = (df['reach_q_b'] & 268435456) > 0
        
        flag_names = [
            'classification_qual_suspect', 'geolocation_qual_suspect',
            'water_fraction_suspect', 'bright_land', 'few_area_observations',
            'few_wse_observations', 'far_range_suspect', 'near_range_suspect',
            'partially_observed', 'classification_qual_degraded',
            'geolocation_qual_degraded', 'lake_flagged', 'below_min_fit_points',
            'no_area_observations', 'no_wse_observations', 'no_pixels'
        ]
        
        flag_display_names = [
            'class_suspect', 'geo_suspect',
            'water_frac', 'bright_land', 'few_area',
            'few_wse', 'far_range', 'near_range',
            'partial_obs', 'class_degraded',
            'geo_degraded', 'lake_flag', 'min_fit_pts',
            'no_area', 'no_wse', 'no_pixels'
        ]
    else:
        print("No quality flags found!")
        return None
    
    # Build heatmap matrix
    n_flags = len(flag_names)
    heatmap_data = np.full((n_flags, n_samples), np.nan)
    
    error_values = df['error_relative_m'].values
    
    for flag_idx, flag in enumerate(flag_names):
        mask = df[flag].values
        heatmap_data[flag_idx, mask] = error_values[mask]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom colormap: Blue -> White -> Red (diverging at threshold)
    colors = ['#0571b0', '#92c5de', '#f7f7f7', '#f4a582', '#ca0020']
    cmap = LinearSegmentedColormap.from_list('blue_white_red', colors, N=256)
    cmap.set_bad(color='lightgray')  # NaN = lightgray
    
    # Error range
    vmin = 0
    vmax = min(2.0, np.nanmax(error_values))
    
    # TwoSlopeNorm: threshold를 중심으로 발산
    norm = TwoSlopeNorm(vmin=vmin, vcenter=threshold, vmax=vmax)
    
    # Plot heatmap
    im = ax.imshow(
        heatmap_data,
        aspect='auto',
        cmap=cmap,
        norm=norm,
        interpolation='nearest'
    )
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.045)
    cbar.ax.tick_params(labelsize=12)
    
    # Colorbar label
    cbar.ax.set_ylabel('|Relative Error| (m)', fontsize=14, fontweight='bold', rotation=270, labelpad=20)
    
    # Threshold marker on colorbar
    cbar.ax.axhline(threshold, color='black', linewidth=2, linestyle='-')
    
    # Target Error 라벨 (오른쪽)
    cbar.ax.text(-0.3, threshold, f'Target Error = {threshold} m', 
                 fontsize=12, fontweight='bold', va='center', ha='right', rotation=270,
                 transform=cbar.ax.get_yaxis_transform())
    
    # Low/High Error labels (왼쪽)
    cbar.ax.text(-0.3, vmin + (threshold - vmin) * 0.5, 'Low\nError', 
                 fontsize=12, fontweight='bold', va='center', ha='right', color='#0571b0',
                 transform=cbar.ax.get_yaxis_transform())
    cbar.ax.text(-0.3, threshold + (vmax - threshold) * 0.5, 'High\nError', 
                 fontsize=12, fontweight='bold', va='center', ha='right', color='#ca0020',
                 transform=cbar.ax.get_yaxis_transform())
    
    # X-axis: ISA (%) - 정렬된 샘플 기준으로 tick 위치 계산
    isa_values = df['isa_pct'].values
    isa_tick_values = [0, 0.5, 1, 2, 4, 6, 10, 20, 30, 40, 50]  # 5% 추가, 60% 제거
    
    x_ticks = []
    x_labels = []
    
    for isa_target in isa_tick_values:
        idx = np.searchsorted(isa_values, isa_target)
        if idx < n_samples:
            x_ticks.append(idx)
            x_labels.append(str(isa_target))
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
    ax.set_xlabel('ISA (%)', fontsize=14, fontweight='bold')
    
    # Y-axis: Quality flags
    ax.set_ylabel('Quality Flag', fontsize=14, fontweight='bold')
    ax.set_yticks(range(n_flags))
    ax.set_yticklabels(flag_display_names, fontsize=12, fontweight='bold')
    
    # Title
    ax.set_title('Quality Flags vs Error: Can flags predict high error?',
                 fontsize=16, fontweight='bold', pad=15)
    
    # Grid lines between flags
    for i in range(n_flags - 1):
        ax.axhline(i + 0.5, color='white', linewidth=1, alpha=0.8)
    
    # Thick black border
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    # Statistics
    print("\n[Flag Statistics]")
    print(f"{'Flag':<25} {'Count':>8} {'High Error':>12} {'Rate':>8}")
    
    for flag_idx, (flag, display) in enumerate(zip(flag_names, flag_display_names)):
        mask = df[flag].values
        n_flagged = mask.sum()
        if n_flagged > 0:
            high_error_rate = (df.loc[mask, 'error_relative_m'] >= threshold).sum() / n_flagged * 100
            print(f"{display:<25} {n_flagged:>8} {int(high_error_rate):>11}% {high_error_rate:>7.1f}%")
    
    overall_rate = (df['error_relative_m'] >= threshold).sum() / len(df) * 100
    print(f"{'Overall':<25} {len(df):>8} {int(overall_rate):>11}% {overall_rate:>7.1f}%")
    
    plt.tight_layout()
    
    # Save
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"\nSaved: {save_path}")
        
        pdf_path = save_path.replace('.png', '.pdf')
        fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved: {pdf_path}")
    
    return fig