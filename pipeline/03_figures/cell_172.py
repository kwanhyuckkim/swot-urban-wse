import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 스타일 초기화 및 설정
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

def resolve_overlaps(values, min_dist):
    """
    겹치는 값들을 찾아 오른쪽(위)으로 밀어내어 위치를 조정하는 함수.
    values: (값, 원래인덱스) 튜플의 리스트 (값 기준 정렬됨)
    min_dist: 텍스트 간 최소 거리
    """
    n = len(values)
    if n == 0:
        return []
    
    # 1. 값만 추출하여 위치 배열 생성
    positions = np.array([v[0] for v in values], dtype=float)
    
    # 2. 겹침 해결 (오른쪽으로 밀기)
    # 여러 번 반복하여 연쇄적인 겹침을 해소합니다.
    for _ in range(3): # 3번 정도 반복하면 대부분 해소됨
        for i in range(1, n):
            dist = positions[i] - positions[i-1]
            if dist < min_dist:
                # 겹치면 뒤의 것을 밈
                push = min_dist - dist
                positions[i] += push
    
    return positions

def plot_cdf_single(ax, data, y_col, geom_col, geometry_categories, title, threshold=0.1832, 
                    show_xlabel=True, zone_key=None):
    """
    CDF plot with tight axis labels and no connecting lines.
    """
    ax.set_facecolor('white')
    
    # 색상 설정
    colors = {
        'Neither': 'dimgray',     
        'Near': 'teal',         
        'Far': 'coral',           
        'Both': 'olivedrab'          
    }
    
    label_map = {
        'Neither': 'Neither',
        'Near': 'Near-Nadir',
        'Far': 'Far-Nadir',
        'Both': 'Both'
    } if zone_key == 'Perpendicular' else {
        'Neither': 'Neither',
        'Near': 'Near',
        'Far': 'Far',
        'Both': 'Both'
    }
    
    stats_dict = {}
    
    # 라벨링을 위한 데이터 수집
    p68_items = []      # (값, 색상, 라벨텍스트)
    cdf_T_items = []    # (값, 색상, 라벨텍스트)
    
    for cat in geometry_categories:
        cat_data = data[data[geom_col] == cat][y_col].dropna()
        if len(cat_data) > 0:
            sorted_data = np.sort(cat_data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            color = colors.get(cat, 'gray')
            ax.plot(sorted_data, cdf, color=color, 
                    label=f'{label_map[cat]} (n={len(cat_data)})', linewidth=2)
            
            p68 = np.percentile(cat_data, 68)
            pct_above_T = 100 * (cat_data > threshold).sum() / len(cat_data)
            cdf_at_T = (cat_data <= threshold).sum() / len(cat_data)
            
            stats_dict[cat] = {
                'n': len(cat_data), 'p68': p68, 'pct_above_T': pct_above_T,
                'cdf_at_T': cdf_at_T, 'color': color, 'label': label_map[cat]
            }
            
            # 마커 (교차점)
            ax.scatter([p68], [0.68], color=color, s=30, zorder=5)
            ax.scatter([threshold], [cdf_at_T], color=color, s=40, zorder=5, marker='s')
            
            # 점선 가이드
            ax.vlines(x=p68, ymin=0, ymax=0.68, colors=color, linestyles=':', linewidth=1, alpha=0.6)
            ax.hlines(y=cdf_at_T, xmin=0, xmax=threshold, colors=color, linestyles=':', linewidth=1, alpha=0.6)
            
            # 라벨 데이터 저장
            p68_items.append({'val': p68, 'color': color, 'text': f'{p68:.2f}'})
            cdf_T_items.append({'val': cdf_at_T, 'color': color, 'text': f'{cdf_at_T:.2f}'.lstrip('0')})

    # 기준선 (Target Error, 68%)
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axhline(y=0.68, color='black', linewidth=1.5, alpha=0.8)
    
    # 고정 텍스트 (68%, T값)
    ax.text(0.01, 0.69, '68%', transform=ax.get_yaxis_transform(), 
            color='black', fontsize=12, fontweight='bold', ha='left', va='bottom')
    ax.text(threshold, 1.01, f'T={threshold}m', transform=ax.get_xaxis_transform(),
            color='black', fontsize=11, fontweight='bold', ha='center', va='bottom')

    # -------------------------------------------------------------------------
    # 스마트 라벨 배치 (선 없음, 축 밀착, 겹침 방지)
    # -------------------------------------------------------------------------
    
    # 1. X축 라벨 (68%tile Error)
    # 정렬
    p68_items.sort(key=lambda x: x['val'])
    
    # 겹침 계산 (최소 간격 0.15)
    x_vals = [(item['val'], i) for i, item in enumerate(p68_items)]
    adj_x_vals = resolve_overlaps(x_vals, min_dist=0.15)
    
    # 그리기: y=0.01 (축 바로 위)
    y_base_x_label = 0.01 
    for i, item in enumerate(p68_items):
        new_x = adj_x_vals[i]
        
        # 화살표 없이 텍스트만 배치
        ax.text(new_x, y_base_x_label, item['text'], 
                transform=ax.get_xaxis_transform(), # x는 data좌표, y는 axis(0~1)좌표
                color=item['color'], fontsize=10.5, fontweight='bold', 
                ha='center', va='bottom')

    # 2. Y축 라벨 (CDF at T)
    # 정렬
    cdf_T_items.sort(key=lambda x: x['val'])
    
    # 겹침 계산 (최소 간격 0.06)
    y_vals = [(item['val'], i) for i, item in enumerate(cdf_T_items)]
    adj_y_vals = resolve_overlaps(y_vals, min_dist=0.06)
    
    # 그리기: x=0.01 (축 바로 옆)
    x_base_y_label = 0.01
    for i, item in enumerate(cdf_T_items):
        new_y = adj_y_vals[i]
        
        # 화살표 없이 텍스트만 배치
        ax.text(x_base_y_label, new_y, item['text'], 
                transform=ax.get_yaxis_transform(), # x는 axis(0~1)좌표, y는 data좌표
                color=item['color'], fontsize=10.5, fontweight='bold', 
                ha='left', va='center')

    ax.set_ylim(0, 1)
    
    if show_xlabel:
        ax.set_xlabel(r'$|Relative\ WSE\ Error|$ (m)', color='black', fontsize=13)
    ax.set_ylabel('CDF', color='black', fontsize=13)
    ax.set_title(title, color='black', fontsize=14, fontweight='bold', pad=10)
    ax.tick_params(colors='black', labelsize=11)
    
    # 범례 위치 변경: Right Center
    ax.legend(loc='center right', fontsize=11, facecolor='white', 
              edgecolor='gray', labelcolor='black', framealpha=0.9)
    
    ax.grid(True, alpha=0.3, color='gray')
    for spine in ax.spines.values():
        spine.set_color('black')
    
    return stats_dict

def plot_stats_panel(ax, all_stats, threshold=0.1832):
    # (기존 통계 패널 코드 유지)
    ax.set_facecolor('white')
    ax.axis('off')
    
    colors = {'Neither': 'dimgray', 'Near': 'teal', 'Far': 'coral', 'Both': 'olivedrab'}
    categories = ['Neither', 'Near', 'Far', 'Both']
    zones = ['Parallel', 'Perpendicular', 'Other']
    
    ax.text(0.5, 1, 'Statistics Summary', transform=ax.transAxes,
            fontsize=16, fontweight='bold', color='black', ha='center', va='top')
    ax.text(0.5, 0.92, '68th Percentile |Error| (m)', transform=ax.transAxes,
            fontsize=13, fontweight='bold', color='black', ha='center', va='top')
    
    header = f"{'Category':<10}{'Parallel':>14}{'Perpendicular':>18}{'Other':>12}"
    ax.text(0.5, 0.84, header, transform=ax.transAxes,
            fontsize=12, color='black', ha='center', va='top', family='monospace')
    ax.text(0.5, 0.79, "─" * 54, transform=ax.transAxes,
            fontsize=12, color='gray', ha='center', va='top', family='monospace')
    
    y_pos = 0.74
    for cat in categories:
        row = f"{cat:<10}"
        for zone in zones:
            if zone in all_stats and cat in all_stats[zone]:
                p68 = all_stats[zone][cat]['p68']
                row += f"{p68:>18.3f}" if zone == 'Perpendicular' else f"{p68:>14.3f}" if zone == 'Parallel' else f"{p68:>12.3f}"
            else:
                row += f"{'N/A':>18}" if zone == 'Perpendicular' else f"{'N/A':>14}" if zone == 'Parallel' else f"{'N/A':>12}"
        ax.text(0.5, y_pos, row, transform=ax.transAxes, fontsize=12, color=colors[cat], ha='center', va='top', family='monospace')
        y_pos -= 0.07
    
    y_pos -= 0.06
    ax.text(0.5, y_pos, f'% with |Error| > T (T={threshold}m)', transform=ax.transAxes,
            fontsize=13, fontweight='bold', color='black', ha='center', va='top')
    
    y_pos -= 0.08
    ax.text(0.5, y_pos, header, transform=ax.transAxes, fontsize=12, color='black', ha='center', va='top', family='monospace')
    y_pos -= 0.05
    ax.text(0.5, y_pos, "─" * 54, transform=ax.transAxes, fontsize=12, color='gray', ha='center', va='top', family='monospace')
    y_pos -= 0.05
    
    for cat in categories:
        row = f"{cat:<10}"
        for zone in zones:
            if zone in all_stats and cat in all_stats[zone]:
                pct = all_stats[zone][cat]['pct_above_T']
                row += f"{pct:>17.1f}%" if zone == 'Perpendicular' else f"{pct:>13.1f}%" if zone == 'Parallel' else f"{pct:>11.1f}%"
            else:
                row += f"{'N/A':>18}" if zone == 'Perpendicular' else f"{'N/A':>14}" if zone == 'Parallel' else f"{'N/A':>12}"
        ax.text(0.5, y_pos, row, transform=ax.transAxes, fontsize=12, color=colors[cat], ha='center', va='top', family='monospace')
        y_pos -= 0.07

def plot_cdf_2x2(df, y_col, geom_col='geometry_category', threshold=0.1832, 
                 figsize=(14, 12), xlim=(0, 2), save=True, show=True):
    df_plot = df.dropna(subset=[y_col, 'flow_angle', geom_col]).copy()
    if len(df_plot) == 0:
        return None
    
    df_plot['flow_zone'] = df_plot['flow_angle'].apply(get_flow_angle_zone)
    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor='white')
    
    zones = [
        ('Parallel', 'Parallel (350-10°, 170-190°)', axes[0, 0]),
        ('Perpendicular', 'Perpendicular (80-100°, 260-280°)', axes[0, 1]),
        ('Other', 'Other Angles', axes[1, 0]),
    ]
    
    all_stats = {}
    geometry_cats = ['Neither', 'Near', 'Far', 'Both']
    
    for zone_key, zone_title, ax in zones:
        df_zone = df_plot[df_plot['flow_zone'] == zone_key]
        title = f'{zone_title}\n(n={len(df_zone):,})'
        stats = plot_cdf_single(ax, df_zone, y_col, geom_col, geometry_cats, title, 
                                threshold, show_xlabel=True, zone_key=zone_key)
        ax.set_xlim(xlim)
        all_stats[zone_key] = stats
    
    plot_stats_panel(axes[1, 1], all_stats, threshold)
    
    fig.suptitle(r'CDF of $|Relative\ WSE\ Error|$ by ISA Location and Flow Angle', 
                 fontsize=18, fontweight='bold', color='black', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save:
        filename = f"figure5_cdf_2x2_{y_col}_{geom_col}.png"
        fig.savefig(OUTPUT_DIR / filename, dpi=300, facecolor='white', bbox_inches='tight')
        filename = f"figure5_cdf_2x2_{y_col}_{geom_col}.pdf"
        fig.savefig(OUTPUT_DIR / filename, dpi=300, facecolor='white', bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR / filename}")
    
    if show:
        plt.show()
    
    return fig