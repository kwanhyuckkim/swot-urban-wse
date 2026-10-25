def filter_best_pixc_files(files):
    file_dict = {}
    for f in files:
        match = re.match(r'(.+)_(P[IG]D\d)_(\d+)\.nc', f.name)
        if not match:
            continue
        base, processor, counter = match.group(1), match.group(2), int(match.group(3))
        if base not in file_dict:
            file_dict[base] = (f, processor, counter)
        else:
            _, old_proc, old_cnt = file_dict[base]
            if (processor == 'PGD0' and old_proc == 'PID0') or (processor == old_proc and counter > old_cnt):
                file_dict[base] = (f, processor, counter)
    return [f for f, _, _ in file_dict.values()]

def get_tile_from_filename(filename):
    parts = filename.split('_')
    if len(parts) >= 7:
        return parts[6]
    return 'UNK'

def get_heading_from_pixc(pixc_file):
    try:
        ds_tvp = xr.open_dataset(pixc_file, group='tvp')
        heading = np.nanmedian(ds_tvp['velocity_heading'].values)
        ds_tvp.close()
        return heading
    except:
        return None

def calculate_sun_position(dt, lat, lon):
    """
    Calculate sun azimuth and elevation for given datetime and location
    
    Returns:
        azimuth: degrees from North (0=N, 90=E, 180=S, 270=W)
        elevation: degrees above horizon
    """
    # Handle pandas Timestamp
    if hasattr(dt, 'to_pydatetime'):
        dt = dt.to_pydatetime()
    
    # Handle timezone - convert to naive datetime (assuming UTC)
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        # Remove timezone info (data is already in UTC)
        dt = dt.replace(tzinfo=None)
    
    # Days since J2000.0
    n = (dt - datetime(2000, 1, 1, 12, 0, 0)).total_seconds() / 86400.0
    
    # Mean longitude of the Sun
    L = (280.460 + 0.9856474 * n) % 360
    
    # Mean anomaly of the Sun
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    
    # Ecliptic longitude
    lambda_sun = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    
    # Obliquity of the ecliptic
    epsilon = math.radians(23.439 - 0.0000004 * n)
    
    # Right ascension and declination
    alpha = math.atan2(math.cos(epsilon) * math.sin(lambda_sun), math.cos(lambda_sun))
    delta = math.asin(math.sin(epsilon) * math.sin(lambda_sun))
    
    # Greenwich Mean Sidereal Time
    GMST = (18.697374558 + 24.06570982441908 * n) % 24
    
    # Local Sidereal Time
    LST = (GMST + lon / 15.0) % 24
    
    # Hour angle
    H = math.radians((LST * 15) - math.degrees(alpha))
    
    # Convert lat to radians
    phi = math.radians(lat)
    
    # Altitude (elevation)
    sin_alt = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(H)
    elevation = math.degrees(math.asin(sin_alt))
    
    # Azimuth
    cos_az = (math.sin(delta) - math.sin(phi) * sin_alt) / (math.cos(phi) * math.cos(math.asin(sin_alt)))
    cos_az = max(-1, min(1, cos_az))  # Clamp to [-1, 1]
    azimuth = math.degrees(math.acos(cos_az))
    
    if math.sin(H) > 0:
        azimuth = 360 - azimuth
    
    return azimuth, elevation

def add_arrows_and_indicators(ax, bounds, heading, sun_azimuth=None, sun_elevation=None, sat_info=None):
    """
    Add North arrow, Satellite position with heading arrow, and Sun position
    - SAT: position based on cross-track sign, arrow shows heading direction
    - SUN: position based on azimuth, arrow points toward plot center
    - N: fixed at top as reference
    """
    # 1. NORTH ARROW - fixed at top right outside
    ax.annotate('', xy=(1.06, 1.08), xytext=(1.06, 0.98),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                annotation_clip=False, zorder=10)
    ax.text(1.06, 1.10, 'N', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', clip_on=False)
    
    # 2. SATELLITE - position based on cross-track AND heading direction
    if sat_info is not None:
        sat_lon = sat_info['lon']
        sat_lat = sat_info['lat']
        sat_alt = sat_info['altitude']
        xtrk = sat_info['cross_track']
        
        # Position based on heading and cross-track
        # Heading 0-90 or 270-360: northward component
        # Heading 90-270: southward component
        if (0 <= heading < 90) or (270 <= heading < 360):
            # Northward: xtrk>0 means sat is on LEFT (west)
            if xtrk >= 0:
                sat_x, sat_y = -0.12, 0.5  # LEFT
            else:
                sat_x, sat_y = 1.12, 0.5   # RIGHT
        else:  # 90 <= heading < 270
            # Southward: xtrk>0 means sat is on RIGHT (east)
            if xtrk >= 0:
                sat_x, sat_y = 1.12, 0.5   # RIGHT
            else:
                sat_x, sat_y = -0.12, 0.5  # LEFT
        
        # SAT box
        ax.plot(sat_x, sat_y, 's', color='blue', markersize=12, transform=ax.transAxes,
                markeredgecolor='white', markeredgewidth=1, clip_on=False, zorder=15)
        ax.text(sat_x, sat_y, 'SAT', transform=ax.transAxes,
                fontsize=6, ha='center', va='center', color='white', fontweight='bold', clip_on=False, zorder=16)
        
        # Heading arrow FROM satellite box
        arrow_len = 0.12
        heading_rad = np.radians(heading)
        dx = arrow_len * np.sin(heading_rad)
        dy = arrow_len * np.cos(heading_rad)
        
        ax.annotate('', xy=(sat_x + dx, sat_y + dy), xytext=(sat_x, sat_y),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                    annotation_clip=False, zorder=14)
        
        # Heading label - position based on arrow direction
        label_x = sat_x + dx * 1.4
        label_y = sat_y + dy * 1.4
        ax.text(label_x, label_y, f'{heading:.0f}°', transform=ax.transAxes,
                ha='center', va='center', fontsize=8, fontweight='bold', color='blue', clip_on=False)
        
        # SAT info text - below the box with enough spacing
        info_text = f'Lon:{sat_lon:.2f}°\nLat:{sat_lat:.2f}°\nAlt:{sat_alt/1000:.0f}km\nXtrk:{xtrk/1000:.1f}km'
        ax.text(sat_x, sat_y - 0.10, info_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=5, color='blue', clip_on=False,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='blue', alpha=0.9, linewidth=0.5),
                zorder=15)
    
    # 3. SUN - position based on azimuth
    if sun_azimuth is not None and sun_elevation is not None:
        if sun_elevation > 0:  # Daytime
            # Determine sun position based on azimuth
            # azimuth: 0=N, 90=E, 180=S, 270=W
            if 45 <= sun_azimuth < 135:  # East (right)
                sun_x, sun_y = 1.12, 0.85
                arrow_dx, arrow_dy = -0.08, 0
                label_x, label_y = sun_x + 0.08, sun_y  # Label to the right
                label_ha, label_va = 'left', 'center'
            elif 135 <= sun_azimuth < 225:  # South (bottom)
                sun_x, sun_y = 0.5, -0.12
                arrow_dx, arrow_dy = 0, 0.08
                label_x, label_y = sun_x, sun_y - 0.08  # Label below
                label_ha, label_va = 'center', 'top'
            elif 225 <= sun_azimuth < 315:  # West (left)
                sun_x, sun_y = -0.12, 0.85
                arrow_dx, arrow_dy = 0.08, 0
                label_x, label_y = sun_x - 0.08, sun_y  # Label to the left
                label_ha, label_va = 'right', 'center'
            else:  # North (top): 315-360 or 0-45
                sun_x, sun_y = 0.5, 1.12
                arrow_dx, arrow_dy = 0, -0.08
                label_x, label_y = sun_x, sun_y + 0.08  # Label above
                label_ha, label_va = 'center', 'bottom'
            
            # SUN box (orange)
            ax.plot(sun_x, sun_y, 's', color='#FFA500', markersize=12, transform=ax.transAxes,
                    markeredgecolor='white', markeredgewidth=1, clip_on=False, zorder=15)
            ax.text(sun_x, sun_y, 'SUN', transform=ax.transAxes,
                    fontsize=5, ha='center', va='center', color='white', fontweight='bold', clip_on=False, zorder=16)
            
            # Arrow FROM sun toward center
            ax.annotate('', xy=(sun_x + arrow_dx, sun_y + arrow_dy), xytext=(sun_x, sun_y),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='#FFA500', lw=1.5),
                        annotation_clip=False, zorder=14)
            
            # Sun Az and El label - positioned away from box
            sun_label = f'Az:{sun_azimuth:.0f}° El:{sun_elevation:.0f}°'
            ax.text(label_x, label_y, sun_label, transform=ax.transAxes, 
                    ha=label_ha, va=label_va,
                    fontsize=6, fontweight='bold', color='#FFA500', clip_on=False)
        else:
            # Night - just show text
            ax.text(1.06, 0.85, 'Night', transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='#555555', clip_on=False)

# Classification colors
CLASSIFICATION_COLORS = [
    'maroon',        # 1: land - 고동색
    'gold',          # 2: land_near_water - 진한 노랑색
    'skyblue',       # 3: water_near_land - 하늘색
    'mediumblue',    # 4: open_water - 새파란색
    'dimgray',       # 5: dark_water - 진한 회색
    'cyan',          # 6: low_coh_water_near_land - 옅은 하늘색
    'darkturquoise', # 7: open_low_coh_water - 옅은 새파란색
]

CLASSIFICATION_LABELS = [
    'land', 'land_near_water', 'water_near_land', 'open_water',
    'dark_water', 'low_coh_water_near_land', 'open_low_coh_water'
]

POLYGON_COLOR = '#404040'

def process_fctimp(fctimp_str, spike_threshold=1.0, output_dir=None):
    if output_dir is None:
        output_dir = f'isa_{fctimp_str}_v2'
    Path(output_dir).mkdir(exist_ok=True)
    
    selected_12 = gpd.read_file('/work/pi_kandread_umass_edu/swot-urban/2_urban_reaches/selected_12_vD.shp')
    fctimp_val = float(fctimp_str)
    sample = selected_12[np.isclose(selected_12["fctimp_mean"], fctimp_val, atol=0.01)].copy()
    if sample.empty:
        print(f"No reach for {fctimp_str}")
        return
    
    df = pd.read_parquet(f'/work/pi_kandread_umass_edu/swot-urban/1_data_processing/fctimp_meann_{fctimp_str}_sample.parquet')
    d = df.copy()
    d['time_dt'] = pd.to_datetime(d['time_str'])
    d = d.sort_values('time_dt').reset_index(drop=True)
    d['error'] = d['wse_relative_m'] - d['wse_usgs_relative_m']  # + = overestimate, - = underestimate
    d['is_spike'] = d['error'] >= spike_threshold
    
    pixc_dir = Path(f'/project/pi_kandread_umass_edu/swot-urban/0_data_collection/SWOT_PIXC/fctimp_{fctimp_str}')
    pixcvec_dir = Path(f'/project/pi_kandread_umass_edu/swot-urban/0_data_collection/SWOT_PIXCVec/fctimp_{fctimp_str}')
    pixc_files = filter_best_pixc_files(list(pixc_dir.glob('SWOT_L2_HR_PIXC_*.nc')))
    pixcvec_files = filter_best_pixc_files(list(pixcvec_dir.glob('SWOT_L2_HR_PIXCVec_*.nc')))
    
    polygon = sample.to_crs('EPSG:4326')
    bounds = polygon.total_bounds
    
    for idx, spike_row in d[d['is_spike']].iterrows():
        print(f"Spike: {spike_row['time_str']} (error={spike_row['error']:.2f}m)")
        
        normals = d[~d['is_spike']].copy()
        if len(normals) == 0:
            continue
        normals['time_diff'] = (normals['time_dt'] - spike_row['time_dt']).abs()
        closest_normal = normals.loc[normals['time_diff'].idxmin()]
        print(f"Normal: {closest_normal['time_str']} (error={closest_normal['error']:.2f}m, Δ={closest_normal['time_diff']})")
        
        # Get Normal data
        normal_data = None
        normal_info = None
        for group_key, nf_list in match_and_group_pixc(closest_normal['time_str'], pixc_files, closest_normal['crid']).items():
            vec_list = [v for v in [find_matching_pixcvec(nf, pixcvec_files) for nf in nf_list] if v]
            if not vec_list:
                continue
            data = extract_and_merge_pixels(nf_list, vec_list, polygon)
            if data:
                normal_data = data
                normal_info = {
                    'time_str': closest_normal['time_dt'].strftime('%Y%m%dT%H%M%S'),
                    'time_dt': closest_normal['time_dt'],
                    'crid': closest_normal['crid'],
                    'error': closest_normal['error'],
                    'tile': get_tile_from_filename(nf_list[0].name),
                    'heading': data['heading'],
                    'group_key': group_key,
                    'sat_info': data['sat_info']
                }
                break
        
        # Get Spike data
        spike_data = None
        spike_info = None
        for group_key, sf_list in match_and_group_pixc(spike_row['time_str'], pixc_files, spike_row['crid']).items():
            vec_list = [v for v in [find_matching_pixcvec(sf, pixcvec_files) for sf in sf_list] if v]
            if not vec_list:
                continue
            data = extract_and_merge_pixels(sf_list, vec_list, polygon)
            if data:
                spike_data = data
                spike_info = {
                    'time_str': spike_row['time_dt'].strftime('%Y%m%dT%H%M%S'),
                    'time_dt': spike_row['time_dt'],
                    'crid': spike_row['crid'],
                    'error': spike_row['error'],
                    'tile': get_tile_from_filename(sf_list[0].name),
                    'heading': data['heading'],
                    'group_key': group_key,
                    'sat_info': data['sat_info']
                }
                break
        
        # Plot Normal-Spike comparison if both exist
        if normal_data and spike_data:
            save_comparison_plots(normal_data, spike_data, normal_info, spike_info, 
                                  polygon, bounds, output_dir, fctimp_str)
    
    print(f"\nDone: {output_dir}")

def match_and_group_pixc(time_str, files, crid, window_minutes=5):
    dt = pd.to_datetime(time_str)
    if dt.tzinfo:
        dt = dt.tz_localize(None)
    start, end = dt - timedelta(minutes=window_minutes), dt + timedelta(minutes=window_minutes)
    groups = {}
    for f in files:
        parts = f.name.split('_')
        if len(parts) >= 10:
            try:
                fs, fe = pd.to_datetime(parts[7], format='%Y%m%dT%H%M%S'), pd.to_datetime(parts[8], format='%Y%m%dT%H%M%S')
                if not (fe < start or fs > end) and parts[9] == crid:
                    key = f"{parts[7]}_{parts[8]}"
                    groups.setdefault(key, []).append(f)
            except:
                continue
    return groups

def find_matching_pixcvec(pixc_file, pixcvec_files):
    pixc_name = pixc_file.name.replace('PIXC_', 'PIXCVec_')
    for pvf in pixcvec_files:
        if pvf.name == pixc_name:
            return pvf
    return None

def extract_and_merge_pixels(pixc_files, pixcvec_files, polygon_gdf):
    all_data = [extract_pixels_with_vec(pf, pvf, polygon_gdf) for pf, pvf in zip(pixc_files, pixcvec_files)]
    all_data = [d for d in all_data if d]
    if not all_data:
        return None
    if len(all_data) == 1:
        return all_data[0]
    merged = {k: np.concatenate([d[k] for d in all_data]) for k in all_data[0].keys() 
              if k not in ['heading', 'sat_info']}
    merged['heading'] = np.nanmean([d['heading'] for d in all_data])
    # Use first satellite info (they should be very similar)
    merged['sat_info'] = all_data[0]['sat_info']
    return merged

def extract_pixels_with_vec(pixc_file, pixcvec_file, polygon_gdf):
    ds = xr.open_dataset(pixc_file, group='pixel_cloud')
    lon, lat = ds['longitude'].values, ds['latitude'].values
    bounds = polygon_gdf.total_bounds
    
    # Get tvp group data (satellite position)
    ds_tvp = xr.open_dataset(pixc_file, group='tvp')
    heading = np.nanmedian(ds_tvp['velocity_heading'].values)
    
    # Satellite position - use median values
    sat_lon = np.nanmedian(ds_tvp['longitude'].values)
    sat_lat = np.nanmedian(ds_tvp['latitude'].values)
    sat_alt = np.nanmedian(ds_tvp['altitude'].values)
    ds_tvp.close()
    
    bbox_mask = ((lon >= bounds[0]) & (lon <= bounds[2]) & (lat >= bounds[1]) & (lat <= bounds[3]))
    if bbox_mask.sum() == 0:
        ds.close()
        return None
    
    classification = ds['classification'].values[bbox_mask]
    water_mask = classification > 2
    if water_mask.sum() == 0:
        ds.close()
        return None
    
    lon_filt, lat_filt = lon[bbox_mask][water_mask], lat[bbox_mask][water_mask]
    poly = polygon_gdf.geometry.iloc[0]
    inside_mask = np.array([poly.contains(Point(x, y)) for x, y in zip(lon_filt, lat_filt)])
    if inside_mask.sum() == 0:
        ds.close()
        return None
    
    final_idx = np.where(bbox_mask)[0][water_mask][inside_mask]
    
    # Get cross-track distance
    cross_track = ds['cross_track'].values[final_idx]
    median_cross_track = np.nanmedian(cross_track)
    
    ds_vec = xr.open_dataset(pixcvec_file)
    reach_id = ds_vec['reach_id'].values[final_idx]
    node_id = ds_vec['node_id'].values[final_idx]
    obs_id = ds_vec['obs_id'].values[final_idx]
    lake_id = ds_vec['lake_id'].values[final_idx]
    
    def is_valid(arr):
        return np.array([len(x) > 0 and x != b'' for x in arr]) if arr.dtype.kind == 'S' else arr > 0
    
    selected_mask = is_valid(reach_id) & is_valid(node_id) & ~is_valid(obs_id) & ~is_valid(lake_id)
    
    result = {
        'lon': lon[final_idx], 'lat': lat[final_idx],
        'lon_all': lon, 'lat_all': lat,
        'height': ds['height'].values[final_idx],
        'classification': ds['classification'].values[final_idx],
        'bright_land': ds['bright_land_flag'].values[final_idx],
        'layover': ds['layover_impact'].values[final_idx],
        'sig0': ds['sig0'].values[final_idx],
        'coh_power': ds['coherent_power'].values[final_idx],
        'azimuth': ds['azimuth_index'].values[final_idx],
        'range': ds['range_index'].values[final_idx],
        'cross_track': cross_track,
        'selected_mask': selected_mask,
        'heading': heading,
        'sat_info': {
            'lon': sat_lon,
            'lat': sat_lat,
            'altitude': sat_alt,
            'cross_track': median_cross_track
        }
    }
    ds.close()
    ds_vec.close()
    return result

def save_comparison_plots(normal_data, spike_data, normal_info, spike_info, polygon, bounds, output_dir, fctimp_str):
    """Save side-by-side Normal vs Spike comparison plots"""
    
    time_str = spike_info['time_str']
    
    # Calculate sun positions
    poly_center_lat = (bounds[1] + bounds[3]) / 2
    poly_center_lon = (bounds[0] + bounds[2]) / 2
    
    normal_sun_az, normal_sun_el = calculate_sun_position(
        normal_info['time_dt'].to_pydatetime() if hasattr(normal_info['time_dt'], 'to_pydatetime') else normal_info['time_dt'],
        poly_center_lat, poly_center_lon
    )
    spike_sun_az, spike_sun_el = calculate_sun_position(
        spike_info['time_dt'].to_pydatetime() if hasattr(spike_info['time_dt'], 'to_pydatetime') else spike_info['time_dt'],
        poly_center_lat, poly_center_lon
    )
    
    normal_info['sun_azimuth'] = normal_sun_az
    normal_info['sun_elevation'] = normal_sun_el
    spike_info['sun_azimuth'] = spike_sun_az
    spike_info['sun_elevation'] = spike_sun_el
    
    # 1. Combined comparison (now 2x4 with sig0)
    plot_combined_comparison(normal_data, spike_data, normal_info, spike_info, polygon, bounds, output_dir, time_str, fctimp_str)
    
    # 2. Height comparison
    plot_height_comparison(normal_data, spike_data, normal_info, spike_info, polygon, bounds, output_dir, time_str, fctimp_str)
    
    # 3. Classification comparison
    plot_classification_comparison(normal_data, spike_data, normal_info, spike_info, polygon, bounds, output_dir, time_str, fctimp_str)
    
    # 4. Sig0 comparison (NEW)
    plot_sig0_comparison(normal_data, spike_data, normal_info, spike_info, polygon, bounds, output_dir, time_str, fctimp_str)
    
    # 5. Statistics comparison
    plot_statistics_comparison(normal_data, spike_data, normal_info, spike_info, output_dir, time_str, fctimp_str)
    
    print(f"  Saved: {time_str}")

def plot_combined_comparison(normal_data, spike_data, normal_info, spike_info, polygon, bounds, output_dir, time_str, fctimp_str):
    """2x4 grid: Top row = Normal, Bottom row = Spike (Height, Classification, Sig0, NLCD)"""
    
    dx = bounds[2] - bounds[0]
    dy = bounds[3] - bounds[1]
    aspect = dy / dx
    
    panel_width = 2.5
    fig_width = panel_width * 4 + 1.2  # 4 columns now
    fig_height = panel_width * aspect * 2 + 0.5
    
    fig, axes = plt.subplots(2, 4, figsize=(fig_width, fig_height), constrained_layout=True)
    
    nlcd_path = '/work/pi_kandread_umass_edu/swot-urban/0_data_collection/1_nlcd/Annual_NLCD_LndCov_2024_CU_C1V1_EPSG4326.tif'
    nlcd_colors = {11: '#476BA1', 12: '#D1DDF9', 21: '#DDC9C9', 22: '#D89382', 23: '#ED0000', 24: '#AA0000',
                   31: '#B2ADA3', 41: '#68AA63', 42: '#1C6330', 43: '#B5C98E', 52: '#CCBA7C', 71: '#E2E2C1',
                   81: '#DDD83E', 82: '#AE7229', 90: '#BBD7ED', 95: '#71A4C1'}
    
    for row, (data, info, label) in enumerate([
        (normal_data, normal_info, 'NORMAL'),
        (spike_data, spike_info, 'SPIKE')
    ]):
        unsel = ~data['selected_mask']
        sel = data['selected_mask']
        heading = info['heading']
        tile = info['tile']
        error = info['error']
        sun_az = info['sun_azimuth']
        sun_el = info['sun_elevation']
        
        # Column 0: Height
        ax = axes[row, 0]
        if sel.sum() > 0:
            ax.scatter(data['lon'][sel], data['lat'][sel], c=data['height'][sel],
                       cmap='viridis', s=0.15, alpha=0.7, marker='x', linewidths=0.15)
        if unsel.sum() > 0:
            ax.scatter(data['lon'][unsel], data['lat'][unsel], c=data['height'][unsel],
                       cmap='viridis', s=0.15, alpha=0.4, marker='o', edgecolors='none')

        polygon.boundary.plot(ax=ax, color=POLYGON_COLOR, linewidth=2)
        ax.set_title(f'{label} | {heading:.0f}°-{tile}\nError: {error:.2f}m', fontsize=10, fontweight='bold',
                     color='green' if label == 'NORMAL' else 'red')
        
        # Column 1: Classification
        ax = axes[row, 1]
        if sel.sum() > 0:
            ax.scatter(data['lon'][sel], data['lat'][sel], c=data['classification'][sel],
                       cmap=ListedColormap(CLASSIFICATION_COLORS), s=0.15, alpha=0.7, marker='x', linewidths=0.15, vmin=1, vmax=7)
        if unsel.sum() > 0:
            ax.scatter(data['lon'][unsel], data['lat'][unsel], c=data['classification'][unsel],
                       cmap=ListedColormap(CLASSIFICATION_COLORS), s=0.15, alpha=0.4, marker='o', vmin=1, vmax=7, edgecolors='none')

        polygon.boundary.plot(ax=ax, color=POLYGON_COLOR, linewidth=2)
        ax.set_title('Classification', fontsize=10)
        
        # Column 2: Sig0 (NEW)
        ax = axes[row, 2]
        if sel.sum() > 0:
            ax.scatter(data['lon'][sel], data['lat'][sel], c=data['sig0'][sel],
                       cmap='cool', s=0.15, alpha=0.7, marker='x', linewidths=0.15)
        if unsel.sum() > 0:
            ax.scatter(data['lon'][unsel], data['lat'][unsel], c=data['sig0'][unsel],
                       cmap='cool', s=0.15, alpha=0.4, marker='o', edgecolors='none')

        polygon.boundary.plot(ax=ax, color=POLYGON_COLOR, linewidth=2)
        ax.set_title('Sig0 (σ₀)', fontsize=10)
        
        # Column 3: NLCD
        ax = axes[row, 3]
        buffer = 0.005
        with rasterio.open(nlcd_path) as src:
            window = src.window(bounds[0]-buffer, bounds[1]-buffer, bounds[2]+buffer, bounds[3]+buffer)
            nlcd_data = src.read(1, window=window)
            unique_vals = np.unique(nlcd_data[nlcd_data > 0])
            cmap_list = [nlcd_colors.get(v, '#FFFFFF') for v in range(int(unique_vals.min()), int(unique_vals.max())+1)]
            extent = [bounds[0]-buffer, bounds[2]+buffer, bounds[1]-buffer, bounds[3]+buffer]
            ax.imshow(nlcd_data, extent=extent, cmap=ListedColormap(cmap_list),
                      vmin=unique_vals.min(), vmax=unique_vals.max(), interpolation='nearest')
        polygon.boundary.plot(ax=ax, color=POLYGON_COLOR, linewidth=2)
        ax.set_title('NLCD', fontsize=10)
        
        # Common settings
        for col, ax in enumerate(axes[row]):
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            ax.set_aspect('equal')
            ax.axis('off')
            # Only add arrows and satellite info to NLCD panel (column 3)
            if col == 3:
                add_arrows_and_indicators(ax, bounds, heading, sun_az, sun_el, info['sat_info'])
    
    fig.suptitle(f'ISA={fctimp_str} | Normal vs Spike Comparison\n(X=selected, O=unselected)', 
                 fontsize=11, fontweight='bold')
    plt.savefig(f'{output_dir}/COMPARE_{time_str}_Combined.png', dpi=1200, bbox_inches='tight', pad_inches=0.15)
    plt.close()

def plot_height_comparison(normal_data, spike_data, normal_info, spike_info, polygon, bounds, output_dir, time_str, fctimp_str):
    """Side-by-side height comparison"""
    
    dx = bounds[2] - bounds[0]
    dy = bounds[3] - bounds[1]
    aspect = dy / dx
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6 * aspect))
    
    for ax, (data, info, label) in zip(axes, [
        (normal_data, normal_info, 'NORMAL'),
        (spike_data, spike_info, 'SPIKE')
    ]):
        unsel = ~data['selected_mask']
        sel = data['selected_mask']
        
        if sel.sum() > 0:
            sc = ax.scatter(data['lon'][sel], data['lat'][sel], c=data['height'][sel],
                           cmap='viridis', s=0.15, alpha=0.7, marker='x', linewidths=0.15)
            plt.colorbar(sc, ax=ax, label='Height (m)', shrink=0.7)
            
        if unsel.sum() > 0:
            ax.scatter(data['lon'][unsel], data['lat'][unsel], c=data['height'][unsel],
                       cmap='viridis', s=0.15, alpha=0.4, marker='o', edgecolors='none')
        
        polygon.boundary.plot(ax=ax, color=POLYGON_COLOR, linewidth=2)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{label} | {info["heading"]:.0f}°-{info["tile"]}\nError: {info["error"]:.2f}m', 
                     fontsize=11, fontweight='bold', color='green' if label == 'NORMAL' else 'red')
    
    fig.suptitle(f'ISA={fctimp_str} | Height Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f'{output_dir}/COMPARE_{time_str}_Height.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_classification_comparison(normal_data, spike_data, normal_info, spike_info, polygon, bounds, output_dir, time_str, fctimp_str):
    """Side-by-side classification comparison"""
    
    dx = bounds[2] - bounds[0]
    dy = bounds[3] - bounds[1]
    aspect = dy / dx
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6 * aspect))
    
    for ax, (data, info, label) in zip(axes, [
        (normal_data, normal_info, 'NORMAL'),
        (spike_data, spike_info, 'SPIKE')
    ]):
        unsel = ~data['selected_mask']
        sel = data['selected_mask']
        if sel.sum() > 0:
            ax.scatter(data['lon'][sel], data['lat'][sel], c=data['classification'][sel],
                       cmap=ListedColormap(CLASSIFICATION_COLORS), s=0.15, alpha=0.7, marker='x', linewidths=0.15, vmin=1, vmax=7)
            
        if unsel.sum() > 0:
            ax.scatter(data['lon'][unsel], data['lat'][unsel], c=data['classification'][unsel],
                       cmap=ListedColormap(CLASSIFICATION_COLORS), s=0.15, alpha=0.4, marker='o', vmin=1, vmax=7, edgecolors='none')

        
        polygon.boundary.plot(ax=ax, color=POLYGON_COLOR, linewidth=2)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{label} | {info["heading"]:.0f}°-{info["tile"]}\nError: {info["error"]:.2f}m', 
                     fontsize=11, fontweight='bold', color='green' if label == 'NORMAL' else 'red')
    
    legend = [Patch(facecolor=CLASSIFICATION_COLORS[i], label=CLASSIFICATION_LABELS[i]) for i in range(7)]
    fig.legend(handles=legend, loc='center right', bbox_to_anchor=(0.99, 0.5), fontsize=9)
    
    fig.suptitle(f'ISA={fctimp_str} | Classification Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.88, 0.94])
    plt.savefig(f'{output_dir}/COMPARE_{time_str}_Classification.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_sig0_comparison(normal_data, spike_data, normal_info, spike_info, polygon, bounds, output_dir, time_str, fctimp_str):
    """Side-by-side sig0 (radar backscatter) comparison"""
    
    dx = bounds[2] - bounds[0]
    dy = bounds[3] - bounds[1]
    aspect = dy / dx
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6 * aspect))
    
    # Get common vmin/vmax for both plots
    all_sig0 = np.concatenate([
        normal_data['sig0'][normal_data['selected_mask']],
        spike_data['sig0'][spike_data['selected_mask']]
    ])
    # Filter out invalid values
    valid_sig0 = all_sig0[np.isfinite(all_sig0)]
    if len(valid_sig0) > 0:
        vmin = np.percentile(valid_sig0, 2)
        vmax = np.percentile(valid_sig0, 98)
    else:
        vmin, vmax = None, None
    
    for ax, (data, info, label) in zip(axes, [
        (normal_data, normal_info, 'NORMAL'),
        (spike_data, spike_info, 'SPIKE')
    ]):
        unsel = ~data['selected_mask']
        sel = data['selected_mask']
        if sel.sum() > 0:
            sc = ax.scatter(data['lon'][sel], data['lat'][sel], c=data['sig0'][sel],
                           cmap='cool', s=0.15, alpha=0.7, marker='x', linewidths=0.15,
                           vmin=vmin, vmax=vmax)
            plt.colorbar(sc, ax=ax, label='Sig0 (dB)', shrink=0.7)
            
        if unsel.sum() > 0:
            ax.scatter(data['lon'][unsel], data['lat'][unsel], c=data['sig0'][unsel],
                       cmap='cool', s=0.15, alpha=0.4, marker='o', edgecolors='none',
                       vmin=vmin, vmax=vmax)

        
        polygon.boundary.plot(ax=ax, color=POLYGON_COLOR, linewidth=2)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{label} | {info["heading"]:.0f}°-{info["tile"]}\nError: {info["error"]:.2f}m', 
                     fontsize=11, fontweight='bold', color='green' if label == 'NORMAL' else 'red')
    
    fig.suptitle(f'ISA={fctimp_str} | Sig0 (σ₀) Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f'{output_dir}/COMPARE_{time_str}_Sig0.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_statistics_comparison(normal_data, spike_data, normal_info, spike_info, output_dir, time_str, fctimp_str):
    """Side-by-side statistics comparison"""
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for row, (data, info, label) in enumerate([
        (normal_data, normal_info, 'NORMAL'),
        (spike_data, spike_info, 'SPIKE')
    ]):
        sel = data['selected_mask']
        color = 'green' if label == 'NORMAL' else 'red'
        
        # Classification distribution
        axes[row, 0].bar(range(1, 8), [np.sum(data['classification'][sel] == i) for i in range(1, 8)], color=color, alpha=0.7)
        axes[row, 0].set_xlabel('Classification')
        axes[row, 0].set_ylabel('Count')
        axes[row, 0].set_title(f'{label}: Classification')
        axes[row, 0].grid(alpha=0.3)
        
        # Bright land
        axes[row, 1].bar(range(2), [np.sum(data['bright_land'][sel] == i) for i in range(2)], color=color, alpha=0.7)
        axes[row, 1].set_xlabel('Bright Land')
        axes[row, 1].set_ylabel('Count')
        axes[row, 1].set_title(f'{label}: Bright Land')
        axes[row, 1].set_xticks([0, 1])
        axes[row, 1].grid(alpha=0.3)
        
        # Boxplots
        if sel.sum() > 0:
            for col, key in zip([2, 3, 4], ['layover', 'sig0', 'coh_power']):
                bp = axes[row, col].boxplot(data[key][sel], vert=True, patch_artist=True)
                bp['boxes'][0].set_facecolor(color)
                bp['boxes'][0].set_alpha(0.5)
                axes[row, col].set_ylabel(key.replace('_', ' ').title())
                axes[row, col].set_title(f'{label}: {key.replace("_", " ").title()}')
                axes[row, col].grid(alpha=0.3)
    
    fig.suptitle(f'ISA={fctimp_str} | Statistics Comparison\nNormal (Error={normal_info["error"]:.2f}m) vs Spike (Error={spike_info["error"]:.2f}m)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f'{output_dir}/COMPARE_{time_str}_Statistics.png', dpi=300, bbox_inches='tight')
    plt.close()