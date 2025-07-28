import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def lon360to180(lon):
    return (lon + 180.0) % 360.0 - 180.0

def lon180to360(lon):
    return lon % 360.0

def coords_to_track(dfl, variable, cmap, norm):
    if len(dfl) < 2 or len(variable) != len(dfl):
        return None, norm
    
    dfl['lon'] = lon180to360(dfl['lon'])
    # longitudes e.g., 350 → 10 become 350 → 370, avoiding the fake jump.
    x = np.rad2deg(np.unwrap(np.deg2rad(dfl['lon'].to_numpy(dtype=float))))
    y = dfl['lat'].to_numpy(dtype=float)
    v = variable.to_numpy(dtype=float)
    segments = []
    colors = []
    start = 0
    for i in range(1, len(x)):
        lon_diff = abs((x[i] - x[i - 1] + 180) % 360 - 180)
        if lon_diff > 180:
            seg_x = x[start:i+1]
            seg_y = y[start:i+1]
            seg_v = 0.5 * (v[start:i] + v[start+1:i+1])

            points = np.array([seg_x, seg_y]).T.reshape(-1, 1, 2)
            segs = np.concatenate([points[:-1], points[1:]], axis=1)
            
            segments.append(segs)
            colors.append(seg_v)
            start = i 
    seg_x = x[start:]
    seg_y = y[start:]
    seg_v = 0.5 * (v[start:-1] + v[start+1:])
    points = np.array([seg_x, seg_y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    segments.append(segs)
    colors.append(seg_v)

    # Flatten
    segments = np.concatenate(segments)
    segment_variable = np.concatenate(colors)

    # Create LineCollection
    lc = LineCollection(segments, cmap=cmap, norm=norm,
                        transform=ccrs.PlateCarree())
    lc.set_array(segment_variable)

    return lc, norm


def plot_node_tracks(df, variable, var_min, var_max,
                     lon_min, lon_max, lat_min, lat_max, cmap):
    track_container = [group for sid, group in df.groupby('track_id')]
    mapcrs = ccrs.PlateCarree(central_longitude=180)
    
    fig = plt.figure(facecolor='white', constrained_layout=True, dpi=100)
    ax1 = fig.add_subplot(111, projection=mapcrs)
    
    xticks = np.arange(-180, 181, 30)
    yticks = np.arange(-70, 71, 15)
    ax1.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax1.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax1.yaxis.set_major_formatter(LatitudeFormatter())
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='k', lw=1.15)
    ax1.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='w')
    ax1.add_feature(cfeature.LAND.with_scale('50m'), facecolor='antiquewhite')
    
    norm = plt.Normalize(var_min, var_max)
    last_lines = None
    for group in track_container:
        trc, _ = coords_to_track(group, group[variable], cmap=cmap, norm=norm)
        trc.set_linewidth(1.5)
        lon0 = group['lon'].iloc[0]
        lat0 = group['lat'].iloc[0]
        track_id = group['track_id'].iloc[0]
        ax1.scatter(lon0, lat0, s=40, color='r', marker='x', transform=ccrs.PlateCarree(), zorder=99)
        ax1.text(lon0, lat0, str(track_id), transform=ccrs.PlateCarree(),fontsize=8,color='k',va='bottom',ha='left',
                 zorder=100)
        
        last_lines = ax1.add_collection(trc)

    cb = plt.colorbar(last_lines, orientation='horizontal', ax=ax1, norm=norm,
                      pad=0.04, aspect=40)
    cb.set_label(f'{variable}', labelpad=5, fontsize=10)
    
    # Set title and remove axis labels
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    plt.show()
    
    return fig