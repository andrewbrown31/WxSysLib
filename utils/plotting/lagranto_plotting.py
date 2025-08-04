#!/usr/bin/env python
import numpy as np
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

def plot_traj(df, variable, var_min, var_max,
              lon_min, lon_max, lat_min, lat_max, cmap, title1, title2):
    track_container = [group for sid, group in df.groupby('track_id')]
    mapcrs = ccrs.PlateCarree(central_longitude=180)
    
    fig = plt.figure(figsize=(7, 4.5), facecolor='white', constrained_layout=True, dpi=100)
    ax1 = fig.add_subplot(111, projection=mapcrs)
    
    xticks = np.arange(-180, 181, 40)
    yticks = np.arange(-70, 71, 20)
    ax1.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax1.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax1.yaxis.set_major_formatter(LatitudeFormatter())
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', lw=1.25)
    ax1.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='w')
    ax1.add_feature(cfeature.LAND.with_scale('50m'), facecolor='antiquewhite')
    
    global_min = var_min
    global_max = var_max
    norm = plt.Normalize(global_min, global_max)

    last_lines = None
    for group in track_container:
        trc, _ = coords_to_track(group, group[variable], cmap=cmap, norm=norm)
        trc.set_linewidth(1)
        last_lines = ax1.add_collection(trc)

    cb = plt.colorbar(last_lines, orientation='horizontal', ax=ax1, norm=norm,
                      pad=0.04, aspect=40)
    cb.set_label(f'{variable}', labelpad=5, fontsize=10)
    
    # Set title and remove axis labels
    ax1.set_title(f"{title1}", loc='left', fontsize=14, pad=5)
    ax1.set_title(f"{title2}", loc='right', fontsize=14, pad=5)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    plt.show()
    
    return fig

def animate_traj(df, variable, var_min, var_max, lon_min, lon_max, lat_min, lat_max, interval, mycamp, title):
    track_container = {sid: group.sort_values('time').reset_index(drop=True)
                       for sid, group in df.groupby('track_id')}
    
    mapcrs = ccrs.PlateCarree(central_longitude=180)
    fig = plt.figure(figsize=(7, 4.5), facecolor='white', constrained_layout=True, dpi=100)
    ax = fig.add_subplot(111, projection=mapcrs)
    
    xticks = np.arange(-180, 181, 40)
    yticks = np.arange(-70, 71, 20)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_extent([180-abs(180-lon_min), 180+abs(lon_max-180), lat_min, lat_max], ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', lw=1.25)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='w')
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='antiquewhite')
    ax.set_title(title, fontsize=14, pad=5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    global_min = var_min
    global_max = var_max
    norm = plt.Normalize(global_min, global_max)
    cmap = plt.get_cmap(mycamp)
    
    scatters = {}
    for track_id, group in track_container.items():
        x0 = group.loc[0, 'lon']
        y0 = group.loc[0, 'lat']
        p0 = group.loc[0, variable]
        # Use the normalized pressure value to determine the color
        color = cmap(norm(p0))
        scat = ax.scatter(x0, y0, s=5, color=[color], edgecolor='k', zorder=99,
                          transform=ccrs.PlateCarree())
        scatters[track_id] = scat

    max_frames = max(len(group) for group in track_container.values())
    
    def update(frame):
        for track_id, group in track_container.items():
            if frame < len(group):
                current = group.loc[frame]
                x, y, val = current['lon'], current['lat'], current[variable]
                scatters[track_id].set_offsets([[x, y]])
                new_color = cmap(norm(val))
                scatters[track_id].set_color([new_color])
        first_track = next(iter(track_container.values()))
        if frame < len(first_track):
            current_time = first_track.loc[frame, 'time']
            current_time = str(int(current_time))
            ax.set_title(f'{str(current_time)} hrs', loc='right', fontsize=14, pad=5)

        return list(scatters.values())
    ax.set_title("")
    ax.set_title(f"{title}", loc='left', fontsize=14, pad=5)
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, orientation='horizontal', ax=ax, pad=0.04, aspect=40)
    cb.set_label(variable, labelpad=5, fontsize=10)
    
    plt.close(fig)
    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=interval,
                                  blit=True, repeat=True)
    
    return ani