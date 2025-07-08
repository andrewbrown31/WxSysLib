""""""
"""

"""

__author__ = "Michael A. Barnes"
__license__ = "Monash University"
__email__ = "michael.barnes@monash.edu"

import geopandas as gpd
import pandas as pd
from datetime import timedelta
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_lon_lat_span(geom):
    """Compute longitude and latitude span of a geometry, handling dateline wrap."""
    lons, lats = [], []

    if geom.is_empty:
        return 0, 0

    if geom.geom_type == "Polygon":
        coords = geom.exterior.coords
        lons = [pt[0] for pt in coords]
        lats = [pt[1] for pt in coords]
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            coords = part.exterior.coords
            lons.extend([pt[0] for pt in coords])
            lats.extend([pt[1] for pt in coords])
    else:
        return 0, 0  # unsupported geometry

    # Normalize longitudes to [0, 360) for wraparound span
    lons_norm = [(lon + 360) % 360 for lon in lons]
    lon_span = max(lons_norm) - min(lons_norm)
    lon_span = min(lon_span, 360 - lon_span)  # adjust for dateline crossing

    # Latitude span is normal
    lat_span = max(lats) - min(lats)

    return lon_span, lat_span

def clusters_in_locality(gdf,day_diff=7,in_lon_span=10,in_lat_span=10):
    
    # Ensure datetime format
    gdf['date'] = pd.to_datetime(gdf['date'])
    
    # Create spatial index
    sindex = gdf.sindex
    
    # Dictionary to store matches
    match_dict = {}
    
    for idx, row in tqdm(gdf.iterrows(),
                         desc="Clustering overturnings",total=len(gdf)):
        src_label = row['label']
        tmin = row['date'] - timedelta(days=day_diff)
        tmax = row['date'] + timedelta(days=day_diff)
    
        time_filtered = gdf[
            (gdf['date'] >= tmin) &
            (gdf['date'] <= tmax) &
            (gdf['label'] != row['label'])
        ]
    
        candidate_idxs = list(sindex.intersection(row['geometry'].bounds))
        candidates = gdf.iloc[candidate_idxs]
        candidates = candidates[candidates.index.isin(time_filtered.index)]
    
        matches = []
        for cidx, candidate in candidates.iterrows():
            inter = row['geometry'].intersection(candidate['geometry'])
            if not inter.is_empty:
                lon_span, lat_span = get_lon_lat_span(inter)
                if lon_span >= in_lon_span and lat_span >= in_lat_span:
                    matches.append(cidx)
    
        if matches:
            match_dict[idx] = matches
    
    # Add match info to DataFrame
    gdf['matches'] = gdf.index.map(lambda i: match_dict.get(i, []))

    return gdf

def get_connected_labels(gdf):
    # Step 1: Build the graph of label connections
    G = nx.Graph()
    
    for idx, row in tqdm(gdf.iterrows(),
                         desc="Find matches",total=len(gdf)):
        src_label = row['label']
        for match_idx in row['matches']:
            dst_label = gdf.loc[match_idx, 'label']
            if src_label != dst_label:
                G.add_edge(src_label, dst_label)
    
    # Step 2: Find connected label groups
    connected_label_groups = list(nx.connected_components(G))
    
    # Step 3: Map label -> row indices
    label_to_indices = defaultdict(set)
    for idx, row in tqdm(gdf.iterrows(),
                         desc="Map labels",total=len(gdf)):
        label_to_indices[row['label']].add(idx)
    
    # Step 4: Build the group_info dict keyed by group_id
    group_info = {}
    
    for group_id, label_group in enumerate(connected_label_groups):
        group_indices = set()
        for label in label_group:
            group_indices.update(label_to_indices[label])
    
        matched_indices = set()
        for idx in group_indices:
            matched_indices.update(gdf.at[idx, 'matches'])
    
        group_info[group_id] = {
            'labels': label_group,
            'row_indices': group_indices,
            'matched_indices': matched_indices
        }
    return group_info

def plot_geoms(gdf):
    
    # Get latitude bounds from your data for better vertical framing
    lat_min, lat_max = gdf.total_bounds[1], gdf.total_bounds[3]
    
    # Loop through each unique date and plot
    for current_date in sorted(gdf['date'].unique()):
        subset = gdf[gdf['date'] == current_date]
    
        fig, ax = plt.subplots(figsize=(12, 6))
        subset.plot(ax=ax, color='blue', edgecolor='black', alpha=0.6)
    
        # Set longitude limits to fixed -180 to 180
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
    
        ax.set_title(f'Geometries on {current_date.strftime("%Y-%m-%d %H:%M:%S")}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True)
    
        plt.show()
