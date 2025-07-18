""""""
"""
This file is part of WaveBreaking.

WaveBreaking provides indices to detect, classify
and track Rossby Wave Breaking (RWB) in climate and weather data.
The tool was developed during my master thesis at the University of Bern.
Link to thesis: https://occrdata.unibe.ch/students/theses/msc/406.pdf

---

Events post-processing functions
"""

__author__ = "Severin Kaderli"
__license__ = "MIT"
__email__ = "severin.kaderli@unibe.ch"

# import modules
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import itertools as itertools
from sklearn.metrics import DistanceMetric
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon, mapping, shape
from shapely.ops import unary_union

dist = DistanceMetric.get_metric("haversine")

from wavebreaking.utils.data_utils import (
    check_argument_types,
    check_empty_dataframes,
    get_dimension_attributes,
)
from wavebreaking.utils import index_utils


@check_argument_types(["data", "events"], [xr.DataArray, gpd.GeoDataFrame])
@check_empty_dataframes
@get_dimension_attributes("data")
def to_xarray(data, events, flag="ones", name="flag", *args, **kwargs):
    """
    Create xarray.DataArray from events stored in a geopandas.GeoDataFrame.
    Grid cells where an event is present are flagged with the value 1.
    Dimension names ("time_name", "lon_name", "lat_name"), size ("ntime", "nlon", "nlat")
    and resolution ("dlon", "dlat") can be passed as key=value argument.

    Parameters
    ----------
        data : xarray.DataArray
            data used for the index calculation
        events : geopandas.GeoDataFrame
            GeoDataFrame with the date and geometry for each event
        flag : string, optional
            column name of the events geopandas.GeoDataFrame
            flag is set where an event is present
            default value is "ones"
        name : string, optional
            name of the xarray variable that is created

    Returns
    -------
        flag: xarray.DataArray
            Data with events flagged with the value 1
    """

    # get grid points
    lon, lat = np.meshgrid(data[kwargs["lon_name"]], data[kwargs["lat_name"]])
    lonf, latf = lon.flatten(), lat.flatten()
    points = gpd.GeoDataFrame(
        pd.DataFrame({"lon": lonf, "lat": latf}),
        geometry=gpd.points_from_xy(lonf, latf),
    )

    # get coordinates of all events at the same time step
    buffer = events.copy()
    buffer.geometry = buffer.geometry.buffer(
        ((kwargs["dlon"] + kwargs["dlat"]) / 2) / 2
    )
    merged = gpd.sjoin(buffer, points, how="inner", predicate="contains").sort_index()
    #print(merged)

    # create empty xarray.Dataset with the same dimension as the original Dataset
    data_flagged = xr.zeros_like(data)

    # flag coordinates in DataSet
    if flag == "ones":
        set_val = np.ones(len(merged))
    else:
        try:
            set_val = merged[flag].values
        except KeyError:
            errmsg = "{} is not a column of the events geopandas.GeoDataFrame.".format(
                flag
            )
            raise KeyError(errmsg)

    data_flagged.loc[
        {
            kwargs["time_name"]: merged.date.to_xarray(),
            kwargs["lat_name"]: merged.lat.to_xarray(),
            kwargs["lon_name"]: merged.lon.to_xarray(),
        }
    ] = set_val

    # change type and name
    if flag == 'ones':
        data_flagged = data_flagged.astype("int8")
    data_flagged.name = name
    data_flagged.attrs["long_name"] = "flag wave breaking"

    return data_flagged

@check_argument_types(["data", "events"], [xr.DataArray, gpd.GeoDataFrame])
@check_empty_dataframes
@get_dimension_attributes("data")
def clim_xarray(data, events, flag="ones", name="flag", *args, **kwargs):
    """
    Create xarray.DataArray from events stored in a geopandas.GeoDataFrame.
    Grid cells where an event is present are flagged with the value 1.
    Dimension names ("time_name", "lon_name", "lat_name"), size ("ntime", "nlon", "nlat")
    and resolution ("dlon", "dlat") can be passed as key=value argument.

    Parameters
    ----------
        data : xarray.DataArray
            data used for the index calculation
        events : geopandas.GeoDataFrame
            GeoDataFrame with the date and geometry for each event
        flag : string, optional
            column name of the events geopandas.GeoDataFrame
            flag is set where an event is present
            default value is "ones"
        name : string, optional
            name of the xarray variable that is created

    Returns
    -------
        flag: xarray.DataArray
            Data with events flagged with the value 1
    """

    # Generate 2D mesh grid and point geometries
    data = data.isel(time=0).copy()
    lats = data[kwargs["lat_name"]].values
    lons = data[kwargs["lon_name"]].values
    lon2d, lat2d = np.meshgrid(lons, lats)
    flat_points = np.column_stack([lon2d.ravel(), lat2d.ravel()])

    # Convert to GeoDataFrame once
    grid_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(flat_points[:, 0], flat_points[:, 1]),
        crs=events.crs,
    )

    # Buffer geometries to approximate cell influence area
    buffered = events.copy()
    #buffer_dist = 0.5 * (kwargs["dlon"] + dlon)
    #buffered["geometry"] = buffered.geometry.buffer(buffer_dist)

    # Spatial join (fast spatial lookup)
    joined = gpd.sjoin(grid_points, buffered, predicate="within", how="inner")

    # Extract 2D grid indices
    row_idx, col_idx = np.unravel_index(joined.index.values, lon2d.shape)

    # Assign value 1 to each hit
    values = np.ones(len(row_idx), dtype="int32")

    if flag == "sum":
        # If multiple geometries (from `index_right`) hit the same grid point,
        # count them all — i.e., how many unique overlaps per grid point.
        # So no change in values; same as "ones", just semantically different
        pass
    elif flag != "ones":
        raise ValueError("flag must be either 'ones' or 'sum'")

    # Accumulate counts
    result_array = np.zeros_like(lon2d, dtype="int32")
    np.add.at(result_array, (row_idx, col_idx), values)

    # Convert to xarray
    result = xr.DataArray(
        result_array,
        coords={kwargs["lat_name"]: lats, kwargs["lon_name"]: lons},
        dims=(kwargs["lat_name"], kwargs["lon_name"]),
        name=name,
    )
    result.attrs["long_name"] = f"{flag} of overlapping events"

    return result

@check_argument_types(["events"], [pd.DataFrame])
@check_empty_dataframes
def track_events(
    events, time_range=None, method="by_overlap", buffer=0, overlap=0, distance=1000,
    overlap_by="union"
):
    """
    Temporal tracking of events.
    Events receive the same label if they spatially overlap at step t
    and t + time_range.

    Parameters
    ----------
        events : geopandas.GeoDataFrame
            GeoDataFrame with the date and coordinates of each identified event
        time_range: int or float, optional
            Time range for temporally tracking the events. The units of
            time_range is hours if the type of the time dimension is np.datetime64.
            If not specified, the smallest time difference larger than zero is used.
        method : {"by_overlap", "by_distance"}, optional
            Method for temporally tracking the events:
                * "by_overlap": Events receive the same label if they spatially
                    overlap at step t and t + time_range.
                * "by_distance": Events receive the same label if their centre of mass
                    is closer than "distance"
        buffer : float, optional
            buffer around event polygon in degrees for the 'by_overlap' method
        overlap : float, optional
            minimum percentage of overlapping for the 'by_overlap' method
        distance : int or float, optional
            maximum distance in km between two events for the 'by_distance' method


    Returns
    -------
        events: geopandas.GeoDataFrame
            GeoDataFrame with label column showing the temporal coherence
    """

    # reset index of events
    events = events.reset_index(drop=True)

    # detect time range
    if time_range is None:
        date_dif = events.date.diff()
        time_range = date_dif[date_dif > pd.Timedelta(0)].min().total_seconds() / 3600

    # select events that are in range of time_range
    #def get_range_combinations(events, index):
    #    """
    #    find events within the next steps that are in time range
    #    """
    #    if events.date.dtype == np.dtype("datetime64[ns]"):
    #        diffs = (events.date - events.date.iloc[index]).dt.total_seconds() / 3600
    #    else:
    #        diffs = abs(events.date - events.date.iloc[index])
    #
    #    diffs = diffs/pd.Timedelta('1 hour') ## Added assuming the dates are always timestaps
    #
    #    check = (diffs > 0) & (diffs <= time_range)
    #
    #    return [(index, close) for close in events[check].index]
    #
    #range_comb = np.asarray(
    #    list(
    #        set(
    #            itertools.chain.from_iterable(
    #                [get_range_combinations(events, index) for index in tqdm(events.index, desc="Finding range combinations")]
    #            )
    #        )
    #    )
    #)

    def get_range_combinations_optimized(events, time_range_hours):
        """
        Efficiently find all index pairs (i, j) where event j occurs within
        `time_range_hours` after event i.
        """
        # Ensure datetime and sorted
        dates = pd.to_datetime(events.date).sort_values()
        indices = dates.index.to_numpy()
        timestamps = dates.values.astype('datetime64[ns]')
        delta = np.timedelta64(int(time_range_hours * 3600), 's')
        
        pairs = []
        
        for i, t_start in tqdm(enumerate(timestamps),total=len(timestamps)):
            # Find index of the first timestamp outside the time window
            end_idx = np.searchsorted(timestamps, t_start + delta, side='right')
            #start_idx = i + 1  # skip self and earlier times
            start_idx = np.searchsorted(timestamps, t_start + delta, side='left')
            if end_idx > start_idx:
                pairs.extend((indices[i], indices[j]) for j in range(start_idx, end_idx))
    
        return np.array(pairs)
    
    # Usage
    range_comb = get_range_combinations_optimized(events, time_range_hours=time_range)

    if len(range_comb) == 0:
        errmsg = "No events detected in the time range: {}".format(time_range)
        raise ValueError(errmsg)

    if method == "by_distance":
        # get centre of mass
        com1 = np.asarray(list(events.loc[range_comb[:, 0]].com))
        com2 = np.asarray(list(events.loc[range_comb[:, 1]].com))

        # calculate distance between coms
        dist_com = np.asarray(
            [dist.pairwise(np.radians([p1, p2]))[0, 1] for p1, p2 in zip(com1, com2)]
        )

        # check which coms are in range of 'distance'
        check_com = dist_com * 6371 < distance

        # select combinations
        combine = range_comb[check_com]

    elif method == "by_overlap":
        print('Overlaps')
        # select geometries that are in time range and add buffer
        #geom1 = events.iloc[range_comb[:, 0]].geometry.buffer(buffer).make_valid()
        #geom2 = events.iloc[range_comb[:, 1]].geometry.buffer(buffer).make_valid()

        # calculate and check the percentage of overlap
        #inter = geom1.intersection(geom2, align=False)
        #check_overlap = (
        #    inter.area.values
        #    / (geom2.area.values + geom1.area.values - inter.area.values)
        #    > overlap
        #)
        #
        ## select combinations
        #combine = range_comb[check_overlap]

        #from shapely.geometry import Polygon
        #import geopandas as gpd
        # 
        def check_overlaps_in_chunks(events, range_comb, buffer, overlap,  
                                     chunk_size=10000,overlap_by="union"):
            """
            Check overlap between buffered geometries for large `range_comb` efficiently in chunks.
            Returns the filtered combinations that pass the overlap threshold.
            """
            valid_combos = []
        
            for i in tqdm(range(0, len(range_comb), chunk_size), desc="Checking overlaps"):
                chunk = range_comb[i:i + chunk_size]
                
                # Get unique indices to reduce redundant geometry access
                idx1 = chunk[:, 0]
                idx2 = chunk[:, 1]
        
                # Extract geometries and buffer (only once per side)
                geom1 = events.geometry.loc[idx1].buffer(buffer).make_valid().reset_index(drop=True)
                geom2 = events.geometry.loc[idx2].buffer(buffer).make_valid().reset_index(drop=True)
        
                # Compute intersections
                inter = geom1.intersection(geom2, align=False)

                if overlap_by=="union":
                    union_area = geom1.area.values + geom2.area.values - inter.area.values
                    # Handle zero-area edge case to avoid division by zero
                    with np.errstate(divide='ignore', invalid='ignore'):
                        iou = inter.area.values / union_area
                        iou[np.isnan(iou)] = 0
                elif overlap_by=="next":
                    # Handle zero-area edge case to avoid division by zero
                    with np.errstate(divide='ignore', invalid='ignore'):
                        iou = inter.area.values / geom2.area.values
                        iou[np.isnan(iou)] = 0
                elif overlap_by=="smallest":
                    small_area=[min(a, b) for a, b in zip(geom1.area.values, geom2.area.values)]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        iou = inter.area.values / small_area
                        iou[np.isnan(iou)] = 0
                else:
                    raise TypeError(f"Overlap_by command not a valid type: {overlap_by} given")    
        
                # Filter combinations by IoU threshold
                passed = iou > overlap
        
                valid_combos.extend(chunk[passed])
        
            return np.array(valid_combos)

        combine=check_overlaps_in_chunks(events, range_comb, buffer, overlap, 
                                         chunk_size=100000,overlap_by=overlap_by)

    else:
        errmsg = "'{}' not supported as method!".format(method)
        hint = " Supported methods are 'by_overlap' and 'by_distance'"
        raise ValueError(errmsg + hint)

    # combine tracked indices to groups
    combine = index_utils.combine_shared(combine)

    # initiate label column
    events["label"] = events.index

    # assign labels to the events
    for item in tqdm(combine, desc="Assign labels"):
        events.loc[item, "label"] = min(item)

    # select smallest possible index number for all events
    #label = events.label.copy()
    #for i in tqdm(np.arange(len(set(events.label))), desc="Select smallest indices"):
    #    label[events.label == sorted(set(events.label))[i]] = i
    #events.label = label
    print('Smallest label....')
    events["label"] = pd.factorize(events["label"])[0]

    # sort list by label and date and return geopandas.GeoDataFrame
    return events.sort_values(by=["label", "date"])

def serial_clustering(
    events, time_range_min=None, time_range_max=None, 
    method="by_overlap", buffer=0, overlap=0, distance=1000
):
    """
    Temporal tracking of events.
    Events receive the same label if they spatially overlap at step t
    and t + time_range.

    Parameters
    ----------
        events : geopandas.GeoDataFrame
            GeoDataFrame with the date and coordinates of each identified event
        time_range: int or float, optional
            Time range for temporally tracking the events. The units of
            time_range is hours if the type of the time dimension is np.datetime64.
            If not specified, the smallest time difference larger than zero is used.
        method : {"by_overlap", "by_distance"}, optional
            Method for temporally tracking the events:
                * "by_overlap": Events receive the same label if they spatially
                    overlap at step t and t + time_range.
                * "by_distance": Events receive the same label if their centre of mass
                    is closer than "distance"
        buffer : float, optional
            buffer around event polygon in degrees for the 'by_overlap' method
        overlap : float, optional
            minimum percentage of overlapping for the 'by_overlap' method
        distance : int or float, optional
            maximum distance in km between two events for the 'by_distance' method


    Returns
    -------
        events: geopandas.GeoDataFrame
            GeoDataFrame with label column showing the temporal coherence
    """

    # reset index of events
    events = events.reset_index(drop=True)
    
    # select events that are in range of time_range
    def get_range_combinations(events, index):
        """
        find events within the next steps that are in time range
        """
        if events.date.dtype == np.dtype("datetime64[ns]"):
            diffs = (events.date - events.date.iloc[index]).dt.total_seconds() / 3600
        else:
            diffs = abs(events.date - events.date.iloc[index])

        diffs = diffs/pd.Timedelta('1 hour') ## Added assuming the dates are always timestaps

        check = (diffs >= time_range_min) & (diffs <= time_range_max)

        return [(index, close) for close in events[check].index]
    
    range_comb = np.asarray(
        list(
            set(
                itertools.chain.from_iterable(
                    [get_range_combinations(events, index) for index in tqdm(events.index, desc="Finding range combinations")]
                )
            )
        )
    )

    if len(range_comb) == 0:
        errmsg = "No events detected in the time range: {}".format(time_range)
        raise ValueError(errmsg)

    if method == "by_overlap":
        print('Overlaps')
        # select geometries that are in time range and add buffer
        geom1 = events.iloc[range_comb[:, 0]].geometry.buffer(buffer).make_valid()
        geom2 = events.iloc[range_comb[:, 1]].geometry.buffer(buffer).make_valid()

        # calculate and check the percentage of overlap
        inter = geom1.intersection(geom2, align=False)
        check_overlap = (
            inter.area.values
            / (geom2.area.values + geom1.area.values - inter.area.values)
            > overlap
        )

        # select combinations
        combine = range_comb[check_overlap]

    else:
        errmsg = "'{}' not supported as method!".format(method)
        hint = " Supported methods are 'by_overlap' and 'by_distance'"
        raise ValueError(errmsg + hint)

    # combine tracked indices to groups
    combine = index_utils.combine_shared(combine)

    # initiate serial_cluster column
    events["serial_cluster"] = events.index
    
    # assign serial_cluster to the events
    for item in tqdm(combine, desc="Assign serial clusters"):
        events.loc[item, "serial_cluster"] = min(item)

    # select smallest possible index number for all events
    serial_cluster = events.serial_cluster.copy()
    for i in tqdm(np.arange(len(set(events.serial_cluster))), desc="Select smallest indices"):
        serial_cluster[events.serial_cluster == sorted(set(events.serial_cluster))[i]] = i
    events.serial_cluster = serial_cluster

    # sort list by serial_cluster and date and return geopandas.GeoDataFrame
    return events.sort_values(by=["label", "date"])

def combine_overturnings_by_overlap(events, buffer=0, overlap=0, overlap_by="union"):
    
    # reset index of events
    events = events.reset_index(drop=True)
    
    # select events that are in range of time_range
    def get_range_combinations(events, index):
        """
        find events within the next steps that are in time range
        """
        if events.date.dtype == np.dtype("datetime64[ns]"):
            diffs = (events.date - events.date.iloc[index]).dt.total_seconds() / 3600
        else:
            diffs = abs(events.date - events.date.iloc[index])
    
        diffs = diffs/pd.Timedelta('1 hour') ## Added assuming the dates are always timestaps
    
        check = diffs ==0
    
        return [(index, close) for close in events[check].index]
    
    range_comb = np.asarray(
        list(
            set(
                itertools.chain.from_iterable(
                    [get_range_combinations(events, index) for index in tqdm(events.index, desc="Finding range combinations")]
                )
            )
        )
    )
    
    def check_overlaps_in_chunks(events, range_comb, buffer, overlap, chunk_size=10000,overlap_by="union"):
        """
        Check overlap between buffered geometries for large `range_comb` efficiently in chunks.
        Returns the filtered combinations that pass the overlap threshold.
        """
        valid_combos = []
    
        for i in tqdm(range(0, len(range_comb), chunk_size), desc="Checking overlaps"):
            chunk = range_comb[i:i + chunk_size]
            
            # Get unique indices to reduce redundant geometry access
            idx1 = chunk[:, 0]
            idx2 = chunk[:, 1]
    
            # Extract geometries and buffer (only once per side)
            geom1 = events.geometry.iloc[idx1].buffer(buffer).make_valid().reset_index(drop=True)
            geom2 = events.geometry.iloc[idx2].buffer(buffer).make_valid().reset_index(drop=True)
    
            # Compute intersections
            inter = geom1.intersection(geom2, align=False)

            if overlap_by=="union":
                union_area = geom1.area.values + geom2.area.values - inter.area.values
                # Handle zero-area edge case to avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    iou = inter.area.values / union_area
                    iou[np.isnan(iou)] = 0
            elif overlap_by=="next":
                # Handle zero-area edge case to avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    iou = inter.area.values / geom2.area.values
                    iou[np.isnan(iou)] = 0
            elif overlap_by=="smallest":
                small_area=[min(a, b) for a, b in zip(geom1.area.values, geom2.area.values)]
                with np.errstate(divide='ignore', invalid='ignore'):
                    iou = inter.area.values / small_area
                    iou[np.isnan(iou)] = 0
            else:
                raise TypeError(f"Overlap_by command not a valid type: {overlap_by} given")    
    
            # Filter combinations by IoU threshold
            passed = iou > overlap
    
            valid_combos.extend(chunk[passed])
    
        return np.array(valid_combos)

    combine=check_overlaps_in_chunks(events, range_comb, buffer, overlap, chunk_size=100000,overlap_by=overlap_by)
    
    # combine tracked indices to groups
    combine = index_utils.combine_shared(combine)
    
    # initiate serial_cluster column
    events["event_cluster"] = events.index
    
    # assign serial_cluster to the events
    for item in tqdm(combine, desc="Assign event clusters"):
        events.loc[item, "event_cluster"] = min(item)
    
    # select smallest possible index number for all events
    event_cluster = events.event_cluster.copy()
    for i in tqdm(np.arange(len(set(events.event_cluster))), desc="Select smallest indices"):
        event_cluster[events.event_cluster == sorted(set(events.event_cluster))[i]] = i
    events.event_cluster = event_cluster
    
    def multipoly_to_poly360(geom):
        def wrap_geometry_to_360(geom):
            """Wrap geometry to 0–360 grid ONLY if both x-bounds are < 0."""
            minx, miny, maxx, maxy = geom.bounds
            should_shift = (minx < 0) and (maxx < 0)
        
            if geom.geom_type == 'Polygon':
                if should_shift:
                    exterior = shift_coords_to_360(geom.exterior.coords)
                    interiors = [shift_coords_to_360(ring.coords) for ring in geom.interiors]
                    return Polygon(exterior, interiors)
                else:
                    return geom
        
            elif geom.geom_type == 'MultiPolygon':
                return MultiPolygon([wrap_geometry_to_360(p) for p in geom.geoms])
        
            else:
                raise TypeError(f"Unsupported geometry type: {geom.geom_type}")
        
        def merge_across_dateline(geom):
            """
            If input is MultiPolygon due to dateline crossing,
            project to 0–360 grid and re-union to form single polygon.
            """
            wrapped = wrap_geometry_to_360(geom)
            if isinstance(wrapped, MultiPolygon):
                # Attempt union in 0–360 space
                merged = unary_union(wrapped)
                return merged
            else:
                return wrapped
        
        # Example usage on your combined_geometry
        merged_360 = merge_across_dateline(geom)
    
        return merged_360
    
    def shift_coords_to_360(coords):
        return [(lon + 360 if lon < 0 else lon, lat) for lon, lat in coords]
    
    def round_to_resolution(value, resolution=0.25):
        return round(value / resolution) * resolution
    
    def shift_coords_to_180(coords):
        return [(lon - 360 if lon >= 180 else lon, lat) for lon, lat in coords]
    
    # Apply to GeoDataFrame `gdf`
    def compute_bounds_row(row):
        geom = row.geometry
        if isinstance(geom, Polygon):
            bounds = geom.bounds
        elif isinstance(geom, MultiPolygon):
            bounds = multipoly_to_poly360(geom).bounds
        else:
            return pd.Series({'minX': None, 'maxX': None})  # Or raise an error
    
        # Shift back to [-180, 180] if necessary
        (minX, minY), (maxX, maxY) = shift_coords_to_180([(bounds[0], bounds[1]), (bounds[2], bounds[3])])
        return pd.Series({'minX': minX, 'maxX': maxX})
    
    # Apply to DataFrame
    events[['west_lon', 'east_lon']] = events.apply(compute_bounds_row, axis=1)
    
    events=events.groupby(['event_cluster']).agg(pd.Series.tolist)
    
    def union_geometries_from_grouped(row):
        return gpd.GeoSeries(row.geometry, crs="EPSG:4326").unary_union
    
    events['geometry']=events.apply(union_geometries_from_grouped,axis=1)
    
    def get_com_bounds_from_union_geometries(row):
        combined_geometry = row.geometry
        #print(row.date)#,combined_geometry,combined_geometry.type)
        if isinstance(combined_geometry, Polygon):
            bounds=combined_geometry.bounds
        elif isinstance(combined_geometry, MultiPolygon):
            bounds=multipoly_to_poly360(combined_geometry).bounds
        else:
            raise TypeError("Expected a Polygon or MultiPolygon")
        
        minX,minY=shift_coords_to_180([(bounds[0],bounds[1])])[0]
        maxX,maxY=shift_coords_to_180([(bounds[2],bounds[3])])[0]
        
        comX=round_to_resolution((bounds[0]+bounds[2])/2)
        comY=round_to_resolution((bounds[1]+bounds[3])/2)
        comX,comY=shift_coords_to_180([(comX,comY)])[0]
    
        return pd.Series({'comX': comX, 'comY': comY,
                          'minX': minX, 'minY': minY,
                          'maxX': maxX, 'maxY': maxY})
    
    events[['comX','comY','minX','minY','maxX','maxY']]=events.apply(get_com_bounds_from_union_geometries,axis=1)
    
    events['date'] = events['date'].apply(lambda x: x[0] if x else None)
    
    events['west_lat'] = events.apply(
        lambda row: row['west_lat'][row['west_lon'].index(row['minX'])] if row['minX'] in row['west_lon'] else None,
        axis=1)
    events['east_lat'] = events.apply(
        lambda row: row['east_lat'][row['east_lon'].index(row['maxX'])] if row['maxX'] in row['east_lon'] else None,
        axis=1)
    
    # check if event is cyclonic by orientation by comparing east_lat and west_lat
    def check_orientation(row):
        lat_west=row.west_lat
        lat_east=row.east_lat
        if abs(lat_west) <= abs(lat_east):
            #return "cyclonic"
            return pd.Series({'orientation': "cyclonic"})
        else:
            #return "anticyclonic"
            return pd.Series({'orientation': "anticyclonic"})
    
    events['orientation']=events.apply(check_orientation,axis=1)
    events=events.reset_index()
    events=events.drop(['event_cluster','west_lon','east_lon'],axis=1)
    
    return events

