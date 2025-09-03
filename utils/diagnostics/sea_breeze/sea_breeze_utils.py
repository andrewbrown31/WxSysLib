import numpy as np
import dask.array as da
import xarray as xr
import pyproj
import scipy
import pandas as pd
import warnings
import datetime as dt
from skimage.segmentation import find_boundaries
from dask.distributed import progress

def vert_mean_wind(wind_ds,mean_heights,vert_coord):
    
    """
    Calculate the vertical mean of u and v wind components over a specified layer.

    Parameters
    ----------
    wind_ds : xarray.Dataset
        Dataset containing 'u' and 'v' wind components.
    mean_heights : tuple or list of float
        The lower and upper bounds of the vertical layer over which to compute the mean.
    vert_coord : str
        The name of the vertical coordinate in the dataset (e.g., 'height', 'level').

    Returns
    -------
    u_mean : xarray.DataArray
        The vertical mean of the 'u' wind component over the specified layer.
    v_mean : xarray.DataArray
        The vertical mean of the 'v' wind component over the specified layer.
    """

    u_mean = wind_ds["u"].sel({vert_coord:slice(mean_heights[0],mean_heights[1])}).mean(vert_coord)
    v_mean = wind_ds["v"].sel({vert_coord:slice(mean_heights[0],mean_heights[1])}).mean(vert_coord)

    return u_mean, v_mean

def daily_mean_wind(wind_ds):

    """
    Calculates the rolling daily mean of u and v wind components from an xarray dataset.

    Parameters
    ----------
    wind_ds : xarray.Dataset
        An xarray dataset containing 'u' and 'v' wind components with a 'time' dimension.

    Returns
    -------
    u_mean : xarray.DataArray
        The rolling daily mean of the 'u' wind component.
    v_mean : xarray.DataArray
        The rolling daily mean of the 'v' wind component.

    Notes
    -----
    The function determines the time step from the dataset and computes the window size for a 24-hour rolling mean.
    The rolling mean is centered and requires at least half the window size of valid data points.
    """

    dt_h = np.round((wind_ds.time.diff("time")[0].values / (1e9 * 60 * 60)).astype(float)).astype(int)
    time_window = int(24 / dt_h)
    min_periods = int(time_window/2)

    u_mean = wind_ds["u"].rolling(dim={"time":time_window},center=True,min_periods=min_periods).mean()
    v_mean = wind_ds["v"].rolling(dim={"time":time_window},center=True,min_periods=min_periods).mean()

    return u_mean, v_mean

def percentile(field,p=99.5,skipna=False):

    """
    Calculate the pth percentile of an xarray DataArray field over all dimensions.

    Parameters
    ----------
    field : xarray.DataArray
        Input data array.
    p : float, optional
        Percentile to compute (default is 99.5).
    skipna : bool, optional
        If True, skip NaN values using climtas (default is False).

    Returns
    -------
    percentile : dask.array.Array
        The computed percentile value.

    Notes
    -----
    Uses dask for computation by default. If skipna is True, uses climtas for NaN-safe calculation.
    """
    
    if skipna:
        import climtas
        #Re-shape field for percentile calculation.
        field_stacked = field.stack(z=list(field.dims))
        return da.array(climtas.blocked.approx_percentile(field_stacked, dim="z", q=p, skipna=True))
    else:
        field_stacked = da.array(field).flatten()
        return da.percentile(field_stacked,p,internal_method="tdigest")

def rotate_wind(u,v,theta):

    """
    Rotate u and v wind components to cross-shore and along-shore directions based on coastline orientation.

    Parameters
    ----------
    u : xarray.DataArray
        U-component of wind (east-west) in m/s.
    v : xarray.DataArray
        V-component of wind (north-south) in m/s.
    theta : xarray.DataArray
        Coastline orientation angles from North, in degrees.

    Returns
    -------
    uprime : xarray.DataArray
        Wind component parallel to the coast (along-shore).
    vprime : xarray.DataArray
        Wind component perpendicular to the coast (cross-shore).
    """

    #Rotate angle to be perpendicular to theta, from E (i.e. mathamatical angle definition)
    rotated_angle=(((theta)%360-90)%360) + 90   
    
    #Define normal angle vectors, pointing onshore
    cx, cy = [-np.cos(np.deg2rad(rotated_angle)), np.sin(np.deg2rad(rotated_angle))]
    
    #Define normal angle vectors, pointing alongshore
    ax, ay = [-np.cos(np.deg2rad(rotated_angle - 90)), np.sin(np.deg2rad(rotated_angle - 90))]    
    
    #Calculate the wind component perpendicular and parallel to the coast by using the normal unit vectors
    uprime = ((u*ax) + (v*ay))
    vprime = ((u*cx) + (v*cy))

    return uprime, vprime            

def get_coastline_angle(lsm=None,R=20,latlon_chunk_size=10,compute=True,path_to_load=None,save=False,path_to_save=None,lat_slice=None,lon_slice=None,smooth=False,sigma=4):
    """
    If compute is True, calculate the dominant coastline angle for each point in the domain based on a land-sea mask.

    Otherwise just loads the angles from disk.

    If computing, constructs a "kernel" for each point based on the angle between that point and coastline points, then takes a weighted average. The weighting function can be customised, but is by default an inverse parabola to distance R, then decreases by distance**4. The weights are set to zero at a distance of 10,000 km, and are undefined at the coast (where linear interpolation is done to fill in the coastline gaps).

    Parameters
    ----------
    lsm : xarray.DataArray, optional
        Binary land-sea mask with latitude ("lat") and longitude ("lon") information.
    R : int, default=20
        The distance (in km) at which the weighting function is changed from 1/p to 1/q. Around 2 times the grid spacing of the lsm seems appropriate based on initial tests.
    latlon_chunk_size : int, default=10
        The size of the chunks over the latitude/longitude dimension for computation.
    compute : bool, default=True
        Whether to compute the angles or load from disk.
    path_to_load : str, optional
        File path to previous output that can be loaded if compute is False.
    save : bool, default=False
        Whether to save the computed angles output if compute is True.
    path_to_save : str, optional
        File path to save output if save is True.
    lat_slice : slice or array-like, optional
        Latitude indices or values to slice when loading angles from disk.
    lon_slice : slice or array-like, optional
        Longitude indices or values to slice when loading angles from disk.
    smooth : bool, default=False
        Whether to smooth the interpolated angles output using a Gaussian filter.
    sigma : float, default=4
        Sigma value for the Gaussian filter if smoothing.

    Returns
    -------
    xarray.Dataset
        Dataset containing arrays of coastline angles (0-360 degrees from North), as well as an array of angle variance as an estimate of uncertainty. Includes additional fields for coastline mask and minimum distance to the coast.

    Notes
    -----
    Thank you to Ewan Short and Jarrah Harrison-Lofthouse for help developing this method.
    """

    if save:
        if path_to_save is None:
            raise AttributeError("Saving but no path speficfied")
        
    if compute:

        assert np.in1d([0,1],np.unique(lsm)).all(), "Land-sea mask must be binary"
        
        warnings.simplefilter("ignore")

        #From the land sea mask define the coastline and a label array
        coast_label = find_boundaries(lsm)*1
        land_label = lsm.values

        #Get lat lon info for domain and coastline, and convert to lower precision
        lon = lsm.lon.values
        lat = lsm.lat.values
        xx,yy = np.meshgrid(lon,lat)
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)    

        #Define coastline x,y indices from the coastline mask
        xl, yl = np.where(coast_label)

        #Get coastline lat lon vectors
        yy_t = np.array([yy[xl[t],yl[t]] for t in np.arange(len(yl))])
        xx_t = np.array([xx[xl[t],yl[t]] for t in np.arange(len(xl))])

        #Repeat the 2d lat lon array over a third dimension (corresponding to the coast dim). Also repeat the yy_t and xx_t vectors over the spatial arrays
        yy_rep = da.moveaxis(da.stack([da.from_array(yy)]*yl.shape[0],axis=0),0,-1).rechunk({0:-1,1:-1,2:latlon_chunk_size})
        xx_rep = da.moveaxis(da.stack([da.from_array(xx)]*xl.shape[0],axis=0),0,-1).rechunk({0:-1,1:-1,2:latlon_chunk_size})
        xx_t_rep = (xx_rep * 0) + xx_t
        yy_t_rep = (yy_rep * 0) + yy_t

        #Calculate the distance and angle between coastal points and all other points using pyproj, then convert to complex space.
        geod = pyproj.Geod(ellps="WGS84")
        def calc_dist(lon1,lat1,lon2,lat2):
            fa,_,d = geod.inv(lon1,lat1,lon2,lat2)
            return d/1e3 * np.exp(1j * np.deg2rad(fa))
        
        stack = da.map_blocks(
                    calc_dist,
                    xx_t_rep,
                    yy_t_rep,
                    xx_rep,
                    yy_rep,
                    dtype=np.complex64,
                    meta=np.array((), dtype=np.complex64))
        del xx_t_rep, yy_t_rep, yy_rep, xx_rep
        
        #Move axes around for convenience later
        stack = da.moveaxis(stack, -1, 0)

        #Get back distance by taking absolute value
        stack_abs = da.abs(stack,dtype=np.float32)
        
        #Create an inverse distance weighting function
        weights = get_weights(stack_abs, p=4, q=2, R=R, slope=-1)

        #Take the weighted mean and convert complex numbers to an angle and magnitude
        print("INFO: Take the weighted mean and convert complex numbers to an angle and magnitude...")
        mean_angles = da.mean((weights*stack), axis=0).persist()
        progress(mean_angles)
        mean_abs = da.abs(mean_angles)
        mean_angles = da.angle(mean_angles)    

        #Flip the angles inside the coastline for convention, and convert range to 0 to 2*pi
        mean_angles = da.where(land_label==1,(mean_angles+np.pi) % (2*np.pi),mean_angles % (2*np.pi))

        #Calculate the weighted circular variance
        print("INFO: Calculating the sum of the weights...")
        total_weight = da.sum(weights, axis=0).persist()
        progress(total_weight)
        print("INFO: Calculating variance...")
        variance = (1 - da.abs(da.sum( (weights/total_weight) * (stack / stack_abs), axis=0))).persist()
        progress(variance)
        del stack, weights, total_weight 

        #Calculate minimum distance to the coast
        print("INFO: Calculating minimum distance to the coast...")
        min_coast_dist = stack_abs.min(axis=0).persist()

        #Convert angles to degrees, and from bearing to orientation of coastline.
        #Also create an xr dataarray object
        angle_da = xr.DataArray(da.rad2deg(mean_angles) - 90,coords={"lat":lat,"lon":lon})
        angle_da = xr.where(angle_da < 0, angle_da+360, angle_da)      

        #Convert variance and coast arrays to xr dataarrays
        var_da = xr.DataArray(variance,coords={"lat":lat,"lon":lon})
        coast = xr.DataArray(coast_label,coords={"lat":lat,"lon":lon})
        mean_abs = xr.DataArray(mean_abs,coords={"lat":lat,"lon":lon})
        mean_angles = xr.DataArray(mean_angles,coords={"lat":lat,"lon":lon})
        min_coast_dist = xr.DataArray(min_coast_dist,coords={"lat":lat,"lon":lon})

        #Create an xarray dataset
        angle_ds =  xr.Dataset({
            "angle":angle_da,
            "variance":var_da,
            "coast":coast,
            "mean_abs":mean_abs,
            "mean_angles":mean_angles,
            "min_coast_dist":min_coast_dist})

        #Do the interpolation across the coastline
        angle_ds = interpolate_angles(angle_ds)
        angle_ds = interpolate_variance(angle_ds)

        #Attributes
        angle_ds["angle"] = angle_ds["angle"].assign_attrs(
            units = "degrees",
            long_name = "Angle of coastline orientation",
            description = "The angle of dominant coastline orientation in degrees from North. Points with a dominant north-south coastline with ocean to the east will have an angle of 0 degrees. The dominant coastline for each point is determined by the weighted mean of the angles between that point and all coastline points in the domain. The weighting function is an inverse parabola to distance R, then decreases by distance**4. The weights are set to zero at a distance of 2000 km, and are undefined at the coast."
            )
        
        angle_ds["variance"] = angle_ds["variance"].assign_attrs(
            units = "[0,1]",
            long_name = "Variance of coastline angles",
            description = "For each point, the variance of the coastline angles in the domain. This is a measure of how many coastlines are influencing a given point. A value of 0 indicates that coastlines are generally in agreement, and a value of 1 indicates that the point is influenced by coastlines in all directions."
            )
        
        angle_ds["coast"] = angle_ds["coast"].assign_attrs(
            units = "[0,1]",
            long_name = "Coastline mask",
            description = "A binary mask of the coastline determined from the land-sea mask. 1 indicates a coastline point, and 0 indicates a non-coastline point."
            )
        
        angle_ds["min_coast_dist"] = angle_ds["min_coast_dist"].assign_attrs(
            units = "km",
            long_name = "Minimum distance to the coast",
            description = "The minimum distance to the coast for each point in the domain."
            )
        
        angle_ds["angle_interp"] = angle_ds["angle_interp"].assign_attrs(
            units = "degrees",
            long_name = "Interpolated coastline orientation",
            description = "The angle of dominant coastline orientation in degrees from North. Points with a dominant north-south coastline with ocean to the east will have an angle of 0 degrees. The dominant coastline for each point is determined by the weighted mean of the angles between that point and all coastline points in the domain. The weighting function is an inverse parabola to distance R, then decreases by distance**4. The weights are set to zero at a distance of 2000 km, and are undefined at the coast, where linear interpolation is then done."
            )
        
        angle_ds["variance_interp"] = angle_ds["variance_interp"].assign_attrs(
            units = "[0,1]",
            long_name = "Interpolated variance of coastline angles",
            description = "For each point, the variance of the coastline angles in the domain. This is a measure of how many coastlines are influencing a given point. A value of 0 indicates that coastlines are generally in agreement, and a value of 1 indicates that the point is influenced by coastlines in all directions. The variance is undefined at the coast, and here the variance is interpolated across the coastline."
            )
        
        angle_ds = angle_ds.assign_attrs(
            description = "Dataset of coastline angles and variance",
            acknowledgmements = "This method was developed with help from Ewan Short and Jarrah Harrison-Lofthouse.",
            R_km = str(R)
            )

    else:

        if path_to_load is not None:
            angle_ds = xr.open_dataset(path_to_load)
            if lat_slice is not None:
                angle_ds = angle_ds.sel(lat=lat_slice)
            if lon_slice is not None:
                angle_ds = angle_ds.sel(lon=lon_slice)
            save = False
        else:
            raise AttributeError("If not computing the angles, path_to_load needs to be specified")

    if save:
        angle_ds.to_netcdf(path_to_save)

    if smooth:
        angle_ds["angle_interp"] = smooth_angles(angle_ds["angle_interp"],sigma)

    return angle_ds    

def interpolate_angles(angle_ds):

    """
    From a dataset of coastline angles, interpolate across the coastline.

    This is used because the result of get_coastline_angle() is not defined along the coastline.
    """

    # Create a meshgrid for the latitude and longitude
    xx,yy = np.meshgrid(angle_ds.lon,angle_ds.lat)

    # Interpolate the angles using scipy's griddata function. Requires conversion to complex space to account for circular nature of angles.
    mean_complex = angle_ds.mean_abs * da.exp(1j*angle_ds.mean_angles)
    points = mean_complex.values.ravel()
    valid = ~np.isnan(points)
    points_valid = points[valid]
    xx_rav, yy_rav = xx.ravel(), yy.ravel()
    xxv = xx_rav[valid]
    yyv = yy_rav[valid]
    interpolated_angles = scipy.interpolate.griddata(np.stack([xxv, yyv]).T, points_valid, (xx, yy), method="linear").reshape(xx.shape) 

    # Convert back to angle in degrees from North
    interpolated_angles = da.rad2deg(da.angle(interpolated_angles))
    interpolated_angle_da = xr.DataArray(interpolated_angles - 90,coords={"lat":angle_ds.lat,"lon":angle_ds.lon})
    interpolated_angle_da = xr.where(interpolated_angle_da < 0, interpolated_angle_da+360, interpolated_angle_da)  

    # Drop un-interpolated variables from the dataset and add the interpolated angle array
    angle_ds = angle_ds.drop_vars(["mean_abs","mean_angles"])
    angle_ds["angle_interp"] = interpolated_angle_da

    return angle_ds

def interpolate_variance(angle_ds):

    """
    From a dataset of coastline variance, interpolate across the coastline.

    This is used because the result of get_coastline_angle() is not defined along the coastline.
    """

    # Create a meshgrid for the latitude and longitude
    xx,yy = np.meshgrid(angle_ds.lon,angle_ds.lat)

    # Interpolate the variance using scipy's griddata function. Requires conversion to complex space to account for circular nature of angles.
    points = angle_ds.variance.values.ravel()
    valid = ~np.isnan(points)
    points_valid = points[valid]
    xx_rav, yy_rav = xx.ravel(), yy.ravel()
    xxv = xx_rav[valid]
    yyv = yy_rav[valid]
    interpolated_variance = scipy.interpolate.griddata(np.stack([xxv, yyv]).T, points_valid, (xx, yy), method="linear").reshape(xx.shape)     
    interpolated_variance_da = xr.DataArray(interpolated_variance,dims=angle_ds.dims,coords=angle_ds.coords)

    # Add the interpolated variance array to the dataset
    angle_ds["variance_interp"] = interpolated_variance_da

    return angle_ds    

def smooth_angles(angles,sigma):
    """
    Smooth angles from get_coastline_angle() using a gaussian filter
    Angles is an xarray dataarray from 0 to 360.
    Sigma is the sigma of the gaussian filter
    """
    z = np.exp(1j * np.deg2rad(angles.values))
    z = np.rad2deg(np.angle(scipy.ndimage.gaussian_filter(z, sigma))) % 360
    return xr.DataArray(z,dims=angles.dims,coords=angles.coords)    

def get_weights(x, p=4, q=2, R=5, slope=-1, r=10000):
    """
    Calculate weights for averaging angles between pixels and coastlines.
    This function computes weights based on the distance from a coastline, using a piecewise function with different inverse powers before and after a specified distance `R`. The weights smoothly transition at `R` with a specified slope, and are set to zero beyond a cutoff distance `r`.
    Parameters
    ----------
    x : array_like
        Distance(s) from the coastline.
    p : float, optional
        Inverse power to decrease weights after distance `R`. Default is 4.
    q : float, optional
        Inverse power to decrease weights before distance `R`. Default is 2.
    R : float, optional
        Distance at which the inverse weighting power changes from `p` to `q`. Default is 5.
    slope : float, optional
        Slope of the function at point `R`. Default is -1.
    r : float, optional
        The distance at which the weights go to zero (to avoid overflows). Default is 10000.
    Returns
    -------
    y : array_like
        Calculated weights for each input distance.
    Notes
    -----
    The function is based on a method by Ewan Short. 
    
    Method
    -------
    Continuity and smoothness is ensured at `x = R` by equating the function and its derivative at that point.

    Let y1 = m1 * (x / R) ** (-p) for x > R.
    Let y2 = S - m2 * (x / R) ** (q) for x <= R.
    Equate y1 and y2 and their derivative at x = R to get
    S = m1 + m2
    slope = -p * m1 = -q * m2 => m1 = -slope/p and m2 = -slope/q
    Thus specifying p, q, R, and the function's slope at x=R determines m1, m2 and S.

    """

    m1 = -slope/p
    m2 = -slope/q
    S = m1 + m2
    y = da.where(x>R,  m1 * (x / R) ** (-p), S - m2 * (x / R) ** (q))
    y = da.where(x==0, np.nan, y)
    y = da.where(x>r, 0, y)
    return y    

def load_aus2200_static(exp_id,lon_slice,lat_slice,chunks="auto"):

    """
    Load static fields for the mjo-enso AUS2200 experiment, stored on the bs94 project.

    Parameters
    ----------
    exp_id : str
        Experiment ID. Must be one of 'mjo-elnino2016', 'mjo-lanina2018', or 'mjo-neutral2013'.
    lon_slice : slice or array-like
        Slice or indices to restrict longitude domain.
    lat_slice : slice or array-like
        Slice or indices to restrict latitude domain.
    chunks : str or dict, optional
        Chunking for xarray open_mfdataset (default is "auto").

    Returns
    -------
    orog : xarray.DataArray
        Orography field for the selected domain.
    lsm : xarray.DataArray
        Binary land-sea mask (1 for land, 0 for sea) for the selected domain.
    """

    assert exp_id in ['mjo-elnino2016', 'mjo-lanina2018', 'mjo-neutral2013'], "exp_id must either be 'mjo-elnino2016', 'mjo-lanina2018' or 'mjo-neutral2013'"
    
    orog = xr.open_mfdataset("/g/data/bs94/AUS2200/"+exp_id+"/v1-0/fx/orog/orog_AUS2200_*_fx.nc",chunks=chunks).\
            sel(lat=lat_slice,lon=lon_slice)
    lsm = xr.open_mfdataset("/g/data/bs94/AUS2200/"+exp_id+"/v1-0/fx/lmask/lmask_AUS2200_*_fx.nc",chunks=chunks).\
            sel(lat=lat_slice,lon=lon_slice)

    return orog.orog, ((lsm.lmask==100)*1)

def load_aus2200_variable(vname, t1, t2, exp_id, lon_slice, lat_slice, freq, hgt_slice=None, chunks="auto", staggered=None, dx=0.022, smooth=False, smooth_axes=None, sigma=2, interp_hgts=False, dh=100):

    """
    Load variables from the mjo-enso AUS2200 experiment, stored on the bs94 project.

    Parameters
    ----------
    vname : str
        Name of AUS2200 variable to load.
    t1 : str
        Start time in "%Y-%m-%d %H:%M".
    t2 : str
        End time in "%Y-%m-%d %H:%M".
    exp_id : str
        Experiment ID. Must be one of 'mjo-elnino2016', 'mjo-lanina2018', or 'mjo-neutral2013'.
    lon_slice : slice or array-like
        Slice or indices to restrict longitude domain.
    lat_slice : slice or array-like
        Slice or indices to restrict latitude domain.
    freq : str
        Time frequency. Must be "10min", "1hr", or "1hrPlev".
    hgt_slice : slice or array-like, optional
        Slice to restrict data in the vertical (in m).
    chunks : str or dict, optional
        Chunking for xarray open_mfdataset (default is "auto").
    staggered : str, optional
        If not None, the data is staggered in the specified dimension ("lat", "lon", or "time").
    dx : float, optional
        The distance to stagger the data by if staggered in lat or lon (in degrees, default is 0.022).
    smooth : bool, optional
        If True, smooth the data using a Gaussian filter.
    smooth_axes : iterable, optional
        If smoothing, the axes to smooth over.
    sigma : float, optional
        If smoothing, the sigma of the Gaussian filter (default is 2).
    interp_hgts : bool, optional
        If True, interpolate the data to regular height levels.
    dh : int, optional
        If interpolating to height levels, the height increment (in m, default is 100).

    Returns
    -------
    da : xarray.DataArray
        The requested variable, optionally smoothed and/or interpolated to regular height levels.

    Notes
    -----
    - This code currently does not support interpolating in height, as the relevant AUS2200 variable (Z_agl) is not on bs94. A version of Z_agl has been pre-computed and stored on ng72, and this is used if interp_hgts is True, but may not be accessible to all users.
    - If the data is being smoothed or interpolated, the relevant dimensions are set to -1 in the chunks dict.
    - De-staggering in lat/lon can fail at the domain edges depending on the slice used. If this happens, try extending the slice by one grid point in the relevant direction.
    """

    #This code makes sure the inputs for experiment id and time frequency match what is on disk 
    assert exp_id in ['mjo-elnino2016', 'mjo-lanina2018', 'mjo-neutral2013'], "exp_id must either be 'mjo-elnino2016', 'mjo-lanina2018' or 'mjo-neutral2013'"
    assert freq in ["10min", "1hr", "1hrPlev"], "exp_id must either be '10min', '1hr', '1hrPlev'"

    #We are loading a list of files from disk using xr.open_mfdataset. This preprocessing 
    # just slices the lats, lons and levels we are interested in for each file, which is more efficient
    def _preprocess(ds):
        ds = ds.sel(lat=lat_slice,lon=lon_slice)
        return ds
    def _preprocess_hgt(ds):
            ds = ds.sel(lat=lat_slice,lon=lon_slice,lev=hgt_slice)
            return ds   

    #Set up the time and lat/lon slices if the data is staggered
    orog, lsm = load_aus2200_static(exp_id,lon_slice,lat_slice)
    if staggered is not None:
        if staggered == "lat":
            lat_slice=slice(lat_slice.start-(dx*0.5),lat_slice.stop+(dx*0.5))
        elif staggered == "lon":
            lon_slice=slice(lon_slice.start-(dx*0.5),lon_slice.stop+(dx*0.5))
        elif staggered == "time":
            if freq == "10min":
                time_delta = dt.timedelta(minutes=10)
                freq_str = freq
            elif (freq == "1hr") | (freq == "1hrPlev"):
                time_delta = dt.timedelta(hours=1)
                freq_str = "1h"
            unstaggered_times = pd.date_range(t1,t2,freq=freq_str)
            t1 = pd.to_datetime(t1) - time_delta
            t2 = pd.to_datetime(t2) + time_delta
        else:
            raise ValueError("Invalid stagger dim")
    
    #Load the data from disk. If hgt_slice is not None, then we are loading 3D data
    fnames = "/g/data/bs94/AUS2200/"+exp_id+"/v1-0/"+freq+"/"+vname+"/"+vname+"_AUS2200*.nc"
    if hgt_slice is not None:
        da = xr.open_mfdataset(fnames, 
                               chunks=chunks, 
                               parallel=True, 
                               preprocess=_preprocess_hgt).sel(time=slice(t1,t2))[vname]
    else:
        da = xr.open_mfdataset(fnames,
                               chunks=chunks,
                               parallel=True,
                               preprocess=_preprocess).sel(time=slice(t1,t2))[vname]
    
    #Destagger the data if required
    if staggered == "lat":
        da = (da.isel(lat=slice(0,-1)).assign_coords({"lat":lsm.lat}) +
                        da.isel(lat=slice(1,da.lat.shape[0])).assign_coords({"lat":lsm.lat})) / 2        
    if staggered == "lon":
        da = (da.isel(lon=slice(0,-1)).assign_coords({"lon":lsm.lon}) +
                       da.isel(lon=slice(1,da.lon.shape[0])).assign_coords({"lon":lsm.lon})) / 2         
    if staggered == "time":
        da = (da.isel(time=slice(0,-1)).assign_coords({"time":unstaggered_times}) +\
                    da.isel(time=slice(1,da.time.shape[0])).assign_coords({"time":unstaggered_times})) / 2

    #Optional smoothing using gaussian filter
    if smooth:
        if smooth_axes is not None:
            for ax in smooth_axes:
                chunks[ax] = -1
            smooth_axes = (np.where(np.in1d(da.isel(time=0).dims,smooth_axes))[0])
        else:
            chunks["lev"] = -1
            chunks["lat"] = -1
            chunks["lon"] = -1
        da = da.map_blocks(
            gaussian_filter_time_slice,
            kwargs={"sigma":sigma,"axes":smooth_axes},
            template=da
        )

    #Interpolate to regular height levels
    if interp_hgts:
        chunks["lev"] = -1
        #Created in aus2200_hybrid_height_calc()
        Z_agl = xr.open_zarr("/g/data/ng72/ab4502/sea_breeze_detection/aus2200_z_agl.zarr/",
                             chunks=chunks).Z_agl.sel(lev=da.lev,lat=da.lat,lon=da.lon)
        da = interp_model_level_to_z(
            Z_agl,
            da.chunk({"lev":-1}),
            "lev",
            np.arange(hgt_slice.start,hgt_slice.stop+dh,dh),
            model="AUS2200"
            )        

    da = da.assign_attrs({"smoothed":smooth})
    if smooth:
        da = da.assign_attrs({"gaussian_smoothing_sigma":sigma})

    return da    

def interp_model_level_to_z(z_da,var_da,mdl_dim,heights,model="ERA5"):

    """
    Linearly interpolate from model level data to height levels. Supported for AUS2200 and ERA5 data.

    Parameters
    ----------
    z_da : xarray.DataArray
        Height data (either AGL or above geoid) for each model level.
    var_da : xarray.DataArray
        Variable to interpolate, with the same model level dimension as z_da.
    mdl_dim : str
        Name of the model level dimension (e.g., 'lev', 'hybrid'). Model levels must be decreasing (height increasing).
    heights : numpy.ndarray
        Array of target height levels to interpolate to.
    model : str, optional
        Model name, either "ERA5" or "AUS2200"

    Returns
    -------
    xarray.DataArray
        Variable interpolated to the specified height levels. If the requested height is below the lowest model level, data from the lowest model level is returned. If above the highest model level, NaNs are returned.

    Notes
    -----
    - If the requested height is below the lowest model level, data from the lowest model level is returned.
    - Note that for ERA5, the lowest model level is within the first few 10s of meters above the surface.
    - If the requested height is above the highest model level, then NaNs are returned.
    """

    if model=="ERA5":
        assert z_da[mdl_dim][0] > z_da[mdl_dim][-1], "Model levels should be decreasing"

    interp_da = xr.apply_ufunc(interp_scipy,
                heights,
                z_da,
                var_da,
                input_core_dims=[ ["height"], [mdl_dim], [mdl_dim]],
                output_core_dims=[["height"]],
                exclude_dims=set((mdl_dim,)),
                dask="parallelized",
                output_dtypes=var_da.dtype,
                vectorize=True)
    interp_da["height"] = heights
    
    return interp_da    

def interp_scipy(x, xp, fp):
    f = scipy.interpolate.interp1d(xp, fp, kind="linear", fill_value="extrapolate")
    return f(x)    

def round_times(ds,freq):
    
    """
    For dataarray, round the time coordinate to the nearst freq

    Useful for AUS2200 where time values are sometimes very slightly displaced from a 10-minute time step
    """

    if freq in ["1hrPlev","1hr"]:
        ds["time"] = ds.time.dt.round("1h")
    elif freq in ["10min"]:
        ds["time"] = ds.time.dt.round("10min")
    else:
        raise Exception("freq must be one of '10min', '1hr', or '1hrPlev'")

    return ds    

def gaussian_filter_time_slice(time_slice,sigma,axes):
    """
    Apply a gaussian filter to a time slice of data. For use with map_blocks
    """
    out_ds = xr.DataArray(scipy.ndimage.gaussian_filter(
        time_slice.isel(time=0), sigma, axes=axes
        ),dims=time_slice.isel(time=0).dims, coords=time_slice.isel(time=0).coords)
    out_ds = out_ds.expand_dims("time")
    out_ds["time"] = time_slice.time
    return out_ds    