import numpy as np
import xarray as xr
import metpy.calc as mpcalc
import dask.array as da
from utils.diagnostics.sea_breeze.sea_breeze_utils import vert_mean_wind, daily_mean_wind

def kinematic_frontogenesis(q,u,v):

    """
    Calculate 2D kinematic frontogenesis using water vapour mixing ratio.

    Identifies regions where moisture fronts are increasing or decreasing due to flow deformation, including sea breeze fronts.

    Uses MetPy formulation but implemented with numpy/xarray for efficiency.
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.frontogenesis.html

    Parameters
    ----------
    q : xarray.DataArray
        Water vapour mixing ratio (or any scalar field), with lat/lon/time coordinates in units kg/kg.
    u : xarray.DataArray
        U wind component, with matching coordinates.
    v : xarray.DataArray
        V wind component, with matching coordinates.

    Returns
    -------
    xarray.Dataset
        2D kinematic frontogenesis in units (g/kg) / 100 km / 3h.

    Notes
    -----
    The input data is rechunked in lat/lon dimensions for gradient calculations.
    """

    #Rechunk data in one lat and lon dim
    q = q.chunk({"lat":-1,"lon":-1})
    u = u.chunk({"lat":-1,"lon":-1})
    v = v.chunk({"lat":-1,"lon":-1})

    #Convert specific humidity to g/kg
    q = q*1000

    #Calculate grid spacing in km using metpy, in x and y
    x, y = np.meshgrid(q.lon,q.lat)
    dx, dy = mpcalc.lat_lon_grid_deltas(x,y)

    #Convert the x and y grid spacing arrays into xarray datasets. Need to interpolate to match the original grid
    dx = xr.DataArray(np.array(dx),dims=["lat","lon"],coords={"lat":q.lat.values, "lon":q.lon.values[0:-1]}).\
            interp({"lon":q.lon,"lat":q.lat},method="linear",kwargs={"fill_value":"extrapolate"}).\
            chunk({"lat":q.chunksizes["lat"][0], "lon":q.chunksizes["lon"][0]})
    dy = xr.DataArray(np.array(dy),dims=["lat","lon"],coords={"lat":q.lat.values[0:-1], "lon":q.lon.values}).\
            interp({"lon":q.lon,"lat":q.lat},method="linear",kwargs={"fill_value":"extrapolate"}).\
            chunk({"lat":q.chunksizes["lat"][0], "lon":q.chunksizes["lon"][0]})

    #Calculate horizontal moisture gradient
    ddy_q = (xr.DataArray(da.gradient(q,axis=q.get_axis_num("lat")), dims=q.dims, coords=q.coords) / dy)
    ddx_q = (xr.DataArray(da.gradient(q,axis=q.get_axis_num("lon")), dims=q.dims, coords=q.coords) / dx)
    mag_dq = np.sqrt( ddy_q**2 + ddx_q**2)

    #Calculate horizontal U and V gradients, as well as divergence and deformation 
    #Following https://www.ncl.ucar.edu/Document/Functions/Contributed/shear_stretch_deform.shtml
    ddy_u = (xr.DataArray(da.gradient(u,axis=q.get_axis_num("lat")), dims=q.dims, coords=q.coords) / dy)
    ddx_u = (xr.DataArray(da.gradient(u,axis=q.get_axis_num("lon")), dims=q.dims, coords=q.coords) / dx)
    ddy_v = (xr.DataArray(da.gradient(v,axis=q.get_axis_num("lat")), dims=q.dims, coords=q.coords) / dy)
    ddx_v = (xr.DataArray(da.gradient(v,axis=q.get_axis_num("lon")), dims=q.dims, coords=q.coords) / dx)
    div = ddx_u + ddy_v
    strch_def = ddx_u - ddy_v
    shear_def = ddx_v + ddy_u
    tot_def = np.sqrt(strch_def**2 + shear_def**2)

    #Calculate the angle between axis of dilitation and isentropes
    psi = 0.5 * np.arctan2(shear_def, strch_def)
    beta = np.arcsin((-ddx_q * np.cos(psi) - ddy_q * np.sin(psi)) / mag_dq)

    #Calculate frontogenesis following MetPy formulation
    F = 0.5 * mag_dq * (tot_def * np.cos(2 * beta) - div) * 1.08e9

    #Assign attributes to output
    out = xr.Dataset({"F":F})
    out["F"] = out["F"].assign_attrs(
        units = "g/kg/100km/3hr",
        long_name = "Moisture frontogenesis",
        description = "2d kinematic moisture frontogenesis parameter.")  

    return out

def calc_sbi(wind_ds,
                angle_da,
                vert_coord="height",
                alpha_height=0,
                height_method="blh",
                blh_da=None,
                blh_rolling=0,
                sb_heights=[500,2000],
                subtract_mean=False,
                height_mean=False,
                mean_heights=[0,4500]
                ):

    """
    Take an xarray dataset of 3d u and v winds, as well as a dataset of coastline angles, and apply the algorithm of Hallgren et al. (2023) to compute the sea breeze index via a single-column method.

    The method looks for an onshore flow at a low level (alpha_height) with an opposing, offshore flow aloft. The SBI is calculated for each vertical "aloft" level and then the maximum is taken. "Aloft" levels can be defined either statically (sb_heights) or using all levels below the boundary layer height (blh_da) if providing a dataset of boundary layer heights. 
    
    The SBI returns values between 0 and 1, where 1 indicates an onshore flow perpendicular to the coast with an exactly opposing offshore flow aloft. If there is no onshore flow or no opposing offshore flow aloft, the SBI is zero.

    Parameters
    ----------
    wind_ds : xarray.Dataset
        Dataset containing "u" and "v" wind component variables, with a vertical coordinate (see vert_coord) in metres.
    angle_da : xarray.DataArray
        DataArray of coastline orientation angles (degrees from North).
    vert_coord : str, optional
        Name of the vertical coordinate in wind_ds.        
    alpha_height : float, optional
        Height level in metres to define the "low-level" wind.        
    height_method : str, optional
        Method for selecting upper level heights to define aloft levels. Either "static" or "blh". "static" uses static height limits (sb_heights), "blh" uses a DataArray of boundary layer heights (blh_da).
    blh_da : xarray.DataArray, optional
        DataArray with boundary layer heights in metres. Used if height_method="blh" to define heights for sea breezes.
    blh_rolling : int, optional
        Number of rolling time windows over which to take the maximum boundary layer height. If zero, no rolling max is taken.        
    sb_heights : list or array-like, optional
        Bounds [min, max] in metres used to define the upper level sea breeze height if height_method="static".
    subtract_mean : bool, optional
        Whether to subtract the mean background wind and calculate perturbation SBI. Uses either the arithmetic mean over a layer (see mean_heights) or the daily mean.
    height_mean : bool, optional
        If subtract_mean is True, then whether to subtract the height mean over mean_heights. If False, subtract the daily mean.
    mean_heights : list or array-like, optional
        If subtract_mean and height_mean are True, then the bounds [min, max] in metres used to define the layer for mean wind calculation.

    Returns
    -------
    xarray.Dataset
        Dataset containing the sea breeze index (sbi).

    Notes
    ----------
    Options are provided to calculate the SBI using perturbation winds, by subtracting either the daily mean wind or the mean wind over a specified layer. However, this is not recommended as it is uncertain whether this method is valid for perturbation winds. The original method of Hallgren et al. (2023) uses total winds.

    References
    ----------
    Hallgren, C., Körnich, H., Ivanell, S., & Sahlée, E. (2023). A Single-Column Method to Identify Sea and Land Breezes in Mesoscale-Resolving NWP Models. Weather and Forecasting, 38(6), 1025-1039. https://doi.org/10.1175/WAF-D-22-0163.1
    """

    #Subtract the mean wind. Define mean as the mean over mean_heights m level, or the daily mean
    if subtract_mean:
        print("SUBTRACTING MEAN FROM WINDS FOR SBI CALC...")
        if height_mean:
            u_mean, v_mean = vert_mean_wind(wind_ds,mean_heights,vert_coord)
        else:
            u_mean, v_mean = daily_mean_wind(wind_ds)
        wind_ds["u"] = wind_ds["u"] - u_mean
        wind_ds["v"] = wind_ds["v"] - v_mean

    #Convert coastline orientation angle to the angle perpendicular to the coastline (pointing away from coast. from north)
    theta = (angle_da + 90) % 360

    #Calculate wind directions (from N) for low level (alpha) and all levels (beta)
    def compute_wind_direction(u, v):
        return (90 - np.rad2deg(np.arctan2(-v, -u))) % 360

    alpha = xr.apply_ufunc(
        compute_wind_direction,
        wind_ds["u"].sel({vert_coord: alpha_height}, method="nearest"),
        wind_ds["v"].sel({vert_coord: alpha_height}, method="nearest"),
        dask="parallelized",
        output_dtypes=[float],
    )

    beta = xr.apply_ufunc(
        compute_wind_direction,
        wind_ds["u"],
        wind_ds["v"],
        dask="parallelized", 
        output_dtypes=[float],  
    )            

    #Calculate the sea breeze index
    def compute_sbi(alpha, beta, theta):
        return (
        np.cos(np.deg2rad(alpha - theta)) *
        np.cos(np.deg2rad(alpha + 180 - beta))
    )
    
    sbi = xr.apply_ufunc(
        compute_sbi,
        alpha, 
        beta,  
        theta,             
        dask="parallelized",  
        output_dtypes=[float],  
    )        

    #Mask to zero everywhere except for the following conditions. Keep masked if angles are nans
    def sbi_conditions(sbi, alpha, beta, theta):
        sb_cond = ( (np.cos(np.deg2rad((alpha - theta)))>0), #Low level flow onshore
            (np.cos(np.deg2rad(beta - (theta+180)))>0), #Upper level flow offshore
            (np.cos(np.deg2rad(alpha + 180 - beta))>0) #Upper level flow opposing
                  )
        sbi_cond = xr.where(sb_cond[0] & sb_cond[1] & sb_cond[2], sbi, 0)
        return xr.where(np.isnan(theta),np.nan,sbi_cond)
    
    sbi = xr.apply_ufunc(
        sbi_conditions,
        sbi,
        alpha,  
        beta,   
        theta,             
        dask="parallelized",  
        output_dtypes=[float],  
    )        

    #Create a height variable in the dataset for masking
    time_dim = wind_ds.u.get_axis_num("time")
    lat_dim = wind_ds.u.get_axis_num("lat")
    lon_dim = wind_ds.u.get_axis_num("lon")
    height_dim = wind_ds.u.get_axis_num(vert_coord)
    _,hh,_,_ = da.meshgrid(
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[time_dim]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[time_dim]][0]}),
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[height_dim]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[height_dim]][0]}),
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[lat_dim]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[lat_dim]][0]}),
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[lon_dim]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[lon_dim]][0]}), indexing="ij")
    wind_ds["height_var"] = (("time",vert_coord,"lat","lon"),hh)

    #Mask sbi to only include levels below a certain height, based on height_method
    if height_method=="static":
        sbi = xr.where((sbi[vert_coord] >= sb_heights[0]) & (sbi[vert_coord] <= sb_heights[1]),sbi,0)
    elif height_method=="blh":
        if blh_rolling > 0:
            blh_da = blh_da.rolling({"time":blh_rolling}).max()
        sbi = xr.where((wind_ds.height_var <= blh_da),sbi,0)
    else:
        raise ValueError("Invalid height method")
    sbi = xr.where(np.isnan(theta),np.nan,sbi)

    #Compute each index as the max in the column
    sbi = sbi.max(vert_coord)

    #Convert to dataset and assign attributes
    sbi_ds = xr.Dataset({
        "sbi":sbi})
    sbi_ds = sbi_ds.assign_attrs(
        subtract_mean=str(subtract_mean),
        alpha_height=alpha_height,
        height_method=height_method,
        height_mean=str(height_mean),
        hgt_levs=str(wind_ds[vert_coord].values),
    )
    if height_method=="static":
        sbi_ds = sbi_ds.assign_attrs(
            sb_heights=str(sb_heights)
        )    
    sbi_ds["sbi"] = sbi_ds["sbi"].assign_attrs(
        units = "[0,1]",
        long_name = "Sea breeze index",
        description = "This index identifies regions where there is an onshore flow at a near-surface level with an opposing, offshore flow aloft in the boundary layer. The SBI is calculated for each vertical layer and then the maximum is taken. Following Hallgren et al. 2023 (10.1175/WAF-D-22-0163.1).")      
    
    return sbi_ds