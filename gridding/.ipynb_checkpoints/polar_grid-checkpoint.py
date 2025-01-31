import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import scipy.interpolate as interp
from tqdm import tqdm

def reg_to_polar_nc(infile,outfile,hemis='sh',ratio_new_grid=0.5,outer_data_val=0,
                    var='z',lonname='longitude',latname='latitude'):
    xf=xr.open_dataset(infile)
    grid_original=grid_from_xarray(xf[var])
    _,grid_polar=make_polar_grid(hemis,grid_original,ratio_new_grid=ratio_new_grid)
    polar_arr=convert_data_to_polar(hemis,grid_polar,grid_original,xf[var],outer_data_val=outer_data_val)
    
    # Create xarray DataArray and save
    #xr.DataArray(
    #    polar_arr,
    #    coords={"time": grid_polar['time'], 
    #            latname: ((latname, lonname), grid_polar['latitude2D']), 
    #            lonname: ((latname, lonname), grid_polar['longitude2D'])},
    #    dims=["time", latname, lonname],
    #    name=var
    #).to_netcdf(outfile)

    #return grid_polar

    # Create xarray DataArray and save
    xr.DataArray(
        polar_arr,
        coords={"time": grid_polar['time'], 
                "y": np.array(list(range(grid_polar['y'].shape[1]))),
                "x": np.array(list(range(grid_polar['x'].shape[0]))),
                latname: (("y", "x"), grid_polar['latitude2D']), 
                lonname: (("y", "x"), grid_polar['longitude2D'])},
        dims=["time", "y", "x"],
        name=var
    ).to_netcdf(outfile)

def make_polar_grid(hemis, grid_original, transform=ccrs.PlateCarree(),ratio_new_grid=1):
    if hemis == 'nh':

        # Set up the North-Polar Stereographic projection
        projection = ccrs.NorthPolarStereo()
        
        # Transform points to the projection coordinates
        x_nb, y_nb = projection.transform_point(180.0, 0.0, transform)  # Northern boundary #180.0, 20.0, transform)  # Northern boundary
        x_sb, y_sb = projection.transform_point(0.0, 0.0, transform)  # Southern boundary #0.0, 20.0, transform)  # Southern boundary
        x_np, y_np = projection.transform_point(0.0, 90.0, transform)  # North Pole #0.0, 90.0, transform)  # North Pole
    
    else:
        # Set up the North-Polar Stereographic projection
        projection = ccrs.SouthPolarStereo()
        
        # Transform points to the projection coordinates
        x_nb, y_nb = projection.transform_point(0, 0, transform)  # Northern boundary
        x_sb, y_sb = projection.transform_point(180.0, 0.0, transform)  # Southern boundary
        x_np, y_np = projection.transform_point(0.0, -90.0, transform)  # South Pole

    
    # Average grid spacing
    dxy = grid_original['grid_distance']*ratio_new_grid
    
    # Create grid in Cartesian coordinates
    xy_vals = np.arange(dxy/2 + y_sb, y_nb, dxy)
    y, x = np.meshgrid(xy_vals, xy_vals)

    # Inverse project the x, y coordinates back to lat/lon
    lonlat = transform.transform_points(projection, x, y)
    lon, lat = lonlat[..., 0], lonlat[..., 1]
    
    # For clon/clat (constant longitude at x_np, varying latitude at xy_vals)
    clonlat = transform.transform_points(projection, np.ones(xy_vals.shape) * x_np, xy_vals)
    clon, clat = clonlat[..., 0], clonlat[..., 1]
    
    # Derive actual dx, dy from the spacing of latitudes along the 0/180-meridian
    dlon = np.abs(clon[1:] - clon[:-1])
    over_np = np.where(dlon > 170.0)[0]

    dxy_real = np.abs(clat[1:] - clat[:-1])
    dxy_real[over_np] = 180.0 - np.abs(clat[over_np]) - np.abs(clat[over_np])
    dxy_real = (dxy_real[1:] + dxy_real[:-1]) * 111111.111

    dx = np.empty(lon.shape)
    dy = np.empty(lon.shape)

    dx[:,1:-1] = dxy_real[np.newaxis,:]
    dx[:,0] = dxy_real[0]
    dx[:,-1] = dxy_real[-1]

    dy[1:-1,:] = dxy_real[:,np.newaxis]
    dy[0,:] = dxy_real[0]
    dy[-1,:] = dxy_real[-1]

    grid={}
    grid['longitude']=None
    grid['latitude']=None
    grid['longitude2D'],grid['latitude2D']=lon,lat
    grid['time']=grid_original['time']
    
    grid['x'],grid['y']=x,y
    grid['dx'],grid['dy'],grid['area'],grid['grid_distance']=calc_grid_distance_area(grid['longitude2D'],grid['latitude2D'])

    return projection, grid 

def convert_data_to_polar(hemis,polar_grid,reg_grid,orig_data,outer_data_val=0):
    ''' Converts a data array of a regular lat/lon grid to a polar grid given the grid dictionary.
    Requires: Running of make_polar_grid function for use.

    Parameters
    ----------
    hemis : str of either 'sh' or 'nh'
        Hemisphere required
    polar_grid : dict produced by make_polar_grid
        Grid information of the polar grid, usually produced by the make_polar_grid function
    reg_grid : dict of the original, regulr grid
        Grid information of the polar grid, usually produced by a grid.py function
    orig_data : np.ndarray with dimensions (z,y,x)
        Data to be converted

    Optional
    ----------
    outer_data_val: float64, default=0
    
    Returns
    -------
    np.ndarray
        Polar data projection of the original data on the required hemisphere
    '''
    ilons, ilats = concat1lonlat(reg_grid['longitude2D'], reg_grid['latitude2D'])
    ilats = ilats[::-1,0]
    ilons = ilons[0,:]

    if hemis == 'nh':
        orig_data=orig_data.where(reg_grid['latitude2D']>0,outer_data_val)
    else:
        orig_data=orig_data.where(reg_grid['latitude2D']<0,outer_data_val)
        
    z_regrid=np.zeros((orig_data.shape[0],polar_grid['latitude2D'].shape[0],polar_grid['longitude2D'].shape[1]))
    for z in tqdm(range(orig_data.shape[0]),total=orig_data.shape[0]):
        ifunc = interp.RectBivariateSpline(ilats, ilons, concat1(orig_data[z,::-1,:].values))
        z_regrid[z] = ifunc(polar_grid['latitude2D'], polar_grid['longitude2D'], grid=False)
        del ifunc

    return z_regrid

def concat1(data):
    ''' Concatenate one latitude band in x-direction to a data array

    To be able to plot circumpolar plots without a data gap, the sphericity of the data
    must explicitly be demonstrated by contatenating the data from lon=180E to appear 
    also as data for lon=180W.

    Parameters
    ----------
    data : np.ndarray with 1-4 dimensions
        Data to be extended
    
    Returns
    -------
    np.ndarray
        Extended data
    '''
    
    if len(data.shape) == 1:
        data = np.concatenate((data, np.reshape(data[0], (1,)) ), axis=0)
    elif len(data.shape) == 2:
        data = np.concatenate((data, np.reshape(data[:,0], (data.shape[0], 1)) ), axis=1)
    elif len(data.shape) == 3:
        data = np.concatenate((data, np.reshape(data[:,:,0], (data.shape[0], data.shape[1], 1)) ), axis=2)
    elif len(data.shape) == 4:
        data = np.concatenate((data, np.reshape(data[:,:,:,0], (data.shape[0], data.shape[1], data.shape[2], 1)) ), axis=3)
    else:
        raise NotImplementedError('Concatenation not implemented for %d dimensions' % len(data.shape))
    
    return data

def concat1lonlat(x, y):
    ''' Concatenate one latitude band in x-direction to coordinate arrays

    To be able to plot circumpolar plots without a data gap, the sphericity of the data
    must explicitly be demonstrated by contatenating the data from lon=180E to appear 
    also as data for lon=180W.

    Parameters
    ----------
    x : np.ndarray with dimensions (y,x)
        Longitudes for each grid point
    y : np.ndarray with dimensions (y,x)
        Latitudes for each grid point
    
    Returns
    -------
    2-tuple of np.ndarray
        Extended coordinate arrays
    '''
    
    # Some map projections need the lon, lat array to be C-aligned.
    lon = np.ascontiguousarray(concat1(x))
    lat = np.ascontiguousarray(concat1(y))

    lon[:,-1] += 360.0

    return lon, lat

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km
    
def grid_from_xarray(data,
                      lonname='longitude',latname='latitude',timename='time', **kwargs):

    ''' Create a grid dictionary from an xarray DataArray
   
    Parameters
    ----------
   
    data : xr.DataArray with shape (nz,ny,nx) and dtype float64

    Optional
    ----------
    lonname : name of the longitude variable in the xarray (default: longitude)
    latname : name of the latitude variable in the xarray (default: latitude)
    timename : name of the time variable in the xarray (default: time)
   
    Returns
    -------
    grid : dictionary of grid definitions required by detection

    grid dictionary includes:
        - longitude : 1D np.array of longitude
        - latitude : 1D np.array of latitude
        - time : 1D np.array of time (pd.datetime)
        - longitude2D : 2D np.array of longitude
        - latitude2D : 2D np.array of latitude
        - dx : x distance (m)
        - dy : y distance (m)
        - area : area of grid cell (m**2)
        - grid_distance : average grid distance over the domain (m)
    '''
    grid={}
    grid['longitude']=data[lonname].values
    grid['latitude']=data[latname].values
    grid['longitude2D'],grid['latitude2D']=np.meshgrid(data[lonname],data[latname])
    grid['time']=pd.to_datetime(data[timename])
    
    grid['dx'],grid['dy'],grid['area'],grid['grid_distance']=calc_grid_distance_area(grid['longitude2D'],grid['latitude2D'])

    return grid

def calc_grid_distance_area(lon,lat):
    """ Function to calculate grid parameters
        It uses haversine function to approximate distances
        It approximates the first row and column to the sencond
        because coordinates of grid cell center are assumed
        lat, lon: input coordinates(degrees) 2D [y,x] dimensions
        dx: distance (m)
        dy: distance (m)
        area: area of grid cell (m2)
        grid_distance: average grid distance over the domain (m)
    """
    dy = np.zeros(lon.shape)
    dx = np.zeros(lat.shape)

    dx[:,1:]=haversine(lon[:,1:],lat[:,1:],lon[:,:-1],lat[:,:-1])
    dy[1:,:]=haversine(lon[1:,:],lat[1:,:],lon[:-1,:],lat[:-1,:])

    dx[:,0] = dx[:,1]
    dy[0,:] = dy[1,:]
    
    dx = dx * 10**3
    dy = dy * 10**3

    area = dx*dy
    grid_distance = np.mean(np.append(dy[:, :, None], dx[:, :, None], axis=2))

    return dx,dy,area,grid_distance

