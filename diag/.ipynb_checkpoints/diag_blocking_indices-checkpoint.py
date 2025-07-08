##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
#################################################### Blocking diagnostics ########################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################

import numpy as np
import xarray as xr
import os
import time
from tqdm import tqdm

def masato_blocking_index(z500,lat_p=60,lat_e=30,lat_band=15):
    
    ''' Masato Blocking Index by Masato et al. (2012): https://doi.org/10.1002/qj.990
    Created by Tess J. Parker (CSIRO Environment) and Michael A. Barnes (ARC CoE 21st Century Weather, Monash University)
    
    Parameters
    ----------
   
    z500 : xr.DataArray with shape (time,latitude,longitude) and dtype float64
        The 500hPa geopotential field.
    
    Options
    -------

    lat_p : float64
        Poleward latitude for calculation
    lat_e : float64
        Equatrward latitude for calculation
    lat_band : float64
        Size of the latitude band for calculation (lat_band = (delta lat)/2 in Masato and related literature)
   
    Returns
    -------
    b_index : xr.DataArray with shape (time,latitude,longitude) and dtype float64
        xarray DataArray of the Masato Blocking Index. Negative values indicate a reversal
        in the geopotential gradient and therefore potential blocking.
    '''
    b_index=xr.zeros_like(z500.sel(latitude=slice(lat_p+lat_band,-(lat_p+lat_band))))
    b_index=b_index.rename('b_index')
    
    for it, itime in enumerate(tqdm(b_index.time.values, desc='Masato blocking index calculation: ')):
        for i,ilat in enumerate(b_index.latitude.values):
            if abs(ilat)<=lat_p and abs(ilat)>=lat_e:
                z_s=(2.0/lat_band)*(z500.sel(time=itime,latitude=slice(ilat,ilat-lat_band)).integrate(coord='latitude'))
                z_n=(2.0/lat_band)*(z500.sel(time=itime,latitude=slice(ilat+lat_band,ilat)).integrate(coord='latitude'))
                if ilat > 0:
                    b_index.loc[itime, ilat, :] = z_n-z_s
                else: 
                    b_index.loc[itime, ilat, :] = z_s-z_n
            else:
                b_index.loc[itime, ilat, :] = np.nan
    return b_index
