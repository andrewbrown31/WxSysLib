#!/usr/bin/env python
from utils.general.file_utils import create_directory
from datetime import datetime, timedelta
from geopy.distance import geodesic
import xarray as xr
import pandas as pd
import numpy as np
import subprocess
import calendar
import os


# modify trajectory
def modify_traj(lagranto_traj_dir):
    df = pd.read_csv(lagranto_traj_dir,sep='\s+',
                     skiprows=[0,1,3],     
                     header=0)
    df['track_id'] = (df['time'] == 0.0).cumsum()
    return df


def create_lagranto_dirstruct(runpath,casename):
    #### Create the case directory ####
    casedir=os.path.join(runpath,casename)
    create_directory(casedir)

    startfdir=os.path.join(runpath,casename,'starf')
    create_directory(startfdir)
    
    pfiledir=os.path.join(runpath,casename,'pfile')
    create_directory(pfiledir)

    trajdir=os.path.join(runpath,casename,'traj')
    create_directory(trajdir)
    
    return casedir,startfdir,pfiledir,trajdir

def convert_datetime(input_datetime):
    dt = datetime.strptime(input_datetime, "%Y-%m-%dT%H")
    return dt.strftime("%Y%m%d_%H")

def create_datetime_string(year, month):
    if 1 <= month <= 12:
        last_day = calendar.monthrange(year, month)[1]
        first_date = datetime(year, month, 1)
        last_date = datetime(year, month, last_day)
        return f"{first_date.strftime('%Y%m%d')}-{last_date.strftime('%Y%m%d')}"
    else:
        return "Invalid month"

def generate_date_strings(start_date, end_date):
    start_year, start_month = start_date.year, start_date.month
    end_year, end_month = end_date.year, end_date.month
    current_year, current_month = start_year, start_month
    date_strings = []
    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        date_string = create_datetime_string(current_year, current_month)
        date_strings.append(date_string)
        if current_month == 12:
            current_month = 1
            current_year += 1
        else:
            current_month += 1     
    return date_strings

def get_Pfiles(reference_datetime, period, lon_range, lat_range, hour_range, pfile_dir):
    if period > 0:
        start_datetime = reference_datetime
        end_datetime = reference_datetime + timedelta(hours=period)
    if period < 0:
        start_datetime = reference_datetime + timedelta(hours=period)
        end_datetime = reference_datetime
    
    date_strings = generate_date_strings(start_datetime, end_datetime)
    ERA5_PL_SIR       = '/g/data/rt52/era5/pressure-levels/reanalysis' 
    ERA5_SFC_SIR      = '/g/data/rt52/era5/single-levels/reanalysis'
    files_u  = [f"{ERA5_PL_SIR}/u/{ds[:4]}/u_era5_oper_pl_{ds}.nc" for ds in date_strings]
    files_v  = [f"{ERA5_PL_SIR}/v/{ds[:4]}/v_era5_oper_pl_{ds}.nc" for ds in date_strings]
    files_w  = [f"{ERA5_PL_SIR}/w/{ds[:4]}/w_era5_oper_pl_{ds}.nc" for ds in date_strings]
    files_sp = [f"{ERA5_SFC_SIR}/sp/{ds[:4]}/sp_era5_oper_sfc_{ds}.nc" for ds in date_strings]
    
    def preprocess(ds):
        return ds.sel(
            time      = ds.time.dt.hour.isin(list(range(*hour_range))),
            longitude = np.arange(*lon_range),
            latitude  = np.arange(*lat_range)
        )
    with xr.open_mfdataset(
        files_u+files_v+files_w+files_sp,
        combine   = 'by_coords',
        chunks    = {'latitude': -1, 'longitude': -1},
        preprocess=preprocess
    ) as ds:
        dataset = ds.sel(time=slice(start_datetime, end_datetime)).isel(latitude=slice(None, None, -1)) # latitude must by ascending order
    
    # ------------------------------------ Create P files ------------------------------------ #
    # U-wind
    u_slice = dataset['u']
    U=xr.DataArray(data=u_slice,
                    dims=['time',"lev","lat","lon"],
                    coords=dict(lev=(["lev"], u_slice.level.values*100),
                                lat=(["lat"], u_slice.latitude.values),
                                lon=(["lon"], u_slice.longitude.values),
                                time=(["time"],u_slice.time.values)
                                ),
                    attrs=dict(standard_name="eastward_wind",
                                long_name="U component of wind",
                                units="m s**-1"
                                ),
                    )
        
    # V-wind
    v_slice = dataset['v']
    V=xr.DataArray(data=v_slice,
                    dims=['time',"lev","lat","lon"],
                    coords=dict(lev=(["lev"], v_slice.level.values*100),
                                lat=(["lat"], v_slice.latitude.values),
                                lon=(["lon"], v_slice.longitude.values),
                                time=(["time"],v_slice.time.values)
                                ),
                    attrs=dict(standard_name="northward_wind",
                                long_name="V component of wind",
                                units="m s**-1"
                                ),
                    )
    
    # OMEGA
    w_slice = dataset['w']
    OMEGA=xr.DataArray(data=w_slice,
                        dims=['time',"lev","lat","lon"],
                        coords=dict(lev=(["lev"], w_slice.level.values*100),
                                lat=(["lat"], w_slice.latitude.values),
                                lon=(["lon"], w_slice.longitude.values),
                                time=(["time"],w_slice.time.values)
                                ),
                        attrs=dict(standard_name="lagrangian_tendency_of_air_pressure",
                                   long_name="Vertical velocity",
                                   units="Pa s**-1"
                                 ),
                    )
    
    # PS
    sp_slice = dataset['sp']
    PS=xr.DataArray(data=sp_slice,
                    dims=['time',"lat","lon"],
                    coords=dict(lat=(["lat"], sp_slice.latitude.values),
                                lon=(["lon"], sp_slice.longitude.values),
                                time=(["time"],sp_slice.time.values)
                                ),
                    attrs=dict(long_name="Surface pressure",
                                units="Pa"
                                ),
                    )
    
    # constrcuct dataset
    ds_lagranto = xr.Dataset({
        'U':U,
        'V':V,
        'OMEGA':OMEGA,
        'PS':PS,
                              })
    
    # lon attrs    
    ds_lagranto.lon.attrs["standard_name"]='longitude'
    ds_lagranto.lon.attrs["long_name"]='longitude'
    ds_lagranto.lon.attrs["unit"]='degrees_east'
    ds_lagranto.lon.attrs["axis"]='X'
        
    # change lat and attrs
    ds_lagranto.lat.attrs["standard_name"]='latitude'
    ds_lagranto.lat.attrs["long_name"]='latitude'
    ds_lagranto.lat.attrs["unit"]='degrees_north'
    ds_lagranto.lat.attrs["axis"]='Y'
    
    # levs attrs
    ds_lagranto.lev.attrs["standard_name"]='air_pressure'
    ds_lagranto.lev.attrs["long_name"]='pressure'
    ds_lagranto.lev.attrs["units"]='Pa'
    ds_lagranto.lev.attrs["positive"]='down'
    ds_lagranto.lev.attrs["axis"]='Z'
    # ------------------------------------ Create P files ------------------------------------ #
    
    # save Pfiles
    for dt in np.arange(0, ds_lagranto.time.size, 1): 
        time0 = datetime.now()
        ds_lagranto_eachtime = ds_lagranto.isel(time=dt)
        prefix = convert_datetime(str(ds_lagranto_eachtime.time.values)[0:13])
        ds_lagranto_eachtime = ds_lagranto_eachtime.expand_dims('time')
        print(f'Creating P{prefix}...')
        ds_lagranto_eachtime.to_netcdf(f"{pfile_dir}/P{prefix}", encoding={'time': {'dtype': 'float32'}})
        time1 = datetime.now()
        print(f'     {time1-time0}')


def run_Lagranto(startfdir,
                 period,
                 trajdir,
                 input_interval,
                 output_interval,
                 reference_date,
                 pfiledir, 
                 quiet=False):
    
    # Lagranto command
    lagranto_command = f"{os.getenv('LAGRANTODIR')}/prog/caltra " \
                       f"{startfdir} " \
                       f"{period} " \
                       f"{trajdir} " \
                       f"-i {input_interval} " \
                       f"-o {output_interval} " \
                       f"-ref {reference_date} " \
                       f"-cdf {pfiledir}/ " \
                       "flat"

    # Execute the command with shell=True for a single string command
    lagranto_process = subprocess.Popen(
        lagranto_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for the process to complete and capture output
    stdout, stderr = lagranto_process.communicate()

    if not quiet:
        if stdout:
            print("Output:", stdout)
        if stderr:
            print("Error:", stderr)
