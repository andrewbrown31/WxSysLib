# Snapshots for composites pl

# package
from datetime import datetime
from glob import glob
import xarray as xr
import pandas as pd
import numpy as np
import subprocess
import calendar
import os
import subprocess

# funcs
def ds_assign_lev(ds):  
    level_range = np.array([   1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,
                             125,  150,  175,  200,  225,  250,  300,  350,  400,  450,  500,
                             550,  600,  650,  700,  750,  775,  800,  825,  850,  875,  900,
                             925,  950,  975, 1000])

    # add level as coordiante
    ds_pl = ds.assign_coords(level=('level', level_range))

    return ds_pl

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
        
        # Increment month and year
        if current_month == 12:
            current_month = 1
            current_year += 1
        else:
            current_month += 1
            
    return date_strings

def create_snapshot_sfc_XY(csv_dir, txt_dir, snap_dir,
                          variables=['10u', '10v', 'msl', 'sst', 'slhf'],
                          dx=0.5, resx=11,
                          quiet=False):
    """
    Generate single-level meteorological snapshots using TempestExtremes for a given set of variables and time range.

    This function prepares and executes a TempestExtremes `NodeFileCompose` command to extract single-level data
    from ERA5 NetCDF files for a set of track points defined in a CSV file. It creates 
    snapshot files on a XY grid.

    XY: Cartesian stereographic projection with eastwards X coordinate vector and northwards Y coordinate vector.
        Grid spacing is equidistant.

    Parameters
    ----------
    csv_dir : str
        Directory to the input track file
    
    txt_dir : str
        Directory to store the generated input ERA5 file list
    
    snap_dir : str
        Output directory where the generated snapshot files will be stored.
    
    variables : list of str, optional
        List of ERA5 single-level variables to extract. Default includes:
        ['u10', 'v10', 'msl', 'sst', 'slhf']

    dx : float, optional
        Grid spacing in great circle degrees for output snapshot grid. Default is 0.5.

    resx : int, optional
        Number of grid cells in each coordinate direction on the XY. Default is 11.

    """
    
    # Generate date strings based on min/max time of the track
    df = pd.read_csv(csv_dir, low_memory=False)
    df['time'] = pd.to_datetime(df['time'])
    sta_date = df.time.min()
    end_date = df.time.max()
    date_strings = generate_date_strings(sta_date, end_date)

    # Set environment variable for TempestExtremes - change to where you installed TempestExtremes
    os.environ['TEMPESTEXTREMESDIR'] = '/home/565/cj0591/tempestextremes/bin'


    # ERA5 single level
    ERA5_SINGLELEVEL_SIR = '/g/data/rt52/era5/single-levels/reanalysis'

    # write txt input file list
    with open(txt_dir, 'w') as in_file:
        for string in date_strings:
            files = []
            for var in variables:
                file = f"{ERA5_SINGLELEVEL_SIR}/{var}/{string[0:4]}/{var}_era5_oper_sfc_{string}.nc"
                files.append(file)
            in_file.write(";".join(files) + "\n")

    # --var
    var_map = {
        '10u': 'u10',
        '10v': 'v10',
        'msl': 'msl',
        'sst': 'sst',
        'slhf': 'slhf'
    }
    var_str = ",".join([f'"{var_map[v]}"' for v in variables])
    varout_str = ",".join([f'"{var_map[v]}"' for v in variables])

    # Build and run TempestExtremes command
    composeNode_command = f"{os.environ['TEMPESTEXTREMESDIR']}/NodeFileCompose " \
                          f"--in_nodefile {csv_dir} --in_nodefile_type SN --in_fmt \"(auto)\" " \
                          f"--in_data_list {txt_dir} --snapshots --dx {dx} --resx {resx} " \
                          f"--out_grid \"XY\" --out_data {snap_dir} " \
                          f"--latname \"latitude\" --lonname \"longitude\" " \
                          f"--var {var_str} --varout {varout_str} "

    composeNode_process = subprocess.run(composeNode_command, shell=True, capture_output=True, text=True)
    stdout = composeNode_process.stdout
    stderr = composeNode_process.stderr

    if not quiet:
        return stdout, stderr

def create_snapshot_sfc_RAD(csv_dir, txt_dir, snap_dir,
                            variables=['10u', '10v', 'msl', 'sst', 'slhf'],
                            dx=0.5, resx=11, resa=32,
                            quiet=False):
    """
    Generate single-level meteorological snapshots using TempestExtremes for a given set of variables and time range.

    This function prepares and executes a TempestExtremes `NodeFileCompose` command to extract single-level data
    from ERA5 NetCDF files for a set of track points defined in a CSV file. It creates 
    snapshot files on a XY grid.

    XY: Cartesian stereographic projection with eastwards X coordinate vector and northwards Y coordinate vector.
        Grid spacing is equidistant.

    Parameters
    ----------
    csv_dir : str
        Directory to the input track file
    
    txt_dir : str
        Directory to store the generated input ERA5 file list
    
    snap_dir : str
        Output directory where the generated snapshot files will be stored.
    
    variables : list of str, optional
        List of ERA5 single-level variables to extract. Default includes:
        ['u10', 'v10', 'msl', 'sst', 'slhf']

    dx : float, optional
        Grid spacing in great circle degrees for output snapshot grid. Default is 0.5.

    resx : int, optional
        Number of grid cells in the radial direction on the RAD grid. Default is 11.

    resa : int, optional
        Number of grid cells in the azimuthal direction on the RAD grid. Default is 32.

    """
    
    # Generate date strings based on min/max time of the track
    df = pd.read_csv(csv_dir, low_memory=False)
    df['time'] = pd.to_datetime(df['time'])
    sta_date = df.time.min()
    end_date = df.time.max()
    date_strings = generate_date_strings(sta_date, end_date)

    # Set environment variable for TempestExtremes - change to where you installed TempestExtremes
    os.environ['TEMPESTEXTREMESDIR'] = '/home/565/cj0591/tempestextremes/bin'


    # ERA5 single level
    ERA5_SINGLELEVEL_SIR = '/g/data/rt52/era5/single-levels/reanalysis'

    # write txt input file list
    with open(txt_dir, 'w') as in_file:
        for string in date_strings:
            files = []
            for var in variables:
                file = f"{ERA5_SINGLELEVEL_SIR}/{var}/{string[0:4]}/{var}_era5_oper_sfc_{string}.nc"
                files.append(file)
            in_file.write(";".join(files) + "\n")

    # --var
    var_map = {
        '10u': 'u10',
        '10v': 'v10',
        'msl': 'msl',
        'sst': 'sst',
        'slhf': 'slhf'
    }
    var_str = ",".join([f'"{var_map[v]}"' for v in variables])
    varout_str = ",".join([f'"{var_map[v]}"' for v in variables])

    # Build and run TempestExtremes command
    composeNode_command = f"{os.environ['TEMPESTEXTREMESDIR']}/NodeFileCompose " \
                          f"--in_nodefile {csv_dir} --in_nodefile_type SN --in_fmt \"(auto)\" " \
                          f"--in_data_list {txt_dir} --snapshots --dx {dx} --resx {resx} --resa {resa} " \
                          f"--out_grid \"RAD\" --out_data {snap_dir} " \
                          f"--latname \"latitude\" --lonname \"longitude\" " \
                          f"--var {var_str} --varout {varout_str} "

    composeNode_process = subprocess.run(composeNode_command, shell=True, capture_output=True, text=True)
    stdout = composeNode_process.stdout
    stderr = composeNode_process.stderr

    if not quiet:
        return stdout, stderr

def create_snapshot_pl_XY(csv_dir, txt_dir, snap_dir,
                          variables=['u', 'v', 'w', 'z', 't', 'q', 'pv'],
                          dx=0.5, resx=11,
                          quiet=False):
    """
    Generate pressure-level meteorological snapshots using TempestExtremes for a given set of variables and time range.

    This function prepares and executes a TempestExtremes `NodeFileCompose` command to extract pressure-level data 
    (e.g., wind, temperature, PV) from ERA5 NetCDF files for a set of track points defined in a CSV file. It creates 
    snapshot files on a XY grid.

    XY: Cartesian stereographic projection with eastwards X coordinate vector and northwards Y coordinate vector.
        Grid spacing is equidistant.

    Parameters
    ----------
    csv_dir : str
        Directory to the input track file
    
    txt_dir : str
        Directory to store the generated input ERA5 file list
    
    snap_dir : str
        Output directory where the generated snapshot files will be stored.
    
    variables : list of str, optional
        List of ERA5 pressure-level variables to extract. Default includes:
        ['u', 'v', 'w', 'z', 't', 'q', 'pv']

    dx : float, optional
        Grid spacing in great circle degrees for output snapshot grid. Default is 0.5.

    resx : int, optional
        Number of grid cells in each coordinate direction on the XY. Default is 11.

    """
    
    # Generate date strings based on min/max time of the track
    df = pd.read_csv(csv_dir, low_memory=False)
    df['time'] = pd.to_datetime(df['time'])
    sta_date = df.time.min()
    end_date = df.time.max()
    date_strings = generate_date_strings(sta_date, end_date)

    # Set environment variable for TempestExtremes - change to where you installed TempestExtremes
    os.environ['TEMPESTEXTREMESDIR'] = '/home/565/cj0591/tempestextremes/bin'


    # ERA5 pressure level
    ERA5_PRESUELEVEL_SIR = '/g/data/rt52/era5/pressure-levels/reanalysis'

    # write txt input file list
    with open(txt_dir, 'w') as in_file:
        for string in date_strings:
            files = []
            for var in variables:
                file = f"{ERA5_PRESUELEVEL_SIR}/{var}/{string[0:4]}/{var}_era5_oper_pl_{string}.nc"
                files.append(file)
            in_file.write(";".join(files) + "\n")

    # --var
    var_str = ",".join([f'"{v}(:)"' for v in variables])
    varout_str = ",".join([f'"{v}"' for v in variables])


    # Build and run TempestExtremes command
    composeNode_command = f"{os.environ['TEMPESTEXTREMESDIR']}/NodeFileCompose " \
                          f"--in_nodefile {csv_dir} --in_nodefile_type SN --in_fmt \"(auto)\" " \
                          f"--in_data_list {txt_dir} --snapshots --dx {dx} --resx {resx} " \
                          f"--out_grid \"XY\" --out_data {snap_dir} " \
                          f"--latname \"latitude\" --lonname \"longitude\" " \
                          f"--var {var_str} --varout {varout_str} "

    composeNode_process = subprocess.run(composeNode_command, shell=True, capture_output=True, text=True)
    stdout = composeNode_process.stdout
    stderr = composeNode_process.stderr

    if not quiet:
        return stdout, stderr
    
def create_snapshot_pl_RAD(csv_dir, txt_dir, snap_dir,
                           variables=['u', 'v', 'w', 'z', 't', 'q', 'pv'],
                           dx=0.5, resx=11, resa=16,
                           quiet=False):
    """
    Generate pressure-level meteorological snapshots using TempestExtremes for a given set of variables and time range.

    This function prepares and executes a TempestExtremes `NodeFileCompose` command to extract pressure-level data 
    (e.g., wind, temperature, PV) from ERA5 NetCDF files for a set of track points defined in a CSV file. It creates 
    snapshot files on a RAD grid.

    RAD: Radial stereographic projection with azimuthal coordinate vector and radial coordinate vector.
         Azimuthal grid spacing is equiangular; radial grid spacing is equidistant.

    Parameters
    ----------
    csv_dir : str
        Directory to the input track file
    
    txt_dir : str
        Directory to store the generated input ERA5 file list
    
    snap_dir : str
        Output directory where the generated snapshot files will be stored.
    
    variables : list of str, optional
        List of ERA5 pressure-level variables to extract. Default includes:
        ['u', 'v', 'w', 'z', 't', 'q', 'pv']

    dx : float, optional
        Grid spacing in great circle degrees for output snapshot grid. Default is 0.5.

    resx : int, optional
        Number of grid cells in the radial direction on the RAD grid. Default is 11.

    resa : int, optional
        Number of grid cells in the azimuthal direction on the RAD grid. Default is 16.
    """
    
    # Generate date strings based on min/max time of the track
    df = pd.read_csv(csv_dir, low_memory=False)
    df['time'] = pd.to_datetime(df['time'])
    sta_date = df.time.min()
    end_date = df.time.max()
    date_strings = generate_date_strings(sta_date, end_date)

    # Set environment variable for TempestExtremes - change to where you installed TempestExtremes
    os.environ['TEMPESTEXTREMESDIR'] = '/home/565/cj0591/tempestextremes/bin'


    # ERA5 pressure level
    ERA5_PRESUELEVEL_SIR = '/g/data/rt52/era5/pressure-levels/reanalysis'

    # write txt input file list
    with open(txt_dir, 'w') as in_file:
        for string in date_strings:
            files = []
            for var in variables:
                file = f"{ERA5_PRESUELEVEL_SIR}/{var}/{string[0:4]}/{var}_era5_oper_pl_{string}.nc"
                files.append(file)
            in_file.write(";".join(files) + "\n")

    # --var
    var_str = ",".join([f'"{v}(:)"' for v in variables])
    varout_str = ",".join([f'"{v}"' for v in variables])


    # Build and run TempestExtremes command
    composeNode_command = f"{os.environ['TEMPESTEXTREMESDIR']}/NodeFileCompose " \
                          f"--in_nodefile {csv_dir} --in_nodefile_type SN --in_fmt \"(auto)\" " \
                          f"--in_data_list {txt_dir} --snapshots --dx {dx} --resx {resx} --resa {resa} " \
                          f"--out_grid \"RAD\" --out_data {snap_dir} " \
                          f"--latname \"latitude\" --lonname \"longitude\" " \
                          f"--var {var_str} --varout {varout_str} "

    composeNode_process = subprocess.run(composeNode_command, shell=True, capture_output=True, text=True)
    stdout = composeNode_process.stdout
    stderr = composeNode_process.stderr

    if not quiet:
        return stdout, stderr