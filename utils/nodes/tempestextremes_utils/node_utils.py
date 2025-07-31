#!/usr/bin/env python
import os
import shutil
import subprocess
import xarray as xr
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils.general.nci_utils import get_GADI_ERA5_filename
from utils.general.date_utils import generate_datetimes_months
from utils.general.file_utils import write_to_filelist,create_directory,read_filelist


def create_Node_dirstruct(runpath,casename):
    #### Create the case directory ####
    casedir=os.path.join(runpath,casename)
    create_directory(casedir)
    #### Create the detectBlobs directory ####
    inputdir=os.path.join(runpath,casename,'input')
    create_directory(inputdir)
    #### Create the detectNodes directory ####
    detectNodesdir=os.path.join(runpath,casename,'detectNodes')
    create_directory(detectNodesdir)
    #### Create the detectNodes directory ####
    stitchNodesdir=os.path.join(runpath,casename,'stitchNodes')
    create_directory(stitchNodesdir)
    #### Create the detectNodes directory ####
    nodeFileComposedir=os.path.join(runpath,casename,'nodeFileCompose')
    create_directory(nodeFileComposedir)
    #### Create the detectNodes directory ####
    logsdir=os.path.join(runpath,casename,'logs')
    create_directory(logsdir)
    
    return casedir,inputdir,detectNodesdir,stitchNodesdir,logsdir

def run_detectNodes(input_filelist, detect_filelist, mpi_np=1,
                    searchby="min",
                    detect_var="msl",
                    merge_dist=6.0,
                    bounds=None,
                    closedcontour_commands="msl,200.0,5.5,0;_DIFF(z(300millibars),z(500millibars)),-58.8,6.5,1.0",
                    output_commands="msl,min,0;_VECMAG(u10,v10),max,2.0;zs,min,0",
                    timeinterval="6hr",
                    lonname="longitude",latname="latitude",
                    logdir="./log/",
                    regional=False,
                    quiet=False,
                    out_command_only=False):    
    """
    This is a short, concise summary of my_function.

    A more detailed explanation of what the function does.
    It can span multiple lines and describe the overall purpose.
    You might include examples of usage here.

    Parameters
    ----------
    param1 : int
        Description of the first parameter. It should be an integer
        representing some quantity.
    param2 : str
        Description of the second parameter. This is a string that
        could be a name or a path.
    optional_param : list of float, optional
        An optional parameter, defaults to None. This should be a list
        of floating-point numbers.

    Returns
    -------
    bool
        True if the operation was successful, False otherwise.
        A more detailed explanation of the return value can go here.

    Raises
    ------
    ValueError
        If `param1` is negative or `param2` is an empty string.
    TypeError
        If `param1` is not an integer.

    See Also
    --------
    another_function : Relevant function for related operations.
    some_class.method : A related method of a class.

    Notes
    -----
    This section can contain any additional information about the function,
    such as algorithms used, limitations, or best practices.
    It's good for conveying context not directly related to parameters or returns.

    Examples
    --------
    >>> my_function(10, "hello")
    True
    >>> my_function(5, "world", optional_param=[1.0, 2.5])
    True
    >>> my_function(-1, "test")
    Traceback (most recent call last):
        ...
    ValueError: param1 cannot be negative.
    """

    # DetectNode command
    detectNode_command = ["mpirun", "-np", f"{int(mpi_np)}",
                            f"{os.environ['TEMPESTEXTREMESDIR']}/DetectNodes",
                            "--in_data_list",f"{input_filelist}",
                            "--out_file_list", f"{detect_filelist}",
                            f"--searchby{searchby}",f"{detect_var}",
                            "--closedcontourcmd",f"{closedcontour_commands}",
                            "--mergedist",f"{merge_dist}",
                            "--outputcmd",f"{output_commands}",
                            "--timefilter",f"{timeinterval}",
                            "--latname",f"{latname}",
                            "--lonname",f"{lonname}",
                            "--logdir",f"{logdir}",
                            ]

                            	
    if regional:
        detectNode_command=detectNode_command+["--regional"]
    if bounds is not None:
        detectNode_command=detectNode_command+["--minlon",f"{bounds[0]}","--maxlon",f"{bounds[1]}","--minlat",f"{bounds[2]}","--maxlat",f"{bounds[3]}"]


    ### Here I prepare the command to printed out ready for the terminal as I need to add in some "" for some variables
    printed_command=detectNode_command.copy()
    indx=np.where(np.array(printed_command)==f"--searchby{searchby}")[0][0]+1
    printed_command[indx]=[f'"{s}"' for s in [printed_command[indx]]][0]
    indx=np.where(np.array(printed_command)=='--closedcontourcmd')[0][0]+1
    printed_command[indx]=[f'"{s}"' for s in [printed_command[indx]]][0]
    indx=np.where(np.array(printed_command)=='--outputcmd')[0][0]+1
    printed_command[indx]=[f'"{s}"' for s in [printed_command[indx]]][0]
    indx=np.where(np.array(printed_command)=='--timefilter')[0][0]+1
    printed_command[indx]=[f'"{s}"' for s in [printed_command[indx]]][0]
    
    print(*printed_command)
    if not out_command_only:
        print('Running command ...')
        detectNode_process = subprocess.Popen(detectNode_command,
                                              stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE, text=True)

        # Wait for the process to complete and capture output
        stdout, stderr = detectNode_process.communicate()
    
        #path,_=os.path.split(input_filelist)
        outfile=logdir+'/detectNodes_outlog.txt'
        with open(outfile, 'w') as file:
            file.write(stdout)
        outfile=logdir+'/detectNodes_errlog.txt'
        with open(outfile, 'w') as file:
            file.write(stderr)
        if not quiet:
             return stdout, stderr

def run_stitchNodes(input_filelist, stitch_file, mpi_np=1,
                    output_filefmt="csv",
                    in_fmt_commands="lon,lat,msl",
                    range_dist=8.0,
                    minim_time="54h",
                    maxgap_time="24h",
                    min_endpoint_dist=12.0,
                    threshold_condition="wind,>=,10.0,10;lat,<=,50.0,10;lat,>=,-50.0,10;zs,<,150,10",
                    quiet=False,
                    out_command_only=False):

    """
    input_filelist --- text file contain a list of DetectNode output files
    stitch_file --- a single text file with stitched nodes
    output_filefmt --- output format (gfdl|csv|csvnohead)
    range_dist --- the distance between two nodes in two consecutive time steps
                   adjust this parameter according to timeinterval in run_detectNodes
                   at least >= merge_dist in run_detectNodes
    minim_time --- the minimum time detected nodes must sustain
    maxgap_time --- the maximum gap within the track
    threshold_var --- threshold variable
    threshold_op --- threshold operation (>|<|==|!=)
    threshold_val --- threshold value
    threshold_time --- the numbe of timesteps threshold must be satisfied
    """

    # StitchNode command
    stitchNode_command = ["mpirun", "-np", f"{int(mpi_np)}",
                             f"{os.environ['TEMPESTEXTREMESDIR']}/StitchNodes",
                             "--in_list",f"{input_filelist}",
                             "--in_fmt",f"{in_fmt_commands}",
                             "--range",f"{range_dist}",
                             "--mintime",f"{minim_time}",
                             "--maxgap",f"{maxgap_time}",
                             "--threshold",f"{threshold_condition}",
                             "--min_endpoint_dist",f"{min_endpoint_dist}",
                             "--out_file_format",f"{output_filefmt}",
                             "--out", f"{stitch_file}"
                             ]
    
    print(*stitchNode_command)
    
    if not out_command_only:
        stitchNode_process = subprocess.Popen(stitchNode_command,
                                              stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE, text=True)

        # Wait for the process to complete and capture output
        stdout, stderr = stitchNode_process.communicate()

        path,_=os.path.split(input_filelist)
        outfile=path+'/stitchNodes_outlog.txt'
        with open(outfile, 'w') as file:
            file.write(stdout)
        outfile=path+'/stitchNodes_errlog.txt'
        with open(outfile, 'w') as file:
            file.write(stderr)
        if not quiet:
             return stdout, stderr

def run_nodeCompose(stitch_file, compose_file, in_fmt, 
                    level_type, grid_type='XY',
                    variables=['u','v'],
                    lonname="longitude",latname="latitude",
                    dx=0.5, resx=11,
                    quiet=False,
                    out_command_only=False):
    """
    Parameters
    ----------
    stitch_file : dtype str
        String with a path to the csv file of the stitch node output data.
    compose_file : dtype str
        String with a path to the netcdf file of the compose node output data.
    level_type : dtype str
        level type of era5 data ['pressure-levels', 'single-levels', 'potential-temperature']
    grid_type : dtype str
        grid type of the output data default='XY'
    variables : list (N=n)
        a list containing names of variables in the form of ['u', 'v']
    lonname : dtype str
        String with the longitude variable name, as in the input netcdfs
    latname : dtype str
        String with the latitude variable name, as in the input netcdfs
    dx: dtype float
        Horizontal grid spacing of the XY grid
    resx: dtype int
        Number of grid ceels in each coordiante on the XY grid
    quiet : bool
        *Optional*, default ``False``. If ``True``, progress information is suppressed.
    out_command_only : bool
        *Optional*, default ``False``. If ``True``, will not run the TE command but instead with output the command for terminal use.
    
    """    
    # create a "time" column
    df = pd.read_csv(stitch_file, skipinitialspace=True)
    df['time'] = pd.to_datetime({
        'year':  df.year,
        'month': df.month,
        'day':   df.day,
        'hour':  df.hour
    })
    
    # extract start and end date
    sta_date = df.time.min()
    end_date = df.time.max()
    months_list = generate_datetimes_months(sta_date,end_date,interval=1)
    
    # input filelist
    inputlist = os.path.dirname(compose_file) + os.sep + 'inputlist.txt'
    with open(inputlist, 'w') as in_file:
        for m in months_list:
            files = []
            for var in variables:
                file = get_GADI_ERA5_filename(var,m,stream='hourly',level_type=level_type)
                files.append(file)
            in_file.write(";".join(files) + "\n")
    
    # --var
    if level_type == 'single-levels':
        varin_str = ",".join([f'{v}' for v in variables])
    if level_type == 'pressure-levels' or level_type == 'potential-temperature':
        varin_str = ",".join([f'{v}(:)' for v in variables])
    varout_str = ",".join([f'{v}' for v in variables])
    
    composeNode_command = [f"{os.environ['TEMPESTEXTREMESDIR']}/NodeFileCompose",
                            "--in_nodefile",f"{stitch_file}",
                            "--in_nodefile_type", "SN",
                            "--in_fmt",f"{in_fmt}",
                            "--snapshots",
                            "--in_data_list",f"{inputlist}",
                            "--dx",f"{dx}",
                            "--resx",f"{resx}",
                            "--out_grid",f"{grid_type}",
                            "--out_data",f"{compose_file}",
                            "--latname",f"{latname}",
                            "--lonname",f"{lonname}",
                            "--var",f"{varin_str}",
                            "--varout",f"{varout_str}",
                            ]
    print(*composeNode_command)
    
    if not out_command_only:
        composeNode_process = subprocess.Popen(composeNode_command,
                                               stdout=subprocess.PIPE, 
                                               stderr=subprocess.PIPE, text=True)
    
        # Wait for the process to complete and capture output
        stdout, stderr = composeNode_process.communicate()
    
        path,_=os.path.split(compose_file)
        outfile=path+'/composeNode_outlog.txt'
        with open(outfile, 'w') as file:
            file.write(stdout)
        outfile=path+'/composeNode_errlog.txt'
        with open(outfile, 'w') as file:
            file.write(stderr)
        if not quiet:
             return stdout, stderr
    
    os.remove(inputlist)

def assign_lev(ds, level_type):
    if level_type == 'pressure-levels':
        level_range = np.array([   1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,
                                 125,  150,  175,  200,  225,  250,  300,  350,  400,  450,  500,
                                 550,  600,  650,  700,  750,  775,  800,  825,  850,  875,  900,
                                 925,  950,  975, 1000])
    if level_type == 'potential-temperature':
        level_range = np.array([265, 275, 285, 300, 315, 320, 330, 350, 370, 395, 430, 475, 530, 600, 700, 850])
    ds_lev = ds.assign_coords(level=('level', level_range))
    return ds_lev
