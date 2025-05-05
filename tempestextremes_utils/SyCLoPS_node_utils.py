#!/usr/bin/env python
import os
import shutil
import subprocess
import xarray as xr
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils.file_utils import write_to_filelist,create_directory,read_filelist


def run_detectNodes(input_filelist, detect_filelist, mpi_np=4,
                    detect_var="msl",merge_dist=6.0,
                    detect_dist=5.5,
                    detect_delta=10,
                    detect_minmaxdist=0,
                    timeinterval="6hr",
                    lonname="longitude",latname="latitude",
                    quiet=False):
    """
    input_filelist --- text file contain a list of required input netcdf files
    detect_filelist --- text file contain a list of names of output files
    mpi_np --- mpi parallel running
    detect_var --- variable used to detect nodes
    merge_dist --- merge distance, if detected nodes wihtin this distance of each other merge into a single node
    detect_dist --- the distance from the detect node
    detect_delta --- the increase/decrease of detect variable from the detected node
    timeinterval --- time interval (1hr|6hr|12hr)
    lonname --- longitude name
    latname --- latitude name
    """

    # These parameters need in cyclone classification
    # For details see Han and Ullrich (2025) https://doi.org/10.1029/2024JD041287
    # Change variables names according to input reanalysis data
    output_commands = "msl,min,0;"\
                      "_VECMAG(u10,v10),max,2.0;"\
                      "msl,posclosedcontour,2.0,0;"\
                      "msl,posclosedcontour,5.5,0;"\
                      "_DIFF(_VECMAG(u(200millibars),v(200millibars)),_VECMAG(u(850millibars),v(850millibars))),avg,10.0;"\
                      "_DIFF(z(300millibars),z(500millibars)),negclosedcontour,6.5,1.0;"\
                      "_DIFF(z(500millibars),z(700millibars)),negclosedcontour,3.5,1.0;"\
                      "_DIFF(z(700millibars),z(925millibars)),negclosedcontour,3.5,1.0;"\
                      "z(500millibars),posclosedcontour,3.5,1.0;"\
                      "vo(500millibars),avg,2.5;"\
                      "r(100millibars),max,2.5;"\
                      "r(850millibars),avg,2.5;"\
                      "t(850millibars),max,0.0;"\
                      "z(850millibars),min,0;"\
                      "zs,min,0;"\
                      "u(850millibars),posminusnegwtarea,5.5;"\
                      "_VECMAG(u(200millibars),v(200millibars)),maxpoleward,1.0"

    # DetectNode command
    detectNode_command = ["mpirun", "-np", f"{int(mpi_np)}",
                            f"{os.environ['TEMPESTEXTREMESDIR']}/DetectNodes",
                            "--in_data_list",f"{input_filelist}",
                            "--out_file_list", f"{detect_filelist}",
                            "--searchbymin",f"{detect_var}",
                            "--closedcontourcmd",f"{detect_var},{detect_delta},{detect_dist},{detect_minmaxdist}",
                            "--mergedist",f"{merge_dist}",
                            "--outputcmd",f"{output_commands}",
                            "--timefilter",f"{timeinterval}",
                            "--latname",f"{latname}",
                            "--lonname",f"{lonname}"
                            ]
    
    detectNode_process = subprocess.Popen(detectNode_command,
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE, text=True)
    
    # Wait for the process to complete and capture output
    stdout, stderr = detectNode_process.communicate()

    path,_=os.path.split(input_filelist)
    outfile=path+'/detectNodes_outlog.txt'
    with open(outfile, 'w') as file:
        file.write(stdout)
    outfile=path+'/detectNodes_errlog.txt'
    with open(outfile, 'w') as file:
        file.write(stderr)
    if not quiet:
         return stdout, stderr

def run_stitchNodes(input_filelist, stitch_file, mpi_np=1,
                    output_filefmt="txt",
                    range_dist=6.0,
                    minim_time="18h",
                    maxgap_time="12h",
                    threshold_var="MSLPCC55",
                    threshold_op=">=",
                    threshold_val=100.0,
                    threshold_time=5,
                    quiet=False):
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

    # These command corresponding to the output_commands in the run_detectNodes
    in_fmt_commands = "lon,lat,MSLP,WS,MSLPCC20,MSLPCC55,DEEPSHEAR,UPPTKCC,MIDTKCC,LOWTKCC,Z500CC,VO500AVG,RH100MAX,RH850AVG,T850,Z850,ZS,U850DIFF,WS200PMX"

    # StitchNode command
    stitchNode_command = ["mpirun", "-np", f"{int(mpi_np)}",
                             f"{os.environ['TEMPESTEXTREMESDIR']}/StitchNodes",
                             "--in_list",f"{input_filelist}",
                             "--in_fmt",f"{in_fmt_commands}",
                             "--range",f"{range_dist}",
                             "--mintime",f"{minim_time}",
                             "--maxgap",f"{maxgap_time}",
                             "--threshold",f"{threshold_var},{threshold_op},{threshold_val},{threshold_time}",
                             "--out_file_format",f"{output_filefmt}",
                             "--out", f"{stitch_file}"
                             ]
    
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
