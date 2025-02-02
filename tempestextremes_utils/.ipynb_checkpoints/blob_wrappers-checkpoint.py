#!/usr/bin/env python
import os
import shutil
import subprocess
import xarray as xr
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils.file_utils import write_to_filelist,create_directory,read_filelist,delete_all_files,delete_file

def create_Blob_dirstruct(runpath,casename):
    #### Create the case directory ####
    create_directory(runpath+casename)
    #### Create the detectBlobs directory ####
    create_directory(runpath+casename+'/detectBlobs')
    #### Create the stitchBlobs directory ####
    create_directory(runpath+casename+'/stitchBlobs')
    #### Create the statBlobs directory ####
    create_directory(runpath+casename+'/statBlobs')

def run_detectBlobs(input_filelist,detect_filelist,quiet=False,clean=False,mpi_np=1,
                    threshold_var="z",threshold_op=">=",threshold_val=1000.0,
                    threshold_dist=0.,geofilterarea_km2=0.0,
                    lonname="longitude",latname="latitude"):

    detectBlob_command = ["mpirun", "-np", f"{int(mpi_np)}",
                            f"{os.environ['TEMPESTEXTREMESDIR']}/DetectBlobs", 
                            "--in_data_list",f"{input_filelist}",
                            "--thresholdcmd",f"{threshold_var},{threshold_op},{threshold_val},{threshold_dist}",
                            "--geofiltercmd", f"area,>=,{geofilterarea_km2}km2",
                            "--timefilter", f"6hr",
                            "--latname", f"{latname}", 
                            "--lonname", f"{lonname}",
                            "--out_list", f"{detect_filelist}"
                            ]

    if clean:
        path,_=os.path.split(detect_filelist)
        delete_all_files(path,extension='.nc')
        delete_all_files(path,extension='log.txt')
        
    # Run the command asynchronously
    process = subprocess.Popen(detectBlob_command, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, text=True)
    
    # Wait for the process to complete and capture output
    stdout, stderr = process.communicate()

    path,_=os.path.split(detect_filelist)
    outfile=path+'/detectBlobs_outlog.txt'
    with open(outfile, 'w') as file:
        file.write(stdout)
    outfile=path+'/detectBlobs_errlog.txt'
    with open(outfile, 'w') as file:
        file.write(stderr)
    
    if not quiet:
        return stdout, stderr

def run_stitchBlobs(detect_filelist,stitch_filelist,quiet=False,clean=False,mpi_np=1,
                    minsize=1,mintime=1,
                    minlat=None,maxlat=None,
                    min_overlap_prev=25.,max_overlap_prev=100.,
                    min_overlap_next=25.,max_overlap_next=100.,
                    lonname="longitude",latname="latitude"):

    stitchBlob_command =["mpirun", "-np", f"{int(mpi_np)}",
                            f"{os.environ['TEMPESTEXTREMESDIR']}/StitchBlobs", 
                            "--in_list",f"{detect_filelist}",
                            "--minsize", f"{minsize}",
                            "--mintime", f"{mintime}",
                            "--min_overlap_prev", f"{min_overlap_prev}","--max_overlap_prev", f"{max_overlap_prev}",
                            "--min_overlap_next", f"{min_overlap_next}","--max_overlap_next", f"{max_overlap_next}",
                            "--latname", f"{latname}", 
                            "--lonname", f"{lonname}",
                            "--out_list", f"{stitch_filelist}"
                            ]
    if minlat:
        stitchBlob_command=stitchBlob_command+["--minlat", f"{minlat}"]
    if maxlat:
        stitchBlob_command=stitchBlob_command+["--maxlat", f"{maxlat}"]

    if clean:
        path,_=os.path.split(stitch_filelist)
        delete_all_files(path,extension='.nc')
        delete_all_files(path,extension='log.txt')

    # Run the command asynchronously
    process = subprocess.Popen(stitchBlob_command, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, text=True)
    
    # Wait for the process to complete and capture output
    stdout, stderr = process.communicate()

    path,_=os.path.split(stitch_filelist)
    outfile=path+'/stitchBlobs_outlog.txt'
    with open(outfile, 'w') as file:
        file.write(stdout)
    outfile=path+'/stitchBlobs_errlog.txt'
    with open(outfile, 'w') as file:
        file.write(stderr)
    
    if not quiet:
        print(stdout)
        print(stderr)

def run_statBlobs(stitch_filelist,stat_file,quiet=False,clean=False,
                  var='object_id',lonname="longitude",latname="latitude",
                  findblobs=False, 
                  outstats='minlat,maxlat,minlon,maxlon,centlon,centlat,area'):

    statBlob_command =["mpirun", "-np", "1",
                            f"{os.environ['TEMPESTEXTREMESDIR']}/BlobStats", 
                            "--in_list",f"{stitch_filelist}",
                            "--out", f"{outstats}",
                            "--var", f"{var}",
                            "--out_headers",  "--out_fulltime",
                            "--latname", f"{latname}", 
                            "--lonname", f"{lonname}",
                            "--out_file", f"{stat_file}"
                            ]

    if findblobs:
        statBlob_command=statBlob_command+["--findblobs"]
        
    if clean:
        delete_file(stat_file)
        path,_=os.path.split(stat_file)
        delete_all_files(path,extension='log.txt')
                    
    # Run the command asynchronously
    process = subprocess.Popen(statBlob_command, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, text=True)
    
    # Wait for the process to complete and capture output
    stdout, stderr = process.communicate()

    path,_=os.path.split(stat_file)
    outfile=path+'/statBlobs_outlog.txt'
    with open(outfile, 'w') as file:
        file.write(stdout)
    outfile=path+'/statBlobs_errlog.txt'
    with open(outfile, 'w') as file:
        file.write(stderr)
    
    if not quiet:
        print(stdout)
        print(stderr)

def read_statBlobs(stat_file):
    #### Open to get the headers out ###
    df = pd.read_csv(stat_file)
    headers = df.columns.tolist()
    #### Now read file with the full headers ###
    df = pd.read_csv(stat_file, skiprows=1, sep='\s+', names=['bnum','bnum_id']+headers)

    def convert_datetime_with_time_offset(row):
        date_part = row.split('-')[0:3]
        offset_seconds = int(row.split('-')[3])
        # Combine date and time as datetime
        date = '-'.join(date_part)
        time = pd.to_datetime(date)
        # Adjust for the UTC offset
        adjusted_time = time + pd.Timedelta(seconds=offset_seconds)
        
        return adjusted_time#.strftime('%Y-%m-%d %H:%M:%S')

    def insert_after(lst, target, value_to_insert):
        try:
            # Find the index of the target value
            target_index = lst.index(target)
            # Insert the new value after the target
            lst.insert(target_index + 1, value_to_insert)
        except ValueError:
            print(f"Value {target} not found in the list.")
        return lst

    new_col_order=insert_after(df.columns.tolist(),'time','datetime')
    df['datetime'] = df['time'].apply(convert_datetime_with_time_offset)
    
    df = df[new_col_order]

    return df