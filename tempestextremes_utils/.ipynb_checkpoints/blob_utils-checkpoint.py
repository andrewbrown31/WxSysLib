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


def create_Blob_dirstruct(runpath,casename):
    #### Create the case directory ####
    create_directory(runpath+casename)
    #### Create the detectBlobs directory ####
    create_directory(runpath+casename+'/detectBlobs')
    #### Create the stitchBlobs directory ####
    create_directory(runpath+casename+'/stitchBlobs')
    #### Create the statBlobs directory ####
    create_directory(runpath+casename+'/statBlobs')

def run_detectBlobs(input_filelist,detect_filelist,quiet=False,mpi_np=1,
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

def run_stitchBlobs(detect_filelist,stitch_filelist,quiet=False,mpi_np=1,
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

def connect_stitchBlobs(stitchfile_t0_temp, stitchfile_t0_final, stitchfile_t1_temp, stitchfile_t1_final,var='object_id'):
    """
    Map labels from xarray1 to corresponding labels in xarray2 efficiently using vectorized operations
    and parallel processing. Unmapped labels from xarray1 are assigned a new unique label starting
    from max(xarray2) + 1.

    Parameters:
    - xarray1: xarray.DataArray containing labels (3D)
    - xarray2: xarray.DataArray containing labels (3D)

    Returns:
    - mapping: Dictionary where keys are labels from xarray1 and values are the corresponding labels in xarray2,
               with unmapped labels starting from max(xarray2) + 1.
    """

    xarray1=xr.open_dataset(stitchfile_t0_temp)[var]
    xarray2=xr.open_dataset(stitchfile_t0_final)[var]
    xarray3=xr.open_dataset(stitchfile_t1_temp)[var]
    
    #print('Mapping indices...')
    # Ensure the arrays have the same shape
    if xarray1.shape != xarray2.shape:
        raise ValueError("Both xarrays must have the same shape.")

    # Flatten the arrays
    flat_xarray1 = xarray1.values.ravel()
    flat_xarray2 = xarray2.values.ravel()

    # Mask to exclude background (label 0)
    mask = flat_xarray1 != 0

    # Create a dictionary to store label mappings
    mapping = {}

    # Extract only the non-background values
    non_bg_xarray1 = flat_xarray1[mask]
    non_bg_xarray2 = flat_xarray2[mask]

    # Create a hashmap (dictionary) for the unique labels in xarray2 for fast lookup
    label2_positions = {}
    for idx, label2 in enumerate(non_bg_xarray2):
        if label2 != 0:
            if label2 not in label2_positions:
                label2_positions[label2] = []
            label2_positions[label2].append(idx)

    # Use parallel processing to speed up the processing of label mappings
    def process_label_mapping(label1):
        # Find all positions of label1 in xarray1
        label1_positions = np.where(non_bg_xarray1 == label1)[0]
        
        # Get corresponding labels in xarray2
        corresponding_labels = non_bg_xarray2[label1_positions]

        # Get unique corresponding labels in xarray2 (excluding background)
        unique_labels_2 = np.unique(corresponding_labels)
        unique_labels_2 = unique_labels_2[unique_labels_2 != 0]

        # Store the mapping if we found any corresponding labels
        if unique_labels_2.size > 0:
            return label1, unique_labels_2.tolist()
        return None

    # Retrieve unique labels from xarray1 (excluding background)
    unique_labels_1 = np.unique(non_bg_xarray1)

    # Parallelize the label mapping process
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_label_mapping, unique_labels_1)

    # Filter out None results and update the mapping dictionary
    for result in results:
        if result:
            mapping[result[0]] = result[1]

    # Find the maximum label in xarray2 (excluding background)
    max_label_xarray2 = np.max(non_bg_xarray2)

    # Start new label assignment from max_label_xarray2 + 1
    new_label_start = max_label_xarray2 + 1

    # Assign new labels to unmapped values from xarray1
    for label1 in unique_labels_1:
        if label1 not in mapping:
            mapping[label1] = [new_label_start]
            new_label_start += 1

    max_value = max(max(v) for v in mapping.values())+1
    for new_ind,k in enumerate(range(int(xarray1.where(xarray1>0).max().values)+1,
                   int(xarray3.where(xarray3>0).max().values))):
        mapping[k]=[max_value+new_ind]

    def apply_mapping_to_xarray(xarray3, mapping):
        """
        Apply a label mapping to xarray3 efficiently using vectorized operations.
    
        Parameters:
        - xarray3: xarray.DataArray containing the labels to be mapped (3D).
        - mapping: Dictionary where keys are the original labels and values are the new labels.
    
        Returns:
        - new_xarray: xarray.DataArray with the mapped labels.
        """
        # Convert xarray3 values to a numpy array for faster operations
        data = xarray3.values
    
        # Prepare a dictionary of old label -> new label mappings for quick lookup
        flat_mapping = {}
        for old_label, new_labels in mapping.items():
            for new_label in new_labels:
                flat_mapping[old_label] = new_label  # Override existing mapping to ensure unique label
    
        # Vectorized approach: Apply the new labels using np.isin()
        for old_label, new_label in flat_mapping.items():#tqdm(flat_mapping.items()):
            mask = data == old_label  # Create a mask where the value is old_label
            data[mask] = new_label  # Set the new value for all positions that match
    
        # Return a new xarray with the updated values
        new_xarray = xarray3.copy()
        new_xarray.values = data
        
        return new_xarray

    #print('Creating new index stitchBlobs...')

    apply_mapping_to_xarray(xarray3, mapping).to_netcdf(stitchfile_t1_final)

def run_and_connect_stitchBlobs(detect_filelist,stitch_filelist,quiet=False,
                                minsize=1,
                                min_overlap_prev=25.,max_overlap_prev=100.,
                                min_overlap_next=25.,max_overlap_next=100.,
                                lonname="longitude",latname="latitude"):
    stitch_filenames=read_filelist(stitch_filelist)
    detect_filenames=read_filelist(detect_filelist)
    for sf,s in tqdm(enumerate(stitch_filenames[0:-1]),total=len(stitch_filenames[0:-1])):
        #print(s)
        detect_filenames_temp=detect_filenames[sf:sf+2]
        name, extension = os.path.splitext(detect_filelist) 
        detect_filelist_temp=name+'_temp'+extension
        write_to_filelist(detect_filenames_temp,detect_filelist_temp)
        stitch_filenames_final=stitch_filenames[sf:sf+2]
        stitch_filenames_temp=[f.split('.nc')[0]+'_temp.nc' for f in stitch_filenames_final]
        name, extension = os.path.splitext(stitch_filelist) 
        stitch_filelist_temp=name+'_temp'+extension
        write_to_filelist(stitch_filenames_temp,stitch_filelist_temp)
    
        #print('Running stitchBlobs...')
        result=run_stitchBlobs(detect_filelist_temp,stitch_filelist_temp,quiet=False,
                           min_overlap_prev=min_overlap_prev,max_overlap_prev=max_overlap_prev,
                           min_overlap_next=min_overlap_next,max_overlap_next=max_overlap_next,
                           minsize=minsize,mintime=1,
                           lonname="longitude",latname="latitude")
    
        if sf == 0:
            shutil.copyfile(stitch_filenames_temp[0], stitch_filenames_final[0])
            shutil.copyfile(stitch_filenames_temp[1], stitch_filenames_final[1])
        else:
            connect_stitchBlobs(stitch_filenames_temp[0], 
                                stitch_filenames_final[0], 
                                stitch_filenames_temp[1], 
                                stitch_filenames_final[1],
                                var='object_id')
            
        os.remove(stitch_filenames_temp[0])
        os.remove(stitch_filenames_temp[1])

def run_statBlobs(stitch_filelist,stat_file,quiet=False,
                  var='object_id',lonname="longitude",latname="latitude",
                  outstats='minlat,maxlat,minlon,maxlon,centlon,centlat,area'):

    statBlob_command =f"{os.environ['TEMPESTEXTREMESDIR']}/BlobStats "\
                        f" --in_list \"{stitch_filelist}\" "\
                        f" --out \"{outstats}\" " \
                        f" --var \"{var}\" " \
                        f" --out_headers --out_fulltime "\
                        f" --latname \"{latname}\" --lonname \"{lonname}\" " \
                        f" --out_file \"{stat_file}\" "
                    
    statBlob_result = subprocess.run(statBlob_command, shell=True, capture_output=True, text=True)

    if not quiet:
        return statBlob_result

def read_statBlobs(stat_file):
    #### Open to get the headers out ###
    df = pd.read_csv(stat_file)
    headers = df.columns.tolist()
    #### Now read file with the full headers ###
    df = pd.read_csv(stat_file, skiprows=1, sep='\s+', names=['track_id','object_id']+headers)

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

