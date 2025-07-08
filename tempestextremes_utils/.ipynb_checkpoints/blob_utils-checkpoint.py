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
from tempestextremes_utils.blob_wrappers import run_stitchBlobs
from gridding.utils import haversine

def separate_blobs_by_closest_cents(df):
    for b in df.bnum.unique():
        dtlist=df[df.bnum==b].datetime.unique()
        for idt,dt in enumerate(dtlist):
            df_dt=df[df.bnum==b][df[df.bnum==b].datetime==dt]
            if idt==0:
                for i,ind in enumerate(df_dt.index.values):
                   df.loc[ind, "track_id"] = i
            else:
                prev_dt=df[df.bnum==b][df[df.bnum==b].datetime==dtlist[idt-1]]
                closest_points=find_closest_points(prev_dt, df_dt)
                #if len(closest_points)>1:
                #    break
                for pid in closest_points.prev_index.unique():
                    min_dist=closest_points[closest_points.prev_index==pid]['distance_to_closest'].min()
                    min_dist_df=closest_points[closest_points['distance_to_closest']==min_dist]
                    if len(closest_points[closest_points.prev_index==pid])>1:
                        for cid in closest_points.current_index.unique():
                            if cid==min_dist_df.current_index.values:
                                df.loc[cid, "track_id"] = min_dist_df.closest_track_id.values
                            else:
                                df.loc[cid, "track_id"] = df[df.bnum==b]['track_id'].max()+1
                    else:
                        df.loc[closest_points[closest_points['prev_index']==pid].current_index.values[0], "track_id"] = closest_points[closest_points['prev_index']==pid].closest_track_id.values[0]
    
    df['bnum_merge'] = df.groupby(['bnum', 'track_id']).ngroup()
                        
    return df      

def find_closest_points(df1, df2):
    closest_points = []
    for i, row2 in df2.iterrows():
        min_distance = float('inf')
        closest_index_df1 = None
        
        for j, row1 in df1.iterrows():
            distance = haversine(row2['centlat'], row2['centlon'], row1['centlat'], row1['centlon'])
            if distance < min_distance:
                min_distance = distance
                closest_index_df1 = j
        
        # Add closest point data to df2 along with df2 index
        closest_points.append({
            "current_index": i,  # Index of the point in df2
            "closest_centlon": df1.loc[closest_index_df1, "centlon"],
            "closest_centlat": df1.loc[closest_index_df1, "centlat"],
            "closest_track_id": df1.loc[closest_index_df1, "track_id"],
            "prev_index": closest_index_df1,  # Index of closest point in df1
            "distance_to_closest": min_distance
        })
    
    # Create a new DataFrame with the closest points and their indices
    closest_df = pd.DataFrame(closest_points)
    
    return closest_df

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

    #print(np.min(non_bg_xarray1))
    #print(np.min(non_bg_xarray2))

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
        
        # Delete all local variables
        for var in list(locals().keys()):
            del locals()[var]

    # Retrieve unique labels from xarray1 (excluding background)
    unique_labels_1 = np.unique(non_bg_xarray1)

    # Parallelize the label mapping process
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_label_mapping, unique_labels_1)

    # Filter out None results and update the mapping dictionary
    for result in results:
        #print(result)
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

    #max_value = max(max(v) for v in mapping.values())+1
    #for new_ind,k in enumerate(range(int(xarray1.where(xarray1>0).max().values)+1,
    #               int(xarray3.where(xarray3>0).max().values)+1)): ### MB added 1 here. 
    #    mapping[k]=[max_value+new_ind]

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
            #print(old_label, new_label)
            mask = data == old_label  # Create a mask where the value is old_label
            data[mask] = new_label  # Set the new value for all positions that match
    
        # Return a new xarray with the updated values
        new_xarray = xarray3.copy()
        new_xarray.values = data
        
        return new_xarray

        # Delete all local variables
        for var in list(locals().keys()):
            del locals()[var]    

        del flat_mapping

    apply_mapping_to_xarray(xarray3, mapping).to_netcdf(stitchfile_t1_final)

    # Delete all local variables
    del mapping,label2_positions

def run_and_connect_stitchBlobs(detect_filelist,stitch_filelist,quiet=False,clean=False,
                                minsize=1,
                                min_overlap_prev=25.,max_overlap_prev=100.,
                                min_overlap_next=25.,max_overlap_next=100.,
                                lonname="longitude",latname="latitude"):

    if clean:
        path,_=os.path.split(stitch_filelist)
        delete_all_files(path,extension='.nc')
        delete_all_files(path,extension='log.txt')
        delete_all_files(path,extension='_temp.txt')

    stitch_filenames=read_filelist(stitch_filelist)
    detect_filenames=read_filelist(detect_filelist)
    for sf,s in tqdm(enumerate(stitch_filenames[0:-1]),total=len(stitch_filenames[0:-1])):
        detect_filenames_temp=detect_filenames[sf:sf+2]
        name, extension = os.path.splitext(detect_filelist) 
        detect_filelist_temp=name+'_temp'+extension
        write_to_filelist(detect_filenames_temp,detect_filelist_temp)
        stitch_filenames_final=stitch_filenames[sf:sf+2]
        stitch_filenames_temp=[f.split('.nc')[0]+'_temp.nc' for f in stitch_filenames_final]
        name, extension = os.path.splitext(stitch_filelist) 
        stitch_filelist_temp=name+'_temp'+extension
        write_to_filelist(stitch_filenames_temp,stitch_filelist_temp)
        
        result=run_stitchBlobs(detect_filelist_temp,stitch_filelist_temp,quiet=quiet,clean=False,mpi_np=1,
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

def lon_convert(lon):
    dist_from_180 = lon - 180.0
    return np.where(dist_from_180 < 0, lon, -(180 - dist_from_180))

def lon_convert2(lon):
    return np.where(lon < 0, 360 + lon, lon)

def merge_and_split_blob_dfs(df_stitch, df_nostitch, 
              timename='time',
              rfn="", textfn="", csvfn="", 
              df_merged_name="df_merged", 
              byvec=["datehour", "area", "var"]):
    df_comm = df_stitch.merge(df_nostitch, on=byvec)
    df_tot = df_stitch.merge(df_nostitch, on=byvec, how='outer')
    df_istot = df_tot[df_tot['bnum_y'].isna()]
    df_isnot = df_tot[df_tot['bnum_x'].isna()]
    
    df_comm["bnum2"] = df_comm["bnum_x"]
    df_tot["bnum2"] = df_tot["bnum_x"]
    df_istot["bnum2"] = df_istot["bnum_x"]
    df_isnot["bnum2"] = df_isnot["bnum_y"]

    names=list(df_stitch)
    for varname in byvec+['bnum','bnum_id']:
        names = [value_str for value_str in names if value_str != varname]
    
    for t in df_istot[timename].unique():
        df_check = df_istot[df_istot[timename] == t]
        df_otherblobs = df_isnot[df_isnot[timename] == t]
        
        if not df_otherblobs.empty:
            for _, row in df_check.iterrows():
                clatmin, clatmax = row["minlat_x"], row["maxlat_x"]
                clonmin, clonmax = row["minlon_x"], row["maxlon_x"]
                axis180 = (clonmin < 0 or clonmax < 0)
                
                if clonmin > clonmax:
                    per_bound = True
                    clonmin = lon_convert(clonmin) if not axis180 else lon_convert2(clonmin)
                    clonmax = lon_convert(clonmax) if not axis180 else lon_convert2(clonmax)
                else:
                    per_bound = False
                
                for _, blob in df_otherblobs.iterrows():
                    blatmin, blatmax = blob["minlat_y"], blob["maxlat_y"]
                    blonmin, blonmax = blob["minlon_y"], blob["maxlon_y"]
                    
                    if per_bound:
                        blonmin = lon_convert(blonmin) if not axis180 else lon_convert2(blonmin)
                        blonmax = lon_convert(blonmax) if not axis180 else lon_convert2(blonmax)
                    
                    if (blatmin >= clatmin and blatmax <= clatmax and
                        blonmin >= clonmin and blonmax <= clonmax):
                        for name in names:
                            blob[name+"_x"] = blob[name+"_y"]
                        blob["bnum_x"] = row["bnum_x"]
                        df_comm = pd.concat([df_comm, blob.to_frame().T], ignore_index=True)

    df_comm['bnum2']=df_comm['bnum_x']
    df_comm = df_comm.drop(columns=[col for col in df_comm.columns if col.endswith("_y")])# and col != "bnum_y"])
    df_comm.columns = df_comm.columns.str.replace(r"_x$|_y$", "", regex=True)
    df_final = df_comm[[*df_stitch.columns, "bnum2"]]
    df_final = df_final.sort_values(by=['bnum',timename])

    df_final = df_final.astype({'bnum':'int'})
    df_final = df_final.drop(columns=['bnum2'])

    df_final["merge_type"] = "None"
    df_final["track_id"] = 0

    for b in df_final.bnum.unique():
        df_temp=df_final[df_final.bnum==b]
        prev_len_temp=1
        for dt in df_temp[timename].unique():
            len_temp=len(df_temp[df_temp[timename]==dt])
            if len_temp>prev_len_temp:
                df_final.loc[((df_final.bnum==b) & (df_final[timename]==dt)),'merge_type']='split'
            if len_temp<prev_len_temp:
                df_final.loc[((df_final.bnum==b) & (df_final[timename]==dt)),'merge_type']='merged'    
            prev_len_temp=len_temp*1

    def insert_after(lst, target, value_to_insert):
        try:
            # Find the index of the target value
            target_index = lst.index(target)
            # Insert the new value after the target
            lst.insert(target_index + 1, value_to_insert)
        except ValueError:
            print(f"Value {target} not found in the list.")
        return lst

    new_col_order=insert_after(df_final.columns.tolist(),'bnum_id','track_id')
    new_col_order=new_col_order[0:-1]
    df_final=df_final[new_col_order]
        
    if rfn:
        df_final.to_pickle(rfn)
        print(f"Wrote {rfn} to file")
    if textfn:
        df_final.to_csv(textfn, sep='\t', index=False, quoting=3)
        print(f"Wrote {textfn} to file")
    if csvfn:
        df_final.to_csv(csvfn, index=False)
        print(f"Wrote {csvfn} to file")
    
    return df_final


