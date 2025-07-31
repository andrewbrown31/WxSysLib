#!/usr/bin/env python

import os
import xarray as xr
from tqdm import tqdm

# Create a single directory
def create_directory(dir_name):
    """
    This is the docstring.
    It should show up.
    """
    # ... function body
    try:
        os.mkdir(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

def delete_file(file_path):
    """Deletes the specified file if it exists."""
    if os.path.isfile(file_path):  # Check if it's a valid file
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    else:
        print(f"File '{file_path}' not found.")

def delete_all_files(directory, extension=None):
    """
    Deletes all files in the specified directory.
    
    Parameters:
        directory (str): The target directory.
        extension (str, optional): If provided, only deletes files with this extension (e.g., '.txt').
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        if os.path.isfile(file_path):  # Ensure it's a file
            if extension is None or file_name.endswith(extension):  # Check extension if specified
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

def write_to_filelist(infilenames,outfile):
    with open(outfile, 'w') as file:
        for infilename in infilenames:
            file.write(infilename + '\n')

def df_to_statfile(df,filename):
    # Write the header row with commas
    with open(filename, "w") as f:
        f.write(",".join(df.columns) + "\n")  # Write header row with commas
    
    # Append the data with tab separation
    df.to_csv(filename, sep="\t", mode="a", index=False, header=False)

def read_filelist(file_path):
    """
    Read a text file into a list, removing any blank entries.

    Parameters:
    - file_path: Path to the text file.

    Returns:
    - A list of non-blank lines from the file.
    """
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def split_netcdf_by_year_month(input_file, output_dir, basename='test'):
    """
    Split a NetCDF file into yearly and monthly chunks, and save each chunk as a separate NetCDF file.
    The output filenames are based on the input filename with the year and month appended.

    Parameters:
    - input_file: Path to the input NetCDF file.
    - output_dir: Directory where the output NetCDF files will be saved.
    """

    # Open the NetCDF file
    ds = xr.open_dataset(input_file)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Group by year and month
    for year, year_group in ds.groupby("time.year"):
        for month, month_ds in year_group.groupby("time.month"):
            # Create the output filename
            output_file = os.path.join(output_dir, f"{basename}_{year}{month:02d}.nc")
            
            # Save the monthly data to a NetCDF file
            month_ds.to_netcdf(output_file)
            print(f"Saved {year}-{month:02d} to {output_file}")

    # Close the dataset
    ds.close()

def split_xarray_by_year_month(ds, output_dir, basename='test'):
    """
    Split a NetCDF file into yearly and monthly chunks, and save each chunk as a separate NetCDF file.
    The output filenames are based on the input filename with the year and month appended.

    Parameters:
    - input_file: Path to the input NetCDF file.
    - output_dir: Directory where the output NetCDF files will be saved.
    """

    # Open the NetCDF file
    #ds = xr.open_dataset(input_file)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by year and month
    for year, year_group in ds.groupby("time.year"):
        for month, month_ds in year_group.groupby("time.month"):
            # Create the output filename
            output_file = os.path.join(output_dir, f"{basename}_{year}{month:02d}.nc")
            
            # month_ds = month_ds.compute()
            # Save the monthly data to a NetCDF file
            month_ds.to_netcdf(output_file)
            print(f"Saved {year}-{month:02d} to {output_file}")

def write_xarray_to_nc(ds: xr.Dataset, out_file: str) -> None:
    enc = {}

    for k in ds.data_vars:
        if ds[k].ndim < 2:
            continue

        enc[k] = {
            "zlib": True,
            "complevel": 9,
            "fletcher32": True,
            "chunksizes": tuple(map(lambda x: x//2, ds[k].shape))
        }

    ds.to_netcdf(out_file, format="NETCDF4", engine="netcdf4", encoding=enc)

def compress_files(detect_filenames,keep_old=False):
    for fn in tqdm(detect_filenames):
        path,f=os.path.split(fn)
        f_noext,ext=os.path.splitext(fn)
        fn_temp=f_noext+'_temp'+ext
        os.rename(fn, fn_temp)
        ds=xr.open_dataset(fn_temp)
        write_xarray_to_nc(ds,fn)
        os.remove(fn_temp)
