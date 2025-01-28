#!/usr/bin/env python

import os
import xarray as xr

# Create a single directory
def create_directory(dir_name):
    try:
        os.mkdir(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")


def write_to_filelist(infilenames,outfile):
    with open(outfile, 'w') as file:
        for infilename in infilenames:
            file.write(infilename + '\n')

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
