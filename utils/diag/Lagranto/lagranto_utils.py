import os
import subprocess

def run_Lagranto(startfdir,
                 period,
                 outputdir,
                 input_interval,
                 output_interval,
                 reference_date,
                 era5dir, 
                 quiet=False):
    
    # Lagranto command
    lagranto_command = f"{os.getenv('LAGRANTODIR')}/prog/caltra " \
                       f"{startfdir} " \
                       f"{period} " \
                       f"{outputdir} " \
                       f"-i {input_interval} " \
                       f"-o {output_interval} " \
                       f"-ref {reference_date} " \
                       f"-cdf {era5dir}/ " \
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
