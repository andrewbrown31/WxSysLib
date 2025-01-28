#! /bin/bash

#PBS -P gb02
#PBS -q normal
#PBS -l walltime=06:00:00
#PBS -l ncpus=12
#PBS -l mem=96GB
#PBS -l wd
#PBS -l other=gdata
#PBS -l jobfs=1GB
#PBS -l storage=gdata/hh5+gdata/rt52+gdata/gb02+scratch/gb02

# load openmpi to parallel running for a year
module load openmpi

##  STEP 0 ----- specify directories 
# LPS_indir  contains a list of era5 files used in TempestExtremes
# LPS_outdir contains a list of output DetectedNodes
LPS_indir=/g/data/gb02/cj0591/TC_run/TC_in
LPS_outdir=/g/data/gb02/cj0591/TC_run/TC_out

# LPS_TC_temp contains DetectNodes output files (temporary)
LPS_TC_temp=/g/data/gb02/cj0591/TC_run/TC

# logDN_dir contains log files for DetectNodes (temporary)
# logSN_dir contains log files for StitchNodes (temporary)
logDN_dir=/g/data/gb02/cj0591/TC_run/TE_log/DN_log
logSN_dir=/g/data/gb02/cj0591/TC_run/TE_log/SN_log

# create if not exit
[ ! -d "$logDN_dir" ] && mkdir -p "$logDN_dir"
[ ! -d "$logSN_dir" ] && mkdir -p "$logSN_dir"
[ ! -d "$LPS_indir" ] && mkdir -p "$LPS_indir"
[ ! -d "$LPS_outdir" ] && mkdir -p "$LPS_outdir"
[ ! -d "$LPS_TC_temp" ] && mkdir -p "$LPS_TC_temp"
[ ! -d "$LPS_TC_trck" ] && mkdir -p "$LPS_TC_trck"

# input ERA5 pressure-level and single-level data on gadi
# modify to your dataset directories if needed
ERA5_PRESUELEVEL_SIR=/g/data/rt52/era5/pressure-levels/reanalysis
ERA5_SINGLELEVEL_SIR=/g/data/rt52/era5/single-levels/reanalysis

##  STEP 1 ----- create list of input and output files ----- STEP 1 ##

# Check the input/output file list is exit or not
if [ -f ${LPS_indir}"/ERA5_in.txt" ]; then
    rm ${LPS_indir}/ERA5_in.txt
fi
if [ -f ${LPS_outdir}"/ERA5_out.txt" ]; then
    rm ${LPS_outdir}/ERA5_out.txt
fi

# run TC detection for period of time e.g., 1980-1985
for year in {1980..1985}
do
    # Loop the filenames month by month
    for filenames in "${ERA5_SINGLELEVEL_SIR}/msl/${year}/"*.nc; do
        # extract timerange: YYYYMMDD-YYYYMMDD
        filebase=$(basename $filenames)
        yearmonth=${filebase:18:17}
        yearmonthday=${filebase:18:20}
            
        # era5 files names (z, zs, msl, 10u, 10v)
        zfile=${ERA5_PRESUELEVEL_SIR}/z/${year}/z_era5_oper_pl_${filebase:18:24}
        zsfile=/g/data/gb02/cj0591/geometric_height/zs_era5_oper_sfc_invariant.nc
        mslfile=${ERA5_SINGLELEVEL_SIR}/msl/${year}/msl_era5_oper_sfc_${filebase:18:24}
        u10file=${ERA5_SINGLELEVEL_SIR}/10u/${year}/10u_era5_oper_sfc_${filebase:18:24}
        v10file=${ERA5_SINGLELEVEL_SIR}/10v/${year}/10v_era5_oper_sfc_${filebase:18:24}
            
        # write era5 files input/output lists
        echo "$zfile;$zsfile;$mslfile;$u10file;$v10file" >> ${LPS_indir}/ERA5_in.txt
        echo "TC/era5.LPS.node.${yearmonth}.txt" >> ${LPS_outdir}/ERA5_out.txt

    done
done

##  STEP 2 ----- TempestExtremes DetectNodes to detect TC ----- STEP 2 ##

# your TempestExtremes directory 
TEMPESTEXTREMESDIR='/home/565/cj0591/tempestextremes/bin'

# input/output era5 fields list from STEP 1
inputfile=/g/data/gb02/cj0591/TC_run/TC_in/ERA5_in.txt
outputfile=/g/data/gb02/cj0591/TC_run/TC_out/ERA5_out.txt

# DetectNodes
# mpirun allows parellel running; mpirun -np nodesnumber
# please add mpirun command ahead of TempestExtremes command
# TC detection is based on warm-core criterion from Zarzycki and Ullrich (2017)
# https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016GL071606
# TC centre is defined as the local minima in the mean sea level pressure, with two additional contour ceriteria
# 1) "msl,200.0,5.5,0" indicate mean sea level pressure must increase by 200 Pa by a 5.5 great circle distance (GCD) from the candidate point
# 2  "_DIFF(z(300millibars),z(500millibars)),-58.8,6.5,1.0" indicate difference between geopotential on the 300 and 500 hPa must decrease by 58.8 m2 s−2 (6 m geopotential height) over a 6.5 GCD, 
#     using the maximum value of this field within 1◦ GCD as reference. This criterion indicates that there must be a coherent warm core at the upper levels

mpirun -np 12 $TEMPESTEXTREMESDIR/DetectNodes \
--in_data_list $inputfile --out_file_list $outputfile \
--searchbymin msl \
--mergedist 6.0 \
--closedcontourcmd "msl,200.0,5.5,0;_DIFF(z(300millibars),z(500millibars)),-58.8,6.5,1.0" \
--outputcmd "msl,min,0;_VECMAG(u10,v10),max,2;zs,min,0" \
--timefilter "6hr" --latname "latitude" --lonname "longitude" --logdir "./TE_log/DN_log"

##  STEP 3  ----- Use TempestExtremes StitchNodes to connect detected LPS ----- STEP 3 ##
# inputfile are a list files of DetectedNodes files for individual months within a year
# outputfile are a single csv or txt file within all stitched lows for a year 
inputfile=/g/data/gb02/cj0591/TC_run/TC_out/ERA5_out.txt
outputfile=/g/data/gb02/cj0591/TC_run/ERA5_track.csv

# StitchNode
$TEMPESTEXTREMESDIR/StitchNodes \
--in_list $inputfile --out $outputfile \
--out_file_format "csv" \
--in_fmt "lon,lat,msl,wind,zs" \
--range 8.0 --mintime "54h" --maxgap "24h" \
--threshold "wind,>=,10.0,10;lat,<=,50.0,10;lat,>=,-50.0,10;zs,<,150,10"

##  STEP 4 ----- remove temporary files generated by STEP 2 ----- STEP 4 ##
LPSDIR=/g/data/gb02/cj0591/TC_run/TC/
pattern="*${year}*"
find "${LPSDIR}" -type f -name "$pattern" -exec rm {} \;

## STEP 5 ----- remove log files genreated by STEP 2 ----- STEP 5 ##
LOGDIR=/g/data/gb02/cj0591/TC_run/TC_run/TE_log/DN_log/
pattern="*log*"
find "${LOGDIR}" -type f -name "$pattern" -exec rm {} \;
