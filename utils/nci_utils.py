from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

def get_GADI_ERA5_filename(invar,indt,stream='hourly',level_type='potential-temperature',**kwargs):
    yyyy=indt.strftime('%Y')
    mm=indt.strftime('%m')
    emd=datetime(int(yyyy),int(mm),1)+relativedelta(months=1)-relativedelta(days=1)
    emd_str=emd.strftime('%Y%m%d')

    if level_type=='potential-temperature':
        gadiproj='uc16'
        levtype='pt'
        if stream=='hourly':
            era5stream='oper'
            era5streamdir='oper'
        if stream=='monthly':
            era5stream='mnth'
            era5streamdir='mnth'
        filename=invar+'_era5_'+levtype+'_'+era5stream+'_an_'+yyyy+mm+'01-'+emd_str+'.nc'
    if level_type=='potential-vorticity':
        gadiproj='uc16'
        levtype='pv'
        if stream=='hourly':
            era5stream='oper'
            era5streamdir='oper'
        if stream=='monthly':
            era5stream='mnth'
            era5streamdir='mnth'
        filename=invar+'_era5_'+levtype+'_'+era5stream+'_an_'+yyyy+mm+'01-'+emd_str+'.nc'
    if level_type=='pressure-levels':
        gadiproj='rt52'
        levtype='pl'
        if stream=='hourly':
            era5stream='oper'
            era5streamdir='reanalysis'
        if stream=='monthly':
            era5stream='moda'
            era5streamdir='monthly-averaged'
        filename=invar+'_era5_'+era5stream+'_'+levtype+'_'+yyyy+mm+'01-'+emd_str+'.nc'
    if level_type=='single-levels':
        gadiproj='rt52'
        levtype='sfc'
        if stream=='hourly':
            era5stream='oper'
            era5streamdir='reanalysis'
        filename=invar+'_era5_'+era5stream+'_'+levtype+'_'+yyyy+mm+'01-'+emd_str+'.nc'
        
    ncpath='/g/data/'+gadiproj+'/era5/'+level_type+'/'+era5streamdir+'/'+invar+ '/'+yyyy+'/'+filename
    
    return ncpath