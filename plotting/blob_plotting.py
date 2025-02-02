import sys
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature, LAND
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.path as mpath
theta = np.linspace(0, 2*np.pi, 100)
map_circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + [0.5, 0.5]) #This for the polar stereographic plots
from PIL import Image,ImageOps

def plot_clim_count(data,image_desc='',dpi=300,outpath=None,outfileprefix=None,
                         projection=ccrs.PlateCarree()):

    figsize=(11,8)
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=dict(projection=projection)) 

    if projection==ccrs.PlateCarree():
        ax.set_extent([-180,180,-90,90],crs=ccrs.PlateCarree())
    else:
        ax.set_boundary(map_circle, transform=ax.transAxes)
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)

    data_nan=np.where(data == 0, np.nan, data)
    pcm=ax.pcolormesh(data.longitude,data.latitude,data_nan,
                  vmin=1, vmax=np.nanmax(data),
                  transform=ccrs.PlateCarree())
            
    ax.set_title(image_desc)

    cbar=plt.colorbar(pcm)
    cbar.ax.set_ylabel('Count', rotation=90)

    if outfileprefix is not None:
        outfile=outpath+outfileprefix+'_count_clim.jpg'
        plt.savefig(outfile, dpi=dpi)
        crop(outfile,padding=10)
        
    plt.show()

def plot_tracks_at_time(track_df,obj_xr,dt,image_desc='',dpi=300,outpath=None,outfileprefix=None,
                         time_coord='datetime',point_to_track_lon='centlon',point_to_track_lat='centlat',
                         track_id_coord='bnum',
                         projection=ccrs.PlateCarree()):
    track_df_dt=track_df[track_df[time_coord]==dt]
    track_df_todt=track_df[track_df[time_coord]<=dt]
    track_id_dt=track_df_dt[track_id_coord].to_list()
    xcom_dt=track_df_dt[point_to_track_lon].to_list()
    ycom_dt=track_df_dt[point_to_track_lat].to_list()
    obj_xr_dt=obj_xr.sel(time=dt)
        
    ################################################################################################################

    figsize=(11,8)
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=dict(projection=projection)) 

    if projection==ccrs.PlateCarree():
        ax.set_extent([-180,180,-90,90],crs=ccrs.PlateCarree())
    else:
        ax.set_boundary(map_circle, transform=ax.transAxes)
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    
    #ax.contourf(obj_xr.longitude,obj_xr.latitude,obj_xr_dt,levels=[1,1e10])
    obj_xr_dt_nan=np.where(obj_xr_dt == 0, np.nan, obj_xr_dt)
    ax.pcolormesh(obj_xr_dt.longitude,obj_xr_dt.latitude,obj_xr_dt_nan,
                  cmap=ListedColormap(['lightseagreen']),
                  vmin=np.nanmin(obj_xr_dt_nan), vmax=np.nanmax(obj_xr_dt_nan),
                  transform=ccrs.PlateCarree())
    for t in track_id_dt:
        track_to_dt=track_df_todt[track_df_todt[track_id_coord]==t]
        xtrack_com=track_to_dt[point_to_track_lon].to_list()
        ytrack_com=track_to_dt[point_to_track_lat].to_list()
        ax.plot(xtrack_com,ytrack_com,color='black',linewidth=1,transform=ccrs.Geodetic())
            
    ax.scatter(xcom_dt,ycom_dt,c='black',s=30,transform=ccrs.PlateCarree())
    ax.set_title(image_desc+' on '+dt.strftime('%Y-%m-%d %H:%M:%S'))

    if outfileprefix is not None:
        outfile=outpath+outfileprefix+'_'+dt.strftime('%Y%m%d%H%M%S')+'.jpg'
        plt.savefig(outfile, dpi=dpi)
        crop(outfile,padding=10)
        
    plt.show()

def crop(path, in_padding=1,pad_type='all',**kwargs):
    Image.MAX_IMAGE_PIXELS = None
    
    try:
        padding = int(in_padding)
        padding = np.asarray([-1*padding, -1*padding, padding, padding])
    except :
        print("Usage: python PNGWhiteTrim.py ../someFolder padding")
        sys.exit(1)
    
    filePaths = glob.glob(path) #search for all png images in the folder
    
    if len(filePaths) == 0:
        print("No files detected!")
    
    for filePath in filePaths:
        image=Image.open(filePath)
        image.load()
        imageSize = image.size
    
        # remove alpha channel
        invert_im = image.convert("RGB")
    
        # invert image (so that white is 0)
        invert_im = ImageOps.invert(invert_im)
        imageBox = invert_im.getbbox()
        imageBox = tuple(np.asarray(imageBox)+padding)

        print(imageBox,imageSize)

        if pad_type=='y-only':
            imageBox=(0,imageBox[1],imageSize[0],imageBox[3])
    
        cropped=image.crop(imageBox)
        print(filePath, "Size:", imageSize, "New Size:", imageBox)
        cropped.save(filePath)