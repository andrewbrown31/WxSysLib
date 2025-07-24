import sys
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature, LAND
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from PIL import Image,ImageOps

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