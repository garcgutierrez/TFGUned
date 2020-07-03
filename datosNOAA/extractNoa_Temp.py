# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:37:07 2019

@author: gutierrez
"""

from pylab import *
import glob as gl
from scipy.io import netcdf
files = gl.glob('C:/Users/gutierrez/Downloads/air*.nc')
import netCDF4
import numpy as np
top_level = 10.0
low_level = 200.0
top_lat = 50.0
low_lat = -50.0
import os
#file1 = files[0]
for file1 in files[::-1]:
    print(file1)
    ncf = netCDF4.Dataset(file1)
    name = (file1.split('\\')[-1]).split('.nc')[0]
    lat = array(ncf['lat'])
    lon = array(ncf['lon'])
    indices_lat = (argwhere(np.logical_and(lat<top_lat,  lat>low_lat))).reshape(-1)
    time = array(ncf['time'])
    indices_level = (argwhere(np.logical_and(array(ncf['level'])>top_level,  array(ncf['level'])<low_level))).reshape(-1)
    uwnd = array(ncf['air'])[:,indices_level[0]:indices_level[-1]+1,indices_lat[0]:indices_lat[-1]+1, :]
    savez_compressed(name+'_50_v2',U=uwnd, lat=lat[indices_lat[0]:indices_lat[-1]+1], lon=lon, levels=array(ncf['level'])[indices_level[0]:indices_level[-1]+1])
    try:
        ncf.close()
        os.remove(file1)
    except Exception as e:
        print(e)