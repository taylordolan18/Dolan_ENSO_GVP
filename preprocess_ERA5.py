#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:20:27 2022

- This code is preprocessing ERA5 500 geopotential heights to be used to initialize a SOM
- A 30-yr rolling climatology is calculated, and the anomalies are calculated
- A sample plot is also included to check our work

@author: taylordolan
"""
#Imports
import os
import netCDF4 as nc
import xarray as xr
import numpy as np
import warnings
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PATH ='/Users/taylordolan/Documents/'

##########################################################################################################
#Functions

def roll_climo(data, month, yrsroll=30, centered=True, time='time'):
    """
    Creating rolling 30-year mean climatology.
    
    Args:
        data (xarray dataarray): Data Array weighted mean already computed.
        month (int): Month for climatology.
        yrsroll (int): Number of years for climatology. Defaults to ``30``.
        centered (boolean): Whether the average is centered. Defaults to ``True``.
        time (str): Time coordinate name. Defaults to ``time``.
    """
    return data[data[f'{time}.month']==month].rolling(time=yrsroll, min_periods=1, center=centered).mean()

def monthly_climo(data, yrsroll=30, centered=True, time='time'):
    """
    Create rolling mean climatology. 
    Performs what xr.DataArray.groupby('time.month').rolling() would do.
    
    Args:
        data (xarray data array): Weighted mean variable.
        yrsroll (int): Number of years for climatology. Defaults to ``30``.
        centered (boolean): Whether the average is centered. Defaults to ``True``.
        time (str): Time coordinate name. Defaults to ``time``.
        
    Returns:
        nino_climo with rolling mean computed along months.
    """
    with warnings.catch_warnings():
        
        # ignore computer performance warning here on chunks
        warnings.simplefilter("ignore")
        
        jan = roll_climo(data, month=1, yrsroll=yrsroll, centered=centered, time=time)
        feb = roll_climo(data, month=2, yrsroll=yrsroll, centered=centered, time=time)
        mar = roll_climo(data, month=3, yrsroll=yrsroll, centered=centered, time=time)
        apr = roll_climo(data, month=4, yrsroll=yrsroll, centered=centered, time=time)
        may = roll_climo(data, month=5, yrsroll=yrsroll, centered=centered, time=time)
        jun = roll_climo(data, month=6, yrsroll=yrsroll, centered=centered, time=time)
        jul = roll_climo(data, month=7, yrsroll=yrsroll, centered=centered, time=time)
        aug = roll_climo(data, month=8, yrsroll=yrsroll, centered=centered, time=time)
        sep = roll_climo(data, month=9, yrsroll=yrsroll, centered=centered, time=time)
        boo = roll_climo(data, month=10, yrsroll=yrsroll, centered=centered, time=time)
        nov = roll_climo(data, month=11, yrsroll=yrsroll, centered=centered, time=time)
        dec = roll_climo(data, month=12, yrsroll=yrsroll, centered=centered, time=time)
        
        nino_climo = xr.concat([jan,feb,mar,apr,may,jun,jul,aug,sep,boo,nov,dec], dim=time).sortby(time)
        
    return nino_climo

##########################################################################################################
#Working with the data
ds = xr.open_dataset(PATH+'e5.Z500_1979-2021_regridded.nc')  #open the dataset
ds = ds.sel(lat = slice(30,49), lon = slice(-109+360, -89.7+360))/9.80665 #select area

climo = monthly_climo(ds['Z'])  #Calculate the climatology
anom= ds['Z'] - climo  #Calculate the anomaly


##########################################################################################################
#Sample Plot To Check
proj = ccrs.LambertConformal(central_longitude=-100)
ax = plt.axes(projection=proj)
anom.isel(time=400).plot.pcolormesh('lon','lat',cmap='seismic', transform=ccrs.PlateCarree(), ax=ax, vmin=-50, vmax=50)
ax.coastlines()
ax.add_feature(cfeature.STATES, linestyle=':')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.BORDERS)
plt.show()
