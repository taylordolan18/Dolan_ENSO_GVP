#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:11:47 2022

- This code regrids ERA5 500hpa Geopotential Heights to match CESM2-LE Z500 heights
- Need to run this code under the new environment 'xesmf_env'

@author: taylordolan
"""
import xarray as xr
import numpy as np
import xesmf as xe

#lat_res = 0.25
#lon_res = 0.25

#cesm_lat_res = 0.94240838
#cesm_lon_res = 1.25

ds_cesm = xr.open_dataset('b.e21.Z500.1031.002.nc').isel(time=0)

ds_lats = ds_cesm.lat
ds_lons = ds_cesm.lon

ds_era5 = xr.open_dataset('e5.Z500_1979-2021.nc') #open the ERA5 dataset


#ds_coarse = xe.util.grid_global(1, 1) #dont know what values to put for the resoluttion

#Find the resolution of the data
# lat_res = (ds['latitude'].max() - ds['latitude'].min())/(ds['latitude'].count()-1.) 
# lon_res = (ds['longitude'].max() - ds['longitude'].min())/(ds['longitude'].count()-1.) 

# ds_out = xe.util.grid_2d(251, 270.3, 1.25, 30, 49, 0.94240838) #Need to regrid the data
# regridder = xe.Regridder(ds, ds_out, 'nearest_s2d',reuse_weights=False) #regrid
# dr_out = regridder(ds['Z'], keep_attrs=True) #keep attributes

#dcoord=1.0
#lat_coord='latitude'
#lon_coord='longitude'

#Identify min max range (already did above as well)
lat0_bnd = int(np.around(ds_lats.min(skipna=True).values))
lat1_bnd = int(np.around(ds_lats.max(skipna=True).values))
lon0_bnd = int(np.around(ds_lons.min(skipna=True).values))
lon1_bnd = int(np.around(ds_lons.max(skipna=True).values))-1

ds_out = xe.util.grid_2d(lon0_b=lon0_bnd-0.625, 
                         lon1_b=lon1_bnd+0.625,
                         d_lon=(ds_lons[5]-ds_lons[4]).values, 
                         
                         lat0_b=lat0_bnd-0.47120419, 
                         lat1_b=lat1_bnd, 
                         d_lat=(ds_lats[5] - ds_lats[4]).values) #test method to regrid


def regrid(ds_in, ds_out, variable, method='bilinear'):
    """Convenience function for one-time regridding"""
    regridder = xe.Regridder(ds_in, ds_out, method=method)
    dr_out = regridder(ds_in[variable], keep_attrs=True)
    dr_out = dr_out.assign_coords(lon=('x', dr_out.coords['lon'][0,:].values), lat=('y', dr_out.coords['lat'][:,0].values))
    dr_out = dr_out.rename(y='lat', x='lon')
    return dr_out

# method_list = [
#     "bilinear",
#     "conservative",
#     "nearest_s2d",
#     "nearest_d2s",
#     "patch",
# ]

x = regrid(ds_era5, ds_out, 'Z') #this gets close


# # # ds_out = xr.Dataset({"latitude": (["latitude"], np.arange(30,49, .94)),"longitude": (["longitude"], np.arange(251, 270.3, 1.3)),})

# regridder = xe.Regridder(ds,ds_out, method='conservative')

# dr_out = regridder(ds['Z'], keep_attrs=True)
# dr_out = dr_out.assign_coords(lon=('x', dr_out.coords['lon'][0,:].values), lat=('y', dr_out.coords['lat'][:,0].values))
# dr_out = dr_out.rename(y='lat', x='lon')

# dr_out.where(dr_out==1).plot.pcolormesh(figsize=(16,12));

x.to_netcdf('e5.Z500_1979-2021_regridded.nc')