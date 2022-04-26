#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:55:13 2021

@author: taylordolan
"""
import os
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.cm as cm
from pandas.tseries.offsets import DateOffset
from dateutil.relativedelta import relativedelta, MO
import cartopy.crs as ccrs
import cartopy as cart
import seaborn as sns
from datetime import datetime
from dateutil import rrule

##   #%% use this function to only run a certain block of code

#Function to find ENSO events

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def find_events(index, operator, threshold, per=5, window=[-3, 3]):
    ''' Return the locations of ENSO warm/cold event peaks for a given index.
    Args:
        index (numpy 1d array): ENSO SST anomaly
        operator (str): ">" (index greater than threshold) or "<" (index smaller
            then threshold)
        threshold (numeric): threshold for event definition
        per (int): minimum persistence for index >/< threshold
                (default=5, unit is consistent with the array grid)
        window  (iterable): range around the event peak within which the peak
            has to be a global minima/maxima; length = 2 (default=[-3,3])
    Returns:
       (pklocs, pks) = (location in the input array, values of extrema)
    '''
    if operator == '>':
        argpeak_op = np.argmax
        comp_op = np.greater
        peak_op = np.max
    elif operator == '<':
        argpeak_op = np.argmin
        comp_op = np.less
        peak_op = np.min
    else:
        raise Exception('operator has to be either > or <')

    if len(window) != 2:
        raise ValueError("window must have length=2")

    locs = np.where(comp_op(index, threshold))[0]
    if len(locs) <= 1:
        return ([], np.array([]))

# Find the beginning (starts) and the end (ends) of events
    jumps = np.where(np.diff(locs) > 1)[0]
    starts = np.insert(locs[jumps+1], 0, locs[0])
    ends = np.append(locs[jumps], locs[-1])

# Ignore the chunks that starts from the beginning or ends at the end of the index
#HAD TO COMMENT THIS OUT TO GET EVERYTHING TO WORK SOMEHOW WITH THE INTENSITY?
    # if starts[0] == 0:
    #     starts = starts[1:]
    #     ends = ends[1:]
    # if ends[-1] == len(index)-1:
    #     starts = starts[:-1]
    #     ends = ends[:-1]

# Chunks of the index that exceed the threshold
    subsets = [index[starts[i]:ends[i]] for i in range(len(starts))]

# Find the location of peaks and apply persistence check
    pklocs = [starts[i]+argpeak_op(subsets[i])
              for i in range(len(subsets))
              if len(subsets[i]) >= per]

# Check for being global extrema within the window
    pklocs_new = []
    local_append = pklocs_new.append
    for loc in pklocs:
        window_start = np.max([0, loc+window[0]])
        window_end = np.min([len(index)-1, loc+window[1]])
        if index[loc] == peak_op(index[window_start:window_end]):
            local_append(loc)

# I don't think this does anything more than copying pklocs_new to pklocs
    pklocs = [int(loc) for loc in pklocs_new if loc != False]

    pks = np.array([index[loc].squeeze() for loc in pklocs])

    return pklocs, pks

def find_enso_threshold(index, warm_threshold, cold_threshold, *args, **kwargs):
    ''' Similar to find_enso_percentile but uses threshold to find ENSO events
    Args:
        index (numpy.ndarray): ENSO SST anomaly monthly index.  Masked array
           is supported
        warm_threshold (numeric): Above which El Nino SST anomalies are
        cold_threshold (numeric): Below which La Nina SST anomalies are
    args and kwargs are fed to :py:func:`find_events`
    Returns:
       (dict, dict) each has keys "locs" and "peaks".  "locs" contains the index
         where an event peaks.  "peaks" contains the corresponding peak values
    '''
    warm = {}
    cold = {}
    if not isinstance(index, np.ma.core.MaskedArray):
        index = np.ma.array(index)
    warm['locs'], warm['peaks'] = find_events(index, '>', warm_threshold,
                                              *args, **kwargs)
    cold['locs'], cold['peaks'] = find_events(index, '<', cold_threshold,
                                              *args, **kwargs)
    return warm, cold

def find_enso_percentile(index, percentile, *args, **kwargs):
    ''' Find ENSO events using percentiles (i.e. insensitive to time mean)
    Args:
        index (numpy.ndarray): ENSO SST anomaly monthly index.  Masked array
           is supported
        percentile (numeric): percentile beyond which El Nino and La Nina
           events are identified
    args and kwargs are fed to :py:func:`find_events`
    Returns:
       (dict, dict) each has keys "locs" and "peaks".  "locs" contains the index
         where an event peaks.  "peaks" contains the corresponding peak values
    Example::
        >>> warm,cold = find_enso_percentile(nino34,15.)
    '''
    if percentile > 50.:
        percentile = 100. - percentile
    if percentile < 0.:
        raise ValueError("percentile cannot be smaller than zero.")
    if percentile > 100.:
        raise ValueError("percentile cannot be bigger than 100.")

    warm = {}
    cold = {}
    if np.ma.isMaskedArray(index):
        if index.mask.any():
            # Ignore masked values
            warm['locs'], warm['peaks'] = find_events(index, '>',
                                                      np.percentile(
                                                          index[~index.mask],
                                                          100-percentile),
                                                      *args, **kwargs)
            cold['locs'], cold['peaks'] = find_events(index, '<',
                                                      np.percentile(
                                                          index[~index.mask],
                                                          percentile),
                                                      *args, **kwargs)
            return warm, cold
    # No masked values:
    warm['locs'], warm['peaks'] = find_events(index, '>',
                                              np.percentile(index,
                                                               100-percentile),
                                              *args, **kwargs)
    cold['locs'], cold['peaks'] = find_events(index, '<',
                                              np.percentile(index,
                                                               percentile),
                                              *args, **kwargs)
    return warm, cold

def compute_duration(nino34, operator, locs, evt_end,
                     remove_merge_event=True):
    ''' Compute the duration of events counting from the event peak (locations
    given by locs) until the termination of events (given by the first
    occurrence of operator(nino34,evt_end)).
    See `Choi et al (2013) <http://dx.doi.org/10.1175/JCLI-D-13-00045.1>`_
    Args:
        nino34 (numpy array): ENSO SST anomaly index
        operator (numpy operator): e.g. numpy.less or numpy.greater
        locs (list of int or numpy array of int): indices of event peaks
        evt_end (scalar number): value of nino34 when an event is considered as
           terminated
        remove_merge_event (bool): remove events that are joined on together
           due to reintensification
    Returns:
        list of int (same length as locs)
    '''
    lengths = []
    for iloc in range(len(locs)):
        loc = locs[iloc]
        after_end = operator(nino34[loc:], evt_end)
        if after_end.any():
            length = np.where(after_end)[0][0]
            if remove_merge_event:
                 # if this is not the last event peak
                if iloc < len(locs)-1:
                    # Only count the duration if the next event occurs after
                    # the termination
                    if locs[iloc+1]-locs[iloc] > length:
                        lengths.append(length)
            else:
                lengths.append(length)

    return lengths

def enso_duration(nino34, percentile, thres_std_fraction,
                  per=5, window=[-3, 3]):
    '''Compute the duration of the warm and cold events
    Args:
        nino34 (numpy 1D array): ENSO SST anomalies
        percentile (numeric): greater than 0 and less than 100
        thres_std_fraction (numeric): the fraction times the nino34 standard
              deviation is used for defining the termination
    Returns:
        dict: with keys "warm" and "cold" each contains a list of integers which
             are the duration of the events
    '''
    warm, cold = find_enso_percentile(nino34, percentile, per, window)
    warm_end = nino34.std()*thres_std_fraction
    cold_end = warm_end*-1
    duration = {}
    duration['warm'] = compute_duration(nino34, np.less, warm['locs'],
                                        warm_end)
    duration['cold'] = compute_duration(nino34, np.greater, cold['locs'],
                                        cold_end)
    duration['warm_locs'] = warm['locs']
    duration['cold_locs'] = cold['locs']
    # Length of duration['warm'] may not match the length of
    # duration['warm_locs'] as double counting is taken care of
    return duration

#######################################################################################################

#Information about all the members
members = [1001.001, 1021.002, 1041.003, 1061.004, 1081.005, 1101.006,
           1121.007, 1141.008, 1161.009, 1181.010, 1231.001, 1231.002, 1231.003,
           1231.004, 1231.005, 1231.006, 1231.007, 1231.008, 1231.009, 1231.010,
           1251.001, 1251.002, 1251.003, 1251.004, 1251.005, 1251.006, 1251.007,
           1251.008, 1251.009, 1251.010, 1281.001, 1281.002, 1281.003, 1281.004,
           1281.005, 1281.006, 1281.007, 1281.008, 1281.009, 1281.010, 1301.001,
           1301.002, 1301.003, 1301.004, 1301.005, 1301.006, 1301.007, 1301.008,
           1301.009, 1301.010, 1231.011, 1231.012, 1231.013, 1231.014, 1231.015,
           1231.016, 1231.017, 1231.018, 1231.019, 1231.020, 1251.011, 1251.012,
           1251.013, 1251.014, 1251.015, 1251.016, 1251.017, 1251.018, 1251.019,
           1251.020, 1281.011, 1281.012, 1281.013, 1281.014, 1281.015, 1281.016,
           1281.017, 1281.018, 1281.019, 1281.020, 1301.011, 1301.012, 1301.013,
           1301.014, 1301.015, 1301.016, 1301.017, 1301.018, 1301.019, 1301.020]

mem_string = '1001.001', '1021.002', '1041.003', '1061.004', '1081.005', '1101.006','1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010','1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007','1251.008', '1251.009', '1251.010', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017',' 1281.018', '1281.019', '1281.020', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020'

#Using member 1 

PATH='/Users/taylordolan/Documents/'
MEMBER='1001.001'
ds = xr.open_dataset(PATH+'1001.001_merged.nc')

#Identify the range
temp = ds['SST'][:,90:102, 152:193]
SST_AVG=temp.mean(dim=['lat', 'lon']).values
time = ds['time'].values  

#Convert to datetime
t = ds.indexes['time'].to_datetimeindex()
df_time = pd.DataFrame(t, columns = ['time'])
df_datetime = pd.to_datetime(t)

#New offset to start the time with Jan (in pandas)
n=1
new_datetime = df_datetime - pd.DateOffset(months=n)

#Comtains just Jan 1880 to Dec 2100
slice_datetime = new_datetime[360::]
year_start=1880
year_end=2100
index=12*np.arange(30)
offset=12*(year_start-1850)
SST_ANOM = np.empty([2652])
np.array([12*(year_end-year_start)])

#DONT MESS WITH THIS LOOP AT ALL, DONT MOVE INDENTS TAYLOR
for j in range(len(SST_ANOM)):
    vals=j+offset-index+(15*12)
    if np.max(vals) >= len(SST_AVG):
        vals=(j+offset)-index
    SST_ANOM[j]=SST_AVG[j+offset]-np.mean(SST_AVG[vals])
    #print(j,j+offset,vals)

SST_FINAL = moving_average(SST_ANOM,3)

#Separate times per month
# jan_inx = slice_datetime[0::12]
# feb_inx = slice_datetime[1::12] 
# mar_inx = slice_datetime[2::12]   
# apr_inx = slice_datetime[3::12]   
# may_inx = slice_datetime[4::12]    
# june_inx = slice_datetime[5::12] 
# july_inx = slice_datetime[6::12]  
# aug_inx = slice_datetime[7::12]   
# sept_inx = slice_datetime[8::12] 
# oct_inx = slice_datetime[9::12]   
# nov_inx = slice_datetime[10::12] 
# dec_inx = slice_datetime[11::12]

#Separate average temps per month
# jan_SST = SST_AVG[0::12]
# feb_SST = SST_AVG[1::12]
# mar_SST = SST_AVG[2::12]
# apr_SST = SST_AVG[3::12]
# may_SST = SST_AVG[4::12]
# jun_SST = SST_AVG[5::12]
# jul_SST = SST_AVG[6::12]
# aug_SST = SST_AVG[7::12]
# sept_SST = SST_AVG[8::12]
# oct_SST = SST_AVG[9::12]
# nov_SST = SST_AVG[10::12]
# dec_SST = SST_AVG[11::12]

#Separate the anomalies each month
# jan_anom = SST_FINAL[0::12]
# feb_anom = SST_FINAL[1::12]
# mar_anom = SST_FINAL[2::12]
# apr_anom = SST_FINAL[3::12]
# may_anom = SST_FINAL[4::12]
# june_anom = SST_FINAL[5::12]
# july_anom = SST_FINAL[6::12]
# aug_anom = SST_FINAL[7::12]
# sept_anom = SST_FINAL[8::12]
# oct_anom = SST_FINAL[9::12]
# nov_anom = SST_FINAL[10::12]
# dec_anom = SST_FINAL[11::12]

#1880 to 1950 Time Frame
# jan_1880_1950_SST = jan_SST[0:71]
# feb_1880_1950_SST = feb_SST[0:71]
# mar_1880_1950_SST = mar_SST[0:71]
# apr_1880_1950_SST = apr_SST[0:71]
# may_1880_1950_SST = may_SST[0:71]
# june_1880_1950_SST = jun_SST[0:71]
# july_1880_1950_SST = jul_SST[0:71]
# aug_1880_1950_SST = aug_SST[0:71]
# sept_1880_1950_SST = sept_SST[0:71]
# oct_1880_1950_SST = oct_SST[0:71]
# nov_1880_1950_SST = nov_SST[0:71]
# dec_1880_1950_SST = dec_SST[0:71]

# jan_1880_1950_anom = jan_anom[0:71]
# feb_1880_1950_anom = feb_anom[0:71]
# mar_1880_1950_anom = mar_anom[0:71]
# apr_1880_1950_anom = apr_anom[0:71]
# may_1880_1950_anom = may_anom[0:71]
# june_1880_1950_anom = june_anom[0:71]
# july_1880_1950_anom = july_anom[0:71]
# aug_1880_1950_anom = aug_anom[0:71]
# sept_1880_1950_anom = sept_anom[0:71]
# oct_1880_1950_anom = oct_anom[0:71]
# nov_1880_1950_anom = nov_anom[0:71]
# dec_1880_1950_anom = dec_anom[0:71]

# reg_1880_1950_SST = [jan_1880_1950_SST, feb_1880_1950_SST, mar_1880_1950_SST, apr_1880_1950_SST, may_1880_1950_SST, june_1880_1950_SST, july_1880_1950_SST, aug_1880_1950_SST, sept_1880_1950_SST, oct_1880_1950_SST, nov_1880_1950_SST, dec_1880_1950_SST]
# reg_1880_1950_anom = [jan_1880_1950_anom, feb_1880_1950_anom, mar_1880_1950_anom, apr_1880_1950_anom, may_1880_1950_anom, june_1880_1950_anom, july_1880_1950_anom, aug_1880_1950_anom, sept_1880_1950_anom, oct_1880_1950_anom, nov_1880_1950_anom, dec_1880_1950_anom]

# #1950 to 2000 Time Frame
# jan_1950_2000_SST = jan_SST[70:121]
# feb_1950_2000_SST = feb_SST[70:121]
# mar_1950_2000_SST = mar_SST[70:121]
# apr_1950_2000_SST = apr_SST[70:121]
# may_1950_2000_SST = may_SST[70:121]
# june_1950_2000_SST = jun_SST[70:121]
# july_1950_2000_SST = jul_SST[70:121]
# aug_1950_2000_SST = aug_SST[70:121]
# sept_1950_2000_SST = sept_SST[70:121]
# oct_1950_2000_SST = oct_SST[70:121]
# nov_1950_2000_SST = nov_SST[70:121]
# dec_1950_2000_SST = dec_SST[70:121]

# jan_1950_2000_anom = jan_anom[70:121]
# feb_1950_2000_anom = feb_anom[70:121]
# mar_1950_2000_anom = mar_anom[70:121]
# apr_1950_2000_anom = apr_anom[70:121]
# may_1950_2000_anom = may_anom[70:121]
# june_1950_2000_anom = june_anom[70:121]
# july_1950_2000_anom = july_anom[70:121]
# aug_1950_2000_anom = aug_anom[70:121]
# sept_1950_2000_anom = sept_anom[70:121]
# oct_1950_2000_anom = oct_anom[70:121]
# nov_1950_2000_anom = nov_anom[70:121]
# dec_1950_2000_anom = dec_anom[70:121]

# reg_1950_2000_SST = [jan_1950_2000_SST, feb_1950_2000_SST, mar_1950_2000_SST, apr_1950_2000_SST, may_1950_2000_SST, june_1950_2000_SST, july_1950_2000_SST, aug_1950_2000_SST, sept_1950_2000_SST, oct_1950_2000_SST, nov_1950_2000_SST, dec_1950_2000_SST]
# reg_1950_2000_anom = [jan_1950_2000_anom, feb_1950_2000_anom, mar_1950_2000_anom, apr_1950_2000_anom, may_1950_2000_anom, june_1950_2000_anom, july_1950_2000_anom, aug_1950_2000_anom, sept_1950_2000_anom, oct_1950_2000_anom, nov_1950_2000_anom, dec_1950_2000_anom]

#2000 - 2025 Timeframe
# jan_2000_2025_SST = jan_SST[120:146]
# feb_2000_2025_SST = feb_SST[120:146]
# mar_2000_2025_SST = mar_SST[120:146]
# apr_2000_2025_SST = apr_SST[120:146]
# may_2000_2025_SST = may_SST[120:146]
# june_2000_2025_SST = jun_SST[120:146]
# july_2000_2025_SST = jul_SST[120:146]
# aug_2000_2025_SST = aug_SST[120:146]
# sept_2000_2025_SST = sept_SST[120:146]
# oct_2000_2025_SST = oct_SST[120:146]
# nov_2000_2025_SST = nov_SST[120:146]
# dec_2000_2025_SST = dec_SST[120:146]

# reg_2000_2025_SST = [jan_2000_2025_SST, feb_2000_2025_SST, mar_2000_2025_SST, apr_2000_2025_SST, may_2000_2025_SST, june_2000_2025_SST, july_2000_2025_SST, aug_2000_2025_SST, sept_2000_2025_SST, oct_2000_2025_SST, nov_2000_2025_SST, dec_2000_2025_SST]

#2025 - 2100 Time Frame
# jan_2025_2100_SST = jan_SST[145:221]
# feb_2025_2100_SST = feb_SST[145:221]
# mar_2025_2100_SST = mar_SST[145:221]
# apr_2025_2100_SST = apr_SST[145:221]
# may_2025_2100_SST = may_SST[145:221]
# june_2025_2100_SST = jun_SST[145:221]
# july_2025_2100_SST = jul_SST[145:221]
# aug_2025_2100_SST = aug_SST[145:221]
# sept_2025_2100_SST = sept_SST[145:221]
# oct_2025_2100_SST = oct_SST[145:221]
# nov_2025_2100_SST = nov_SST[145:221]
# dec_2025_2100_SST = dec_SST[145:221]

# jan_2025_2100_anom = jan_anom[145:221]
# feb_2025_2100_anom = feb_anom[145:221]
# mar_2025_2100_anom = mar_anom[145:221]
# apr_2025_2100_anom = apr_anom[145:221]
# may_2025_2100_anom = may_anom[145:221]
# june_2025_2100_anom = june_anom[145:221]
# july_2025_2100_anom = july_anom[145:221]
# aug_2025_2100_anom = aug_anom[145:221]
# sept_2025_2100_anom = sept_anom[145:221]
# oct_2025_2100_anom = oct_anom[145:221]
# nov_2025_2100_anom = nov_anom[145:221]
# dec_2025_2100_anom = dec_anom[145:221]

# reg_2025_2100_SST = [jan_2025_2100_SST, feb_2025_2100_SST, mar_2025_2100_SST, apr_2025_2100_SST, may_2025_2100_SST, june_2025_2100_SST, july_2025_2100_SST, aug_2025_2100_SST, sept_2025_2100_SST, oct_2025_2100_SST, nov_2025_2100_SST, dec_2025_2100_SST]
# reg_2025_2100_anom = [jan_2025_2100_anom, feb_2025_2100_anom, mar_2025_2100_anom, apr_2025_2100_anom, may_2025_2100_anom, june_2025_2100_anom, july_2025_2100_anom, aug_2025_2100_anom, sept_2025_2100_anom, oct_2025_2100_anom, nov_2025_2100_anom, dec_2025_2100_anom]

#2075 - 2100 TimeFrame
# jan_2075_2100_SST = jan_SST[195:220]
# feb_2075_2100_SST = feb_SST[195:220]
# mar_2075_2100_SST = mar_SST[195:220]
# apr_2075_2100_SST = apr_SST[195:220]
# may_2075_2100_SST = may_SST[195:220]
# june_2075_2100_SST = jun_SST[195:220]
# july_2075_2100_SST = jul_SST[195:220]
# aug_2075_2100_SST = aug_SST[195:220]
# sept_2075_2100_SST = sept_SST[195:220]
# oct_2075_2100_SST = oct_SST[195:220]
# nov_2075_2100_SST = nov_SST[195:220]
# dec_2075_2100_SST = dec_SST[195:220]

# reg_2075_2100_SST = [jan_2075_2100_SST, feb_2075_2100_SST, mar_2075_2100_SST, apr_2075_2100_SST, may_2075_2100_SST, june_2075_2100_SST, july_2075_2100_SST, aug_2075_2100_SST, sept_2075_2100_SST, oct_2075_2100_SST, nov_2075_2100_SST, dec_2075_2100_SST]

#Calculate the std of SST_AVG from 1880 - 1950
# jan_1880_1950_STD = jan_1880_1950_SST.std()
# feb_1880_1950_STD = feb_1880_1950_SST.std()
# mar_1880_1950_STD = mar_1880_1950_SST.std()
# apr_1880_1950_STD = apr_1880_1950_SST.std()
# may_1880_1950_STD = may_1880_1950_SST.std()
# june_1880_1950_STD = june_1880_1950_SST.std()
# july_1880_1950_STD = july_1880_1950_SST.std()
# aug_1880_1950_STD = aug_1880_1950_SST.std()
# sept_1880_1950_STD = sept_1880_1950_SST.std()
# oct_1880_1950_STD = oct_1880_1950_SST.std()
# nov_1880_1950_STD = nov_1880_1950_SST.std()
# dec_1880_1950_STD = dec_1880_1950_SST.std()

# STD_1880_1950 = [jan_1880_1950_STD, feb_1880_1950_STD, mar_1880_1950_STD, apr_1880_1950_STD, may_1880_1950_STD, june_1880_1950_STD, july_1880_1950_STD, aug_1880_1950_STD, sept_1880_1950_STD, oct_1880_1950_STD, nov_1880_1950_STD, dec_1880_1950_STD]
# Mean_SST_1880_1950 = [jan_1880_1950_SST.mean()-273,feb_1880_1950_SST.mean()-273,mar_1880_1950_SST.mean()-273,apr_1880_1950_SST.mean()-273,may_1880_1950_SST.mean()-273,june_1880_1950_SST.mean()-273,july_1880_1950_SST.mean()-273,aug_1880_1950_SST.mean()-273,sept_1880_1950_SST.mean()-273,oct_1880_1950_SST.mean()-273,nov_1880_1950_SST.mean()-273,dec_1880_1950_SST.mean()-273]
# Mean_SST_1880_1950_array = np.array(Mean_SST_1880_1950)
# STD_1880_1950_array = np.array(STD_1880_1950)

#Caclulate the std of SST_AVG from 1950 - 2000
# jan_1950_2000_STD = jan_1950_2000_SST.std()
# feb_1950_2000_STD = feb_1950_2000_SST.std()
# mar_1950_2000_STD = mar_1950_2000_SST.std()
# apr_1950_2000_STD = apr_1950_2000_SST.std()
# may_1950_2000_STD = may_1950_2000_SST.std()
# june_1950_2000_STD = june_1950_2000_SST.std()
# july_1950_2000_STD = july_1950_2000_SST.std()
# aug_1950_2000_STD = aug_1950_2000_SST.std()
# sept_1950_2000_STD = sept_1950_2000_SST.std()
# oct_1950_2000_STD = oct_1950_2000_SST.std()
# nov_1950_2000_STD = nov_1950_2000_SST.std()
# dec_1950_2000_STD = dec_1950_2000_SST.std()

# STD_1950_2000 = [jan_1950_2000_STD, feb_1950_2000_STD, mar_1950_2000_STD, apr_1950_2000_STD, may_1950_2000_STD, june_1950_2000_STD, july_1950_2000_STD, aug_1950_2000_STD, sept_1950_2000_STD, oct_1950_2000_STD, nov_1950_2000_STD, dec_1950_2000_STD]
# Mean_SST_1950_2000 = [jan_1950_2000_SST.mean()-273,feb_1950_2000_SST.mean()-273,mar_1950_2000_SST.mean()-273,apr_1950_2000_SST.mean()-273,may_1950_2000_SST.mean()-273,june_1950_2000_SST.mean()-273,july_1950_2000_SST.mean()-273,aug_1950_2000_SST.mean()-273,sept_1950_2000_SST.mean()-273,oct_1950_2000_SST.mean()-273,nov_1950_2000_SST.mean()-273,dec_1950_2000_SST.mean()-273]
# Mean_SST_1950_2000_array = np.array(Mean_SST_1950_2000)
# STD_1950_200_array = np.array(STD_1950_2000)

#Calculate the std of SST_AVG from 2025 - 2100
# jan_2025_2100_STD = jan_2025_2100_SST.std()
# feb_2025_2100_STD = feb_2025_2100_SST.std()
# mar_2025_2100_STD = mar_2025_2100_SST.std()
# apr_2025_2100_STD = apr_2025_2100_SST.std()
# may_2025_2100_STD = may_2025_2100_SST.std()
# june_2025_2100_STD = june_2025_2100_SST.std()
# july_2025_2100_STD = july_2025_2100_SST.std()
# aug_2025_2100_STD = aug_2025_2100_SST.std()
# sept_2025_2100_STD = sept_2025_2100_SST.std()
# oct_2025_2100_STD = oct_2025_2100_SST.std()
# nov_2025_2100_STD = nov_2025_2100_SST.std()
# dec_2025_2100_STD = dec_2025_2100_SST.std()

# STD_2025_2100 = [jan_2025_2100_STD,feb_2025_2100_STD,mar_2025_2100_STD,apr_2025_2100_STD,may_2025_2100_STD,june_2025_2100_STD,july_2025_2100_STD,aug_2025_2100_STD,sept_2025_2100_STD,oct_2025_2100_STD,nov_2025_2100_STD,dec_2025_2100_STD]
# Mean_SST_2025_2100 = [jan_2025_2100_SST.mean()-273,feb_2025_2100_SST.mean()-273,mar_2025_2100_SST.mean()-273,apr_2025_2100_SST.mean()-273,may_2025_2100_SST.mean()-273,june_2025_2100_SST.mean()-273,july_2025_2100_SST.mean()-273,aug_2025_2100_SST.mean()-273,sept_2025_2100_SST.mean()-273,oct_2025_2100_SST.mean()-273,nov_2025_2100_SST.mean()-273,dec_2025_2100_SST.mean()-273]
# Mean_SST_2025_2100_array = np.array(Mean_SST_2025_2100)
# STD_2025_2100_array = np.array(STD_2025_2100)

########################################################################################################

#Dataframe with SST_FINAL and the times
df_anom = pd.DataFrame(SST_FINAL, columns = ['Anomaly']) #df with SST_FINAL
df_slice_time = pd.DataFrame(slice_datetime, columns = ['Date']) #df with slice time
df_com = pd.concat([df_anom, df_slice_time], axis=1, join='inner') #combined df above
df_com['year'] = df_com['Date'].dt.year
df_com['month'] = df_com['Date'].dt.month

#################################################################################################################

#Testing NOAA Data Table for ENSO Indices/phases
months = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
years = np.arange(1880,2101,1,dtype=int)
SST_resize = np.resize(SST_FINAL,(221,12))
SST_resize_df = pd.DataFrame(SST_resize)
SST_resize_df.columns = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
SST_resize_df.index = years

#Export as an Excel Sheet
# SST_resize_df.to_excel("output.xlsx",
#              sheet_name='ENSO_table')  

#############################################################################################

#Plug in for functions

mem_1_warm_extrema = find_events(SST_FINAL, '>', 0.49, 5, [-3, 3])
mem_1_cold_extrema = find_events(SST_FINAL, '<', -0.49, 5, [-3, 3])

warm_cold = find_enso_threshold(SST_FINAL, 0.49, -0.49)

warm_cold_percentile = find_enso_percentile(SST_FINAL,10)
test_warm_duration = compute_duration(SST_FINAL, np.less, mem_1_warm_extrema[0], 0.49)
test_cold_duration = compute_duration(SST_FINAL, np.greater, mem_1_cold_extrema[0], -0.49)
duration = enso_duration(SST_FINAL,10,1)

####################################################################################################################

#Calculate ENSO events based on the intensity using the peaks
#Used members 1- 10, and did nino and nina and took 33% for each intensity
#Nino -> weak: 1.7 - 2.4, mod: 2.41 - 2.81, strong: 2.84 - 4.06
#Nina -> weak: -1.75 to -2.32, mod: -2.33 to -2.69, strong: -2.7 to -4.46

warm_duration_locs = duration['warm_locs']
warm_duration = duration['warm'] 
warm_events = warm_cold_percentile[0]
warm_peaks = warm_events['peaks']
warm_peaks_list = warm_peaks.tolist()
time_warm_start = slice_datetime[warm_duration_locs]

# weak_nino = list(filter(lambda x: (x<2) , warm_peaks_list))
# mod_nino = list(filter(lambda x: (x>=2 and x<=2.9) , warm_peaks_list))
# strong_nino = list(filter(lambda x: (x>=3), warm_peaks_list))


cold_duration_locs = duration['cold_locs']
cold_duration = duration['cold']
cold_events = warm_cold_percentile[1]
cold_peaks = cold_events['peaks']
time_cold_start = slice_datetime[cold_duration_locs]

warm_peak_array = np.array(warm_peaks)
cold_peak_array = np.array(cold_peaks)

tws_list = list(time_warm_start)
tcs_list = list(time_cold_start)

#Separate ENSO events by intensity with the associated time
warm_peak_times_df = pd.DataFrame(time_warm_start, columns = ['time'])
warm_peak_times_df['warm_peaks'] = warm_peak_array
weak_nino1 = warm_peak_times_df[(warm_peak_times_df['warm_peaks'] >= 1.7) & (warm_peak_times_df['warm_peaks'] <= 2.4)]
mod_nino1 = warm_peak_times_df[(warm_peak_times_df['warm_peaks'] >= 2.41) & (warm_peak_times_df['warm_peaks'] <= 2.81)]
strong_nino1 = warm_peak_times_df[(warm_peak_times_df['warm_peaks'] > 2.81) & (warm_peak_times_df['warm_peaks'] <= 4.06)]

cold_peak_times_df = pd.DataFrame(time_cold_start, columns = ['time'])
cold_peak_times_df['cold_peaks'] = cold_peak_array
weak_nina1 = cold_peak_times_df[(cold_peak_times_df['cold_peaks'] <= -1.75) & (cold_peak_times_df['cold_peaks'] >= -2.32)]
mod_nina1 = cold_peak_times_df[(cold_peak_times_df['cold_peaks'] < -2.32) & (cold_peak_times_df['cold_peaks'] >= -2.69)]
strong_nina1 = cold_peak_times_df[(cold_peak_times_df['cold_peaks'] < -2.7) & (cold_peak_times_df['cold_peaks'] >= -4.46)]

####################################################################################################################
#Match up duration to a time

warm_duration_locs = duration['warm_locs']
warm_duration = duration['warm']  #Missing the last value
warm_duration_array = np.array(warm_duration)
time_warm_start = slice_datetime[warm_duration_locs]
warm_duration_end = [] 

#choose the smaller list to iterate
small_list = len(warm_duration) < len(warm_duration_locs) and warm_duration or warm_duration_locs
for i in range(0, len(warm_duration)): 
	warm_duration_end.append(warm_duration[i] + warm_duration_locs[i]) 

time_warm_end = slice_datetime[warm_duration_end]

cold_duration_locs = duration['cold_locs']
cold_duration = duration['cold']
cold_duration_array = np.array(cold_duration)
time_cold_start = slice_datetime[cold_duration_locs]
cold_duration_end = []

small_list = len(cold_duration) < len(cold_duration_locs) and cold_duration or cold_duration_locs
for i in range(0, len(cold_duration)): 
	cold_duration_end.append(cold_duration[i] + cold_duration_locs[i]) 

time_cold_end = slice_datetime[cold_duration_end]

######################################################################################################################
#Use for plotting
del df_com['year']
del df_com['month']

neutral_df = df_com[(df_com['Anomaly'] < .5) & (df_com['Anomaly'] > -.5)]
nino_df = df_com[(df_com['Anomaly'] >= 0.5)]
nina_df = df_com[(df_com['Anomaly'] <= -0.5)] 

#Use for Plotting and putting dots on anomaly graph
w_c_slice_warm = warm_cold_percentile[0]
wc_more_slice = w_c_slice_warm['peaks'] #x-coord
time_slice_warm = [slice_datetime[59], slice_datetime[252],slice_datetime[539], slice_datetime[635], slice_datetime[755], slice_datetime[791], slice_datetime[835], slice_datetime[957], slice_datetime[1114], slice_datetime[1163], slice_datetime[1212], slice_datetime[1488], slice_datetime[1607], slice_datetime[1644], slice_datetime[1703], slice_datetime[1785], slice_datetime[1846], slice_datetime[1918], slice_datetime[2134], slice_datetime[2170], slice_datetime[2242], slice_datetime[2301], slice_datetime[2361], slice_datetime[2494], slice_datetime[2556], slice_datetime[2580], slice_datetime[2603]]

w_c_slice_cold = warm_cold_percentile[1]
wc_more_cold_slice = w_c_slice_cold['peaks']
time_slice_cold = [slice_datetime[70], slice_datetime[431], slice_datetime[480], slice_datetime[551], slice_datetime[766], slice_datetime[802], slice_datetime[863], slice_datetime[982], slice_datetime[1188], slice_datetime[1307], slice_datetime[1512], slice_datetime[1619], slice_datetime[1664], slice_datetime[1798], slice_datetime[1931], slice_datetime[2026], slice_datetime[2087], slice_datetime[2148], slice_datetime[2159], slice_datetime[2316], slice_datetime[2350], slice_datetime[2446]]

######################################################################################################################

#Read the text file I created
   
# data = open('1001.001_merged.nc_stats.txt', "r")
# lines = data.readlines()

# # remove /n at the end of each line
# for index, line in enumerate(lines):
#       lines[index] = line.strip()

# #Convert list of strings to a list of all lists
# lines_list = [i.split(", ") for i in lines]

######################################################################################

#Creating a DataFrame with all the ENSO info

nino_event = []
nina_event = []
txt1 = 'El Nino'
txt2 = 'La Nina'
txt3 = 'Neutral'

for i in warm_peaks:
    if i >=0.5:
          nino_event.append(txt1)
    else:
            nino_event.append(txt3)
            
for i in cold_peaks:
        if i <=-0.5:
            nina_event.append(txt2)


text1 = 'Weak'
text2 = 'Moderate'
text3 = 'Strong'
intensity = []
intensity1 = []
for i in warm_peak_array:
    if i >=2.84:
        intensity.append(text3)
    else:
        if i <2.41:
            intensity.append(text1)
        else:
            intensity.append(text2)

for i in cold_peak_array:
    if i <= -2.7:
        intensity1.append(text3)
    else:
        if i > -2.33:
            intensity1.append(text1)
        else:
            intensity1.append(text2)
list1 = []
list2 = []
for i, j, k, l, m, n in zip(nina_event, time_cold_start, time_cold_end, cold_duration_array, cold_peak_array, intensity1):
    # print(i,j,k,l,m,n)
    list2.append((i,j,k,l,m,n))
array1 = np.vstack(list2)
new_column_names = ['Type', 'Time Start', 'Time End', 'Duration', 'Peaks', 'Strength'] 
array_pd_nina = pd.DataFrame(array1, columns = new_column_names)

for i, j, k, l, m, n in zip(nino_event, time_warm_start, time_warm_end, warm_duration_array, warm_peak_array, intensity):
    # print(i,j,k,l,m,n)
    list1.append((i,j, k, l, m, n))
array = np.vstack(list1)
column_names = ['Type', 'Time Start', 'Time End', 'Duration', 'Peaks', 'Strength']    
array_pd_nino = pd.DataFrame(array, columns=column_names)

frames = [array_pd_nino, array_pd_nina]

result = pd.concat(frames)
result.to_csv('Member_1_Results.csv',index=False)  #this is the final result

###############################################################################################

#Sort df_com to find when neutral is occurring

neutral_with_percentile = df_com[(df_com['Anomaly'] >=-1.535) & (df_com['Anomaly'] <=1.655)]
# del neutral_with_percentile['year']
# del neutral_with_percentile['month']

###############################################################################################

#Importing data to dataframe then reading it
# test_df = pd.DataFrame(warm_peak_array, columns = ['Peaks'])
# test_df.to_csv('testwarm.csv')
# new_df = pd.read_csv('testwarm.csv') #warm peak intensities
# del new_df['Unnamed: 0']

# new_df = pd.read_csv('1281.017_Results.csv') 

######################################################################################################################
#PLOTS

#Anomaly Plot (KEEP)
# fig = plt.figure(figsize=(10, 6),dpi=600)  
# plt.plot(slice_datetime,SST_FINAL,color='black')
# plt.plot(nino_df.Date, nino_df.Anomaly, color = 'r')
# plt.plot(nina_df.Date, nina_df.Anomaly, color = 'b')
# plt.plot(neutral_df.Date, neutral_df.Anomaly, color = 'k')
# plt.axhline(y=0.5, color='black', linestyle='--')
# plt.axhline(y=-0.5, color='black', linestyle='--') 
# plt.title('ENSO Anomalies Member 1')
# plt.xlabel('Years')
# plt.ylabel('ENSO Anomaly')

#New Anomaly Plot with colors filled in between
# el_nino = 0.5
# la_nina = -0.5
# # fig = plt.figure(figsize=(16,3),dpi=600)  
# # # plt.plot(slice_datetime,SST_FINAL,color='black',) #Comment out for no neutral
# plt.fill_between(slice_datetime, SST_FINAL,el_nino,
#                    where=(SST_FINAL >= el_nino), 
#                    alpha=0.30, color='red', interpolate=True)
# plt.fill_between(slice_datetime,SST_FINAL, la_nina,
#                    where=(SST_FINAL <= la_nina),
#                    alpha=0.30, color='blue', interpolate=True)
# plt.scatter(x,wc_more_slice, marker='o', c='black')
# # plt.scatter(time_slice_cold, wc_more_cold_slice, marker='o', c='black')
# # plt.axhline(y=0.5, color='black', linestyle='--')
# # plt.axhline(y=-0.5, color='black', linestyle='--') 
# # plt.axhline(y=-1.53, color = 'green', linestyle='--')
# # plt.axhline(y=1.65, color='green', linestyle='--')
# # plt.title('ENSO Timeseries Member 1')
# # plt.xlabel('Years')
# # plt.ylabel('Standard Departure (°C)')
# plt.show()

# plt.plot(time_warm_start[0:26], warm_duration_array) #27, #26
# plt.ylabel('Duration')
# plt.xlabel('Time')
# plt.title('Duration of El Niño Events in Member 1')
# plt.show()

# plt.plot(time_cold_start[0:20], cold_duration_array)
# plt.ylabel('Duration')
# plt.xlabel('Time')
# plt.title('Duration of La Niña Events in Member 1')
# plt.show()

#Bar Graph
# fig = plt.figure(figsize = (10, 5))
# months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
# X_axis = np.arange(len(months))
# STD_1880_1950 = [jan_1880_1950_STD, feb_1880_1950_STD, mar_1880_1950_STD, apr_1880_1950_STD, may_1880_1950_STD, june_1880_1950_STD, july_1880_1950_STD, aug_1880_1950_STD, sept_1880_1950_STD, oct_1880_1950_STD, nov_1880_1950_STD, dec_1880_1950_STD]
# STD_1950_2000 = [jan_1950_2000_STD, feb_1950_2000_STD, mar_1950_2000_STD, apr_1950_2000_STD, may_1950_2000_STD, june_1950_2000_STD, july_1950_2000_STD, aug_1950_2000_STD, sept_1950_2000_STD, oct_1950_2000_STD, nov_1950_2000_STD, dec_1950_2000_STD]
# STD_2025_2100 = [jan_2025_2100_STD,feb_2025_2100_STD,mar_2025_2100_STD,apr_2025_2100_STD,may_2025_2100_STD,june_2025_2100_STD,july_2025_2100_STD,aug_2025_2100_STD,sept_2025_2100_STD,oct_2025_2100_STD,nov_2025_2100_STD,dec_2025_2100_STD]
# plt.bar(X_axis - 0.2, STD_1880_1950, width = 0.3, label = '1880 - 1950', color = 'g')
# plt.bar(X_axis, STD_1950_2000, width = 0.3, label = '1950 - 2000', color = 'pink')
# plt.bar(X_axis + 0.2, STD_2025_2100, width = 0.3, label = '2025 - 2100', color = 'black')
# plt.xticks(X_axis, months)
# plt.xlabel("Months")
# plt.ylabel("Standard Deviation")
# plt.title("Member 1 Standard Deviations of Mean SST in the Niñ0 3.4 Region")
# plt.legend()
# plt.show()

#Bar graph with mean and std  (finished version)
# fig = plt.figure(figsize = (10, 5))
# months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
# x_pos = np.arange(len(months))
# CTEs = Mean_SST_1880_1950_array
# error = STD_1880_1950_array
# plt.bar(x_pos - 0.2, CTEs, yerr=error,width = 0.3,capsize=5,label = '1880 - 1950', color = 'g')
# plt.bar(x_pos, Mean_SST_1950_2000_array, yerr=STD_1950_200_array,width = 0.3,align='center', ecolor='black', capsize=5,label = '1950 - 2000',color = 'pink')
# plt.bar(x_pos + 0.2, Mean_SST_2025_2100_array, yerr=STD_2025_2100_array,width = 0.3,align='center', ecolor='black',capsize=5,label = '2025 - 2100', color = 'blue')
# plt.xticks(x_pos, months)
# plt.title('Member 1 Mean SST and Standard Deviation in the Niño 3.4 Region')
# plt.legend()
# plt.show()

#Box and Whisker Plot SST (1880 - 1950)
# columns=["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"])
# jan_bx1 = (reg_1880_1950_SST[0]-273)
# feb_bx1 = (reg_1880_1950_SST[1]-273)
# mar_bx1 = (reg_1880_1950_SST[2]-273)
# apr_bx1 = (reg_1880_1950_SST[3]-273)
# may_bx1 = (reg_1880_1950_SST[4]-273)
# june_bx1 = (reg_1880_1950_SST[5]-273)
# july_bx1 = (reg_1880_1950_SST[6]-273)
# aug_bx1 = (reg_1880_1950_SST[7]-273)
# sept_bx1 = (reg_1880_1950_SST[8]-273)
# oct_bx1 = (reg_1880_1950_SST[9]-273)
# nov_bx1 = (reg_1880_1950_SST[10]-273)
# dec_bx1 = (reg_1880_1950_SST[11]-273)
# box = plt.boxplot([jan_bx1, feb_bx1, mar_bx1, apr_bx1, may_bx1, june_bx1, july_bx1, aug_bx1, sept_bx1, oct_bx1, nov_bx1, dec_bx1], patch_artist=True, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black","markersize":"6"}, boxprops=dict(facecolor="black", color="black"),capprops=dict(color="black"),whiskerprops=dict(color="black"),flierprops=dict(color="black", markeredgecolor="black"),medianprops=dict(color="black"),labels=["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"])
# colors = ['aquamarine', 'lightblue', 'lightgreen', 'tan', 'orchid', 'coral', 'gold', 'teal', 'magenta', 'pink', 'lavender', 'violet']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
# plt.title("Member 1 1880 - 1950 Sea Surface Temperatures")
# plt.show()

#Box and Whisker Plot SST (1950 - 2000)
# jan_bx2 = (reg_1950_2000_SST[0]-273)
# feb_bx2 = (reg_1950_2000_SST[1]-273)
# mar_bx2 = (reg_1950_2000_SST[2]-273)
# apr_bx2 = (reg_1950_2000_SST[3]-273)
# may_bx2 = (reg_1950_2000_SST[4]-273)
# june_bx2 = (reg_1950_2000_SST[5]-273)
# july_bx2 = (reg_1950_2000_SST[6]-273)
# aug_bx2 = (reg_1950_2000_SST[7]-273)
# sept_bx2 = (reg_1950_2000_SST[8]-273)
# oct_bx2 = (reg_1950_2000_SST[9]-273)
# nov_bx2 = (reg_1950_2000_SST[10]-273)
# dec_bx2 = (reg_1950_2000_SST[11]-273)
# box = plt.boxplot([jan_bx2, feb_bx2, mar_bx2, apr_bx2, may_bx2, june_bx2, july_bx2, aug_bx2, sept_bx2, oct_bx2, nov_bx2, dec_bx2], patch_artist=True, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black","markersize":"6"}, boxprops=dict(facecolor="black", color="black"),capprops=dict(color="black"),whiskerprops=dict(color="black"),flierprops=dict(color="black", markeredgecolor="black"),medianprops=dict(color="black"),labels=["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"])
# colors = ['aquamarine', 'lightblue', 'lightgreen', 'tan', 'orchid', 'coral', 'gold', 'teal', 'magenta', 'pink', 'lavender', 'violet']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
# plt.title("Member 1 1950 - 2000 Sea Surface Temperatures")
# plt.show()

#Box and Whisker Plot SST (2025 - 2100)
# jan_bx3 = (reg_2025_2100_SST[0]-273)
# feb_bx3 = (reg_2025_2100_SST[1]-273)
# mar_bx3 = (reg_2025_2100_SST[2]-273)
# apr_bx3 = (reg_2025_2100_SST[3]-273)
# may_bx3 = (reg_2025_2100_SST[4]-273)
# june_bx3 = (reg_2025_2100_SST[5]-273)
# july_bx3 = (reg_2025_2100_SST[6]-273)
# aug_bx3 = (reg_2025_2100_SST[7]-273)
# sept_bx3 = (reg_2025_2100_SST[8]-273)
# oct_bx3 = (reg_2025_2100_SST[9]-273)
# nov_bx3 = (reg_2025_2100_SST[10]-273)
# dec_bx3 = (reg_2025_2100_SST[11]-273)
# box = plt.boxplot([jan_bx3, feb_bx3, mar_bx3, apr_bx3, may_bx3, june_bx3, july_bx3, aug_bx3, sept_bx3, oct_bx3, nov_bx3, dec_bx3], patch_artist=True, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black","markersize":"6"}, boxprops=dict(facecolor="black", color="black"),capprops=dict(color="black"),whiskerprops=dict(color="black"),flierprops=dict(color="black", markeredgecolor="black"),medianprops=dict(color="black"),labels=["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"])
# colors = ['aquamarine', 'lightblue', 'lightgreen', 'tan', 'orchid', 'coral', 'gold', 'teal', 'magenta', 'pink', 'lavender', 'violet']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
# plt.title("Member 1 2025 - 2100 Sea Surface Temperatures")
# plt.show()

#Box and Whisker plot SST (2000 - 2025)
# jan_bx4 = (reg_2000_2025_SST[0] - 273)
# feb_bx4 = (reg_2000_2025_SST[1] - 273)
# mar_bx4 = (reg_2000_2025_SST[2] - 273)
# apr_bx4 = (reg_2000_2025_SST[3] - 273)
# may_bx4 = (reg_2000_2025_SST[4] - 273)
# june_bx4 = (reg_2000_2025_SST[5] - 273)
# july_bx4 = (reg_2000_2025_SST[6] - 273)
# aug_bx4 = (reg_2000_2025_SST[7] - 273)
# sept_bx4 = (reg_2000_2025_SST[8] - 273)
# oct_bx4 = (reg_2000_2025_SST[9] - 273)
# nov_bx4 = (reg_2000_2025_SST[10] - 273)
# dec_bx4 = (reg_2000_2025_SST[11] - 273)
# box = plt.boxplot([jan_bx4, feb_bx4, mar_bx4, apr_bx4, may_bx4, june_bx4, july_bx4, aug_bx4, sept_bx4, oct_bx4, nov_bx4, dec_bx4], patch_artist=True, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black","markersize":"6"}, boxprops=dict(facecolor="black", color="black"),capprops=dict(color="black"),whiskerprops=dict(color="black"),flierprops=dict(color="black", markeredgecolor="black"),medianprops=dict(color="black"),labels=["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"])
# colors = ['aquamarine', 'lightblue', 'lightgreen', 'tan', 'orchid', 'coral', 'gold', 'teal', 'magenta', 'pink', 'lavender', 'violet']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
# plt.title("Member 1 2000-2025 Sea Surface Temperatures")
# plt.show()

#Box and whisker plot SST (2075 - 2100)
# jan_bx5 = (reg_2075_2100_SST[0] - 273)
# feb_bx5 = (reg_2075_2100_SST[1] - 273)
# mar_bx5 = (reg_2075_2100_SST[2] - 273)
# apr_bx5 = (reg_2075_2100_SST[3] - 273)
# may_bx5 = (reg_2075_2100_SST[4] - 273)
# june_bx5 = (reg_2075_2100_SST[5] - 273)
# july_bx5 = (reg_2075_2100_SST[6] - 273)
# aug_bx5 = (reg_2075_2100_SST[7] - 273)
# sept_bx5 = (reg_2075_2100_SST[8] - 273)
# oct_bx5 = (reg_2075_2100_SST[9] - 273)
# nov_bx5 = (reg_2075_2100_SST[10] - 273)
# dec_bx5 = (reg_2075_2100_SST[11] - 273)
# box = plt.boxplot([jan_bx5, feb_bx5, mar_bx5, apr_bx5, may_bx5, june_bx5, july_bx5, aug_bx5, sept_bx5, oct_bx5, nov_bx5, dec_bx5], patch_artist=True, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black","markersize":"6"}, boxprops=dict(facecolor="black", color="black"),capprops=dict(color="black"),whiskerprops=dict(color="black"),flierprops=dict(color="black", markeredgecolor="black"),medianprops=dict(color="black"),labels=["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"])
# colors = ['aquamarine', 'lightblue', 'lightgreen', 'tan', 'orchid', 'coral', 'gold', 'teal', 'magenta', 'pink', 'lavender', 'violet']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
# plt.title("Member 1 2075 - 2100 Sea Surface Temperatures")
# plt.show()

#Box and Whisker Plot Anomalies (1880 - 1950)
# jan_anom_bx1 = (reg_1880_1950_anom[0])
# feb_anom_bx1 = (reg_1880_1950_anom[1])
# mar_anom_bx1 = (reg_1880_1950_anom[2])
# apr_anom_bx1 = (reg_1880_1950_anom[3])
# may_anom_bx1 = (reg_1880_1950_anom[4])
# june_anom_bx1 = (reg_1880_1950_anom[5])
# july_anom_bx1 = (reg_1880_1950_anom[6])
# aug_anom_bx1 = (reg_1880_1950_anom[7])
# sept_anom_bx1 = (reg_1880_1950_anom[8])
# oct_anom_bx1 = (reg_1880_1950_anom[9])
# nov_anom_bx1 = (reg_1880_1950_anom[10])
# dec_anom_bx1 = (reg_1880_1950_anom[11])
# box = plt.boxplot([jan_anom_bx1, feb_anom_bx1, mar_anom_bx1, apr_anom_bx1, may_anom_bx1, june_anom_bx1, july_anom_bx1, aug_anom_bx1, sept_anom_bx1, oct_anom_bx1, nov_anom_bx1, dec_anom_bx1], patch_artist=True, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black","markersize":"6"}, boxprops=dict(facecolor="black", color="black"),capprops=dict(color="black"),whiskerprops=dict(color="black"),flierprops=dict(color="black", markeredgecolor="black"),medianprops=dict(color="black"),labels=["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"])
# colors = ['aquamarine', 'lightblue', 'lightgreen', 'tan', 'orchid', 'coral', 'gold', 'teal', 'magenta', 'pink', 'lavender', 'violet']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
# plt.title("Member 1 1880 - 1950 Temperature Anomalies")
# plt.show()

#Box and Whisker Plot Anomalies (1950 - 2000)
# jan_anom_bx2 = (reg_1950_2000_anom[0])
# feb_anom_bx2 = (reg_1950_2000_anom[1])
# mar_anom_bx2 = (reg_1950_2000_anom[2])
# apr_anom_bx2 = (reg_1950_2000_anom[3])
# may_anom_bx2 = (reg_1950_2000_anom[4])
# june_anom_bx2 = (reg_1950_2000_anom[5])
# july_anom_bx2 = (reg_1950_2000_anom[6])
# aug_anom_bx2 = (reg_1950_2000_anom[7])
# sept_anom_bx2 = (reg_1950_2000_anom[8])
# oct_anom_bx2 = (reg_1950_2000_anom[9])
# nov_anom_bx2 = (reg_1950_2000_anom[10])
# dec_anom_bx2 = (reg_1950_2000_anom[11])
# box = plt.boxplot([jan_anom_bx2, feb_anom_bx2, mar_anom_bx2, apr_anom_bx2, may_anom_bx2, june_anom_bx2, july_anom_bx2, aug_anom_bx2, sept_anom_bx2, oct_anom_bx2, nov_anom_bx2, dec_anom_bx2], patch_artist=True, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black","markersize":"6"}, boxprops=dict(facecolor="black", color="black"),capprops=dict(color="black"),whiskerprops=dict(color="black"),flierprops=dict(color="black", markeredgecolor="black"),medianprops=dict(color="black"),labels=["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"])
# colors = ['aquamarine', 'lightblue', 'lightgreen', 'tan', 'orchid', 'coral', 'gold', 'teal', 'magenta', 'pink', 'lavender', 'violet']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
# plt.title("Member 1 1950 - 2000 Temperature Anomalies")
# plt.show()

#Box and Whisker Plot Anomalies (2025 - 2100)
# jan_anom_bx3 = (reg_2025_2100_anom[0])
# feb_anom_bx3 = (reg_2025_2100_anom[1])
# mar_anom_bx3 = (reg_2025_2100_anom[2])
# apr_anom_bx3 = (reg_2025_2100_anom[3])
# may_anom_bx3 = (reg_2025_2100_anom[4])
# june_anom_bx3 = (reg_2025_2100_anom[5])
# july_anom_bx3 = (reg_2025_2100_anom[6])
# aug_anom_bx3 = (reg_2025_2100_anom[7])
# sept_anom_bx3 = (reg_2025_2100_anom[8])
# oct_anom_bx3 = (reg_2025_2100_anom[9])
# nov_anom_bx3 = (reg_2025_2100_anom[10])
# dec_anom_bx3 = (reg_2025_2100_anom[11])
# box = plt.boxplot([jan_anom_bx3, feb_anom_bx3, mar_anom_bx3, apr_anom_bx3, may_anom_bx3, june_anom_bx3, july_anom_bx3, aug_anom_bx3, sept_anom_bx3, oct_anom_bx3, nov_anom_bx3, dec_anom_bx3], patch_artist=True, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black","markersize":"6"}, boxprops=dict(facecolor="black", color="black"),capprops=dict(color="black"),whiskerprops=dict(color="black"),flierprops=dict(color="black", markeredgecolor="black"),medianprops=dict(color="black"),labels=["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"])
# colors = ['aquamarine', 'lightblue', 'lightgreen', 'tan', 'orchid', 'coral', 'gold', 'teal', 'magenta', 'pink', 'lavender', 'violet']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
# plt.title("Member 1 2025 - 2100 Temperature Anomalies")
# plt.show()

#Import text files and turn into a Pandas DataFrame
# df_text = pd.read_csv('1001.001_merged.nc_stats.txt', delimiter = "\t", header=None)

#Loop through all 90 members and output as netcdf
# for x in mem_string:
#     data = xr.open_mfdataset(PATH+'*'+x+'*.nc',combine = 'by_coords', concat_dim="time")
#     data.to_netcdf(x+'_merged.nc')

#############################################################################################################
#Calculate ENSO events based on the intensity (my creation) ######################################################
#Weak = <1.5, Moderate = 1.5 - 1.9, Strong = 2.0 - 2.4, Very Strong = >2.5
#DOESNT WORK

# weak_nino = df_anom[(df_anom['Anomaly'] <1.5) & (df_anom['Anomaly'] >=0.5)]
# weak_nina = df_anom[(df_anom['Anomaly'] >=-1.5) & (df_anom['Anomaly'] <=-0.5)]

# mod_nino = df_anom[(df_anom['Anomaly'] >=1.5) & (df_anom['Anomaly'] <=1.9 )]
# mod_nina = df_anom[(df_anom['Anomaly'] <=-1.5) & (df_anom['Anomaly'] >=-1.9)]
                   
# strong_nino = df_anom[(df_anom['Anomaly'] >=2.0) & (df_anom['Anomaly'] <=2.4)]
# strong_nina = df_anom[(df_anom['Anomaly'] <=-2.0) & (df_anom['Anomaly'] >=-2.4)]

# vstrong_nino = df_anom[(df_anom['Anomaly'] >=2.5)]
# vstrong_nina = df_anom[(df_anom['Anomaly'] <=-2.5)]

# weak_warm_array = weak_nino.to_numpy().flatten()
# weak_cold_array = weak_nina.to_numpy().flatten()
# weak_enso = np.concatenate((weak_warm_array, weak_cold_array))
# mod_warm_array = mod_nino.to_numpy().flatten()
# mod_cold_array = mod_nina.to_numpy().flatten()
# mod_enso = np.concatenate((mod_warm_array, mod_cold_array))
# strong_warm_array = strong_nino.to_numpy().flatten()
# strong_cold_array = strong_nina.to_numpy().flatten()
# strong_enso = np.concatenate((strong_warm_array, strong_cold_array))
# vstrong_warm_array = vstrong_nino.to_numpy().flatten()
# vstrong_cold_array = vstrong_nina.to_numpy().flatten()
# vstrong_enso = np.concatenate((vstrong_warm_array, vstrong_cold_array))

# #Plug in for functions
# weak_warm = find_events(weak_enso, '>', 0.49, 5, [-3,3])
# weak_cold = find_events(weak_enso, '<', -0.49, 5, [-3, 3])
# weak_w_c = find_enso_threshold(weak_enso, 0.49, -0.49)
# weak_wc_percent = find_enso_percentile(weak_enso,10)

# mod_warm = find_events(mod_enso, '>', 1.4, 5, [-3,3])
# mod_cold = find_events(mod_enso, '<', -1.4, 5, [-3,3])
# mod_w_c = find_enso_threshold(mod_enso, 1.4, -1.4)
# mod_wc_percent = find_enso_percentile(mod_enso,10)

# strong_warm = find_events(strong_enso, '>', 1.9, 5, [-3,3])
# strong_cold = find_events(strong_enso, '<', -1.9, 5, [-3,3])
# strong_w_c = find_enso_threshold(strong_enso, 1.9, -1.9)
# strong_wc_percent = find_enso_percentile(strong_enso,10)

# vstrong_warm = find_events(vstrong_enso, '>', 2.4, 5, [-3,3])
# vstrong_cold = find_events(vstrong_enso, '<', -2.4, 5, [-3,3])
# vstrong_w_c = find_enso_threshold(vstrong_enso, 2.4, -2.4)
# vstrong_wc_percent = find_enso_percentile(vstrong_enso,10)


#Member 2 Information

# PATH='/Users/taylordolan/Documents/Member_2'
# MEMBER_two='1021.002'
# ds_two = xr.open_mfdataset(PATH+'*'+MEMBER_two+'*.nc', combine = 'by_coords', concat_dim="time")

# temp_two = ds_two['SST'][:,90:102, 152:193]
# SST_AVG_two = temp_two.mean(dim=['lat', 'lon']).values
# time_two = ds_two['time'].values 

# t_two = ds_two.indexes['time'].to_datetimeindex()
# df_time_two = pd.DataFrame(t_two, columns = ['time'])
# df_datetime_two = pd.to_datetime(t_two)
# new_datetime_two = df_datetime_two - pd.DateOffset(months=n)
# slice_datetime_two = new_datetime_two[360::]

# SST_ANOM_two = np.empty([2652])

# for j in range(len(SST_ANOM_two)):
#     vals=j+offset-index+(15*12)
#     if np.max(vals) >= len(SST_AVG_two):
#         vals=(j+offset)-index
#     SST_ANOM_two[j]=SST_AVG_two[j+offset]-np.mean(SST_AVG_two[vals])
#     #print(j,j+offset,vals)

# SST_FINAL_two = moving_average(SST_ANOM_two,3)

# df_anom_two = pd.DataFrame(SST_FINAL_two, columns = ['Anomaly']) #df with SST_FINAL
# df_slice_time_two = pd.DataFrame(slice_datetime_two, columns = ['Date']) #df with slice time
# df_com_two = pd.concat([df_anom_two, df_slice_time_two], axis=1, join='inner')

# neutral_df_two = df_com_two[(df_com_two['Anomaly'] <= .4) & (df_com_two['Anomaly'] >= -.4)]
# nino_df_two = df_com_two[(df_com_two['Anomaly'] >= .5)]
# nina_df_two = df_com_two[(df_com_two['Anomaly'] <= -.5)]

# mem_2_warm_extrema = find_events(SST_FINAL_two, '>', 0.49, 5, [-3, 3])
# mem_2_cold_extrema = find_events(SST_FINAL_two, '<', -0.49, 5, [-3, 3])
# warm_cold_mem_2 = find_enso_threshold(SST_FINAL_two, 0.49, -0.49)

# months = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
# years = np.arange(1880,2101,1,dtype=int)
# SST_resize_two = np.resize(SST_FINAL_two,(221,12))
# SST_resize_df_two = pd.DataFrame(SST_resize_two)
# SST_resize_df_two.columns = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
# SST_resize_df_two.index = years

# #Export as an Excel Sheet
# SST_resize_df_two.to_excel("ENSO_table_mem2.xlsx",
#               sheet_name='ENSO_table_mem2') 

# neutral_df_two = df_com_two[(df_com_two['Anomaly'] <= .4) & (df_com_two['Anomaly'] >= -.4)]
# nino_df_two = df_com_two[(df_com_two['Anomaly'] >= .5)]
# nina_df_two = df_com_two[(df_com_two['Anomaly'] <= -.5)]

#PLOT
# fig = plt.figure(figsize=(10, 6),dpi=600)  
# plt.plot(slice_datetime_two,SST_FINAL_two,color='black')
# plt.plot(nino_df_two.Date, nino_df_two.Anomaly, color = 'r')
# plt.plot(nina_df_two.Date, nina_df_two.Anomaly, color = 'b')
# plt.plot(neutral_df_two.Date, neutral_df_two.Anomaly, color = 'k')
# plt.axhline(y=0.5, color='black', linestyle='--')
# plt.axhline(y=-0.5, color='black', linestyle='--') 
# plt.title('ENSO Anomalies Member 2')
# plt.xlabel('Years')
# plt.ylabel('ENSO Anomaly')

# #Member 2 PLOT
# fig = plt.figure(figsize=(16,3),dpi=600)  
# # plt.plot(slice_datetime,SST_FINAL,color='black',)
# plt.fill_between(slice_datetime_two, SST_FINAL_two,el_nino,
#                   where=(SST_FINAL_two >= el_nino),
#                   alpha=0.30, color='red', interpolate=True)
# plt.fill_between(slice_datetime_two, SST_FINAL_two, la_nina,
#                   where=(SST_FINAL_two <= la_nina),
#                   alpha=0.30, color='blue', interpolate=True)
# plt.axhline(y=0.5, color='black', linestyle='--')
# plt.axhline(y=-0.5, color='black', linestyle='--') 
# plt.title('ENSO Timeseries Member 2')
# plt.xlabel('Years')
# plt.ylabel('Standard Departure (°C)')
# plt.show()

#Subplots for Member 1 & 2 
# plt.subplot(1, 2, 1)
# plt.plot(slice_datetime,SST_FINAL,color='black')
# plt.plot(nino_df.Date, nino_df.Anomaly, color = 'r')
# plt.plot(nina_df.Date, nina_df.Anomaly, color = 'b')
# plt.plot(neutral_df.Date, neutral_df.Anomaly, color = 'k')
# plt.axhline(y=0.5, color='black', linestyle='--')
# plt.axhline(y=-0.5, color='black', linestyle='--') 
# plt.title('ENSO Anomalies Member 1')
# plt.xlabel('Years')
# plt.ylabel('ENSO Anomaly')

# plt.subplot(1, 2, 2) # index 2
# plt.plot(slice_datetime_two,SST_FINAL_two,color='black')
# plt.plot(nino_df_two.Date, nino_df_two.Anomaly, color = 'r')
# plt.plot(nina_df_two.Date, nina_df_two.Anomaly, color = 'b')
# plt.plot(neutral_df_two.Date, neutral_df_two.Anomaly, color = 'k')
# plt.axhline(y=0.5, color='black', linestyle='--')
# plt.axhline(y=-0.5, color='black', linestyle='--') 
# plt.title('ENSO Anomalies Member 2')
# plt.xlabel('Years')
# plt.show()

##########################################################################################
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=50))
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# plt.gcf().autofmt_xdate()
# plt.plot(SST_ANOM)
# plt.plot(np.roll(SST_FINAL,1),slice_datetime,color="red")
# plt.plot([0,2652],[0.5, 0.5],color="black",linestyle="-")
# plt.plot([0,2652],[-0.5, -0.5],color="black",linestyle="-")

# #How many times each event occurred
# count_nino = df_com[df_com['Anomaly'] >= 0.5].groupby('Date')['Anomaly'].count() #el nino
# count_nina = df_com[df_com['Anomaly'] <=-0.5].groupby('Date')['Anomaly'].count() #la nina

# #Sort enso anomalies in groups
# neutral_df = df_com[(df_com['Anomaly'] <= .4) & (df_com['Anomaly'] >= -.4)]
# nino_df = df_com[(df_com['Anomaly'] >= .5)]
# nina_df = df_com[(df_com['Anomaly'] <= -.5)]

#practice
# plot = plt.scatter(slice_datetime, SST_FINAL, c=c, s= 5,cmap='coolwarm')
# plot.set_clim(-5.0, 5.0)
# fig.colorbar(plot)
# plt.grid(True, 'both')
# plt.plot(slice_datetime,SST_FINAL)
# plt.show()

# dt = neutral_df['Date']
# dt['year'] = dt['Date'].dt.year
# dt['month'] = dt['Date'].dt.month

# dt.resample('1M')
# # month = pd.Timedelta('30d')
# # month = np.timedelta64(1, 'M')
# month = pd.offsets.MonthBegin(0)
# # in_block = ((dt - dt.shift(-31)).abs() == month) | (dt.diff() == month)
# # filt = neutral_df.loc[in_block]
# breaks = dt.diff() != month
# groups = breaks.cumsum()
# # for neutral_df in filt.groupby(groups):
# #      print(neutral_df, end='\n\n')

# neutral_df['groups'] = groups

# def is_at_least_five_consec(month_diff):
#     consec_count = 0
#     #print(month_diff)
#     for index , val in enumerate(month_diff):
#         if index != 0 and val == 1:
#                 consec_count += 1
#                 if index != 0 and val == 2:
#                     consec_count += 1
#                     if val == 3:
#                       consec_count += 1
#                       if consec_count == 4:
#                           return True
         
#Get ENSO phases

#Neutral Phase
# g = neutral_df.groupby([neutral_df.year])

# def is_at_least_three_consec(month_diff):
#     consec_count = 0
#     #print(month_diff)
#     for index , val in enumerate(month_diff):
#         if index != 0 and val == 1:
#                 consec_count += 1
#                 if consec_count == 2:
#                     return True
        
# res = g.filter(lambda x : is_at_least_three_consec(x['month'].diff().values.tolist()))
# consecutives = res['month'].diff().ne(0).cumsum()
# res['Cumulative'] = res["month"].diff(1) #probs not right way to do it
# res['dup'] = res['Cumulative'].duplicated()
# res.groupby((res['Cumulative'].shift() != res['Cumulative']).cumsum())
# # res.groupby('Cumulative').filter(lambda x: len(x) >= 5)  #MAYBE
# res[res['Cumulative'].isin(res['Cumulative'].value_counts()[res['Cumulative'].value_counts()>=5].index)]
# v = res.Cumulative.value_counts()
# res[res.Cumulative.isin(v.index[v.gt(5)])]
# res[res.groupby('Cumulative')['Cumulative'].transform('size') >= 5]
# more_than_5 = res['Cumulative'].value_counts()
# more_than_5 = list(more_than_5[more_than_5>=5].index)
# more_than_5_rows = res[res['Cumulative'].isin(more_than_5)]
# res.groupby('Cumulative').filter(lambda x: len(x) >= 5)  #MAYBE

          
#Possible Methods
# res['consec'] = res.groupby('month').Date.diff().dt.days.ne(31).cumsum()
# # s = res.groupby('month').Date.diff().dt.days.ne(30).cumsum()
# res.groupby(['month']).size().reset_index(drop=True)
# res['Phase'] = res["month"].diff(-1).le(-1)
# res['Sum'] = res['Cumulative'].cumsum()
# res['consec'] = res.month.groupby([res.month.diff().ne(0).cumsum()]).transform('size').ge(5).astype(int)
# res['Cumulative'] = res.groupby('month')['month'].cumsum() #close!
# res['Cumulative'] = res.groupby('month')['month'].diff()
# res.loc[mask, 'Cumulative'] += res.groupby('Cumulative').cumcount()
# res['Neutral'] = res['Cumulative'] <=-1
# res['Cumulative'] = res.groupby('month')['month'].cumsum().lt(5).astype(int)