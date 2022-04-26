#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:52:22 2022
@author: taylordolan

Code to loop through all available members and output each as a separate dataframe
to analyze each member (final thing to do for ENSO)

Members being used: 1001, 1021, 1041, 1061, 1081, 1101, 1121, 1141, 1161, 1181,
                    1231.001 - 1231.020, 1251.001 - 1251.010, 1281.001 - 1281.020,
                    and 1301.001 - 1301.020

Total number of members: 80
Missing members: 1251.011 - 1251.020, MOAR (last 10 members)

"""
#Imports
import glob as glob
import xarray as xr
import numpy as np
import pandas as pd
import csv
import os

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
###########################################################################################

# PATH='/storage/taylor/data_complete/'
PATH = '/Users/taylordolan/Documents/'

for file in glob.glob(PATH+'*_merged.nc'):
    ds = xr.open_dataset(file)
    temp = ds['SST'][:,90:102, 152:193]
    SST_AVG=temp.mean(dim=['lat', 'lon']).values
    time = ds['time'].values  
    t = ds.indexes['time'].to_datetimeindex()
    df_time = pd.DataFrame(t, columns = ['time'])
    df_datetime = pd.to_datetime(t)
    n=1
    new_datetime = df_datetime - pd.DateOffset(months=n)
    slice_datetime = new_datetime[360::]
    year_start=1880
    year_end=2100
    index=12*np.arange(30)
    offset=12*(year_start-1850)
    SST_ANOM = np.empty([2652])
    np.array([12*(year_end-year_start)])
    for j in range(len(SST_ANOM)):
        vals=j+offset-index+(15*12)
        if np.max(vals) >= len(SST_AVG):
            vals=(j+offset)-index
        SST_ANOM[j]=SST_AVG[j+offset]-np.mean(SST_AVG[vals])
    
    SST_FINAL = moving_average(SST_ANOM,3)

    warm_extrema = find_events(SST_FINAL, '>', 0.49, 5, [-3, 3])
    cold_extrema = find_events(SST_FINAL, '<', -0.49, 5, [-3, 3])
    warm_cold = find_enso_threshold(SST_FINAL, 0.49, -0.49)
    warm_cold_percentile = find_enso_percentile(SST_FINAL,10)
    test_warm_duration = compute_duration(SST_FINAL, np.less, warm_extrema[0], 0.49)
    test_cold_duration = compute_duration(SST_FINAL, np.greater, cold_extrema[0], -0.49)
    duration = enso_duration(SST_FINAL,10,1)
    
    warm_events = warm_cold_percentile[0]
    warm_peaks = warm_events['peaks']
    warm_peaks_list = warm_peaks.tolist()
    cold_events = warm_cold_percentile[1]
    cold_peaks = cold_events['peaks']
    warm_peak_array = np.array(warm_peaks)
    cold_peak_array = np.array(cold_peaks)
    
    warm_duration_locs = duration['warm_locs']
    warm_duration = duration['warm'] 
    warm_duration_array = np.array(warm_duration)
    time_warm_start = slice_datetime[warm_duration_locs]
    warm_duration_end = [] 
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
    for k in warm_peak_array:
        if k > 2.81:
            intensity.append(text3)
        elif k <2.41:
            intensity.append(text1)
        else:
            intensity.append(text2)


    for j in cold_peak_array:
        if j <= -2.7:
            intensity1.append(text3)
        elif j >= -2.33:
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
    # result.to_csv(file+'_Results_test.csv',index=False)
    


